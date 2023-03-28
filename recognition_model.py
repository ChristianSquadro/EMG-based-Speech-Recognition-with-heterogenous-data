import os
import sys
import numpy as np
import logging
import subprocess
from ctcdecode import CTCBeamDecoder, DecoderState
import jiwer
import random
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
import torch.nn.functional as F

from read_emg import EMGDataset, SizeAwareSampler
from architecture import Model
from data_utils import combine_fixed_length, decollate_tensor

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_string('output_directory', 'output', 'where to save models and outputs')
flags.DEFINE_integer('batch_size', 32, 'training batch size')
flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
flags.DEFINE_integer('learning_rate_warmup', 1000, 'steps of linear warmup')
flags.DEFINE_integer('learning_rate_patience', 5, 'learning rate decay patience')
flags.DEFINE_string('start_training_from', None, 'start training from this model')
flags.DEFINE_float('l2', 0, 'weight decay')
flags.DEFINE_string('evaluate_saved', None, 'run evaluation on given model file')

def test(model, testset, device):
    model.eval()

    blank_id = 1
    decoder = CTCBeamDecoder(testset.text_transform.chars, blank_id=blank_id, log_probs_input=True,
            model_path='lm.binary', alpha=1.5, beta=1.85)
    state = DecoderState(decoder)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1)
    references = []
    predictions = []
    batch_idx = 0
    with torch.no_grad():
        for example in dataloader:
            X_raw = nn.utils.rnn.pad_sequence(example['raw_emg'], batch_first=True).to(device)
            tgt = nn.utils.rnn.pad_sequence(example['text_int'], batch_first=True).to(device)

            #Prediction without the 197-th batch because of missing label
            if batch_idx != 181:
                out_enc, out_dec = model(X_raw, tgt)
                pred  = F.log_softmax(out_dec, -1)

                beam_results, beam_scores, timesteps, out_lens = decoder.decode(pred)
                pred_int = beam_results[0,0,:out_lens[0,0]].tolist()

                pred_text = testset.text_transform.int_to_text(pred_int)
                target_text = testset.text_transform.clean_text(example['text'][0])

                references.append(target_text)
                predictions.append(pred_text)

        batch_idx += 1
        
    model.train()
    #remove empty strings because I had an error in the calculation of WER function
    predictions = [predictions[i] for i in range(len(predictions)) if len(references[i]) > 0]
    references = [references[i] for i in range(len(references)) if len(references[i]) > 0]
    return jiwer.wer(references, predictions)


def train_model(trainset, devset, device, writer, n_epochs=200, report_every=1, alpha=0.7):
    #Define Dataloader
    dataloader_training = torch.utils.data.DataLoader(trainset, pin_memory=(device=='cuda'), num_workers=0, collate_fn=EMGDataset.collate_raw, batch_size=2)
    dataloader_evaluation = torch.utils.data.DataLoader(devset, batch_size=1)

    #Define model and loss function
    n_chars = len(devset.text_transform.chars)
    model = Model(devset.num_features, n_chars + 1, device, True).to(device)
    loss_fn=nn.CrossEntropyLoss(ignore_index=0)

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from)
        model.load_state_dict(state_dict, strict=False)

    #Define optimizer and scheduler for the learning rate
    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[125,150,175], gamma=.5)

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = FLAGS.learning_rate
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration*target_lr/FLAGS.learning_rate_warmup)

    batch_idx = 0
    train_loss= 0
    eval_loss = 0
    optim.zero_grad()
    for epoch_idx in range(n_epochs):
        model.train()
        losses = []
        for example in dataloader_training:
            schedule_lr(batch_idx)

            #Preprosessing of the input and target for the model
            X_raw = nn.utils.rnn.pad_sequence(example['raw_emg'], batch_first=True).to(device)
            y = nn.utils.rnn.pad_sequence(example['text_int'], batch_first=True).to(device)

            #Shifting target for input decoder and loss
            tgt= y[:,:-1]
            target= y[:,1:]

            #Prediction
            out_enc, out_dec = model(X_raw, tgt)

            #Decoder Loss
            out_dec=out_dec.permute(0,2,1)
            loss_dec = loss_fn(out_dec, target)

            #Encoder Loss
            out_enc = F.log_softmax(out_enc, 2)
            out_enc = out_enc.transpose(1,0)
            loss_enc = F.ctc_loss(out_enc, y, example['lengths'], example['text_int_lengths'], blank = len(devset.text_transform.chars)+1) 

            #Combination the two losses
            loss = (1 - alpha) * loss_dec + alpha * loss_enc
            losses.append(loss.item())
            train_loss += loss.item()

            #Gradient Update
            loss.backward()
            if (batch_idx+1) % 2 == 0:
                optim.step()
                optim.zero_grad()
            
            
            #Increment counter and print the loss training       
            batch_idx += 1
            writer.add_scalar('Loss/Training', train_loss / batch_idx, batch_idx)
            train_loss= 0


            if batch_idx % report_every == 0:     
                #Evaluation
                model.eval()
                with torch.no_grad():
                    for idx, example in enumerate(dataloader_evaluation):
                        X_raw = nn.utils.rnn.pad_sequence(example['raw_emg'], batch_first=True).to(device)
                        y = nn.utils.rnn.pad_sequence(example['text_int'], batch_first=True).to(device)
                    
                        #Shifting target for input decoder and loss
                        tgt= y[:,:-1]
                        target= y[:,1:]

                        #Prediction without the 197-th batch because of missing label
                        out_enc, out_dec = model(X_raw, tgt)
                        #Decoder Loss
                        out_dec=out_dec.permute(0,2,1)
                        loss = loss_fn(out_dec, target)
                        eval_loss += loss.item()
                        
                        #just for now
                        if idx == 10:
                            break

                #Writing on tensorboard
                writer.add_scalar('Loss/Evaluation', eval_loss / batch_idx, batch_idx)
                eval_loss= 0

        #Testing and change learning rate
        val = test(model, devset, device)
        writer.add_scalar('WER/Evaluation',val, batch_idx)
        lr_sched.step()
    
        #Logging
        train_loss = np.mean(losses)
        logging.info(f'finished epoch {epoch_idx+1} - training loss: {train_loss:.4f} validation WER: {val*100:.2f}')
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory,'model.pt'))

    model.load_state_dict(torch.load(os.path.join(FLAGS.output_directory,'model.pt'))) # re-load best parameters
    return model

def evaluate_saved():
    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    testset = EMGDataset(test=True)
    n_chars = len(testset.text_transform.chars)
    model = Model(testset.num_features, n_chars+1).to(device)
    model.load_state_dict(torch.load(FLAGS.evaluate_saved))
    print('WER:', test(model, testset, device))

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    logging.info(subprocess.run(['git','rev-parse','HEAD'], stdout=subprocess.PIPE, universal_newlines=True).stdout)
    logging.info(subprocess.run(['git','diff'], stdout=subprocess.PIPE, universal_newlines=True).stdout)

    logging.info(sys.argv)

    trainset = EMGDataset(dev=False,test=False)
    devset = EMGDataset(dev=True)
    logging.info('output example: %s', devset.example_indices[0])
    logging.info('train / dev split: %d %d',len(trainset),len(devset))

    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    writer = SummaryWriter(log_dir="./content/runs")

    model = train_model(trainset, devset ,device, writer)

if __name__ == '__main__':
    FLAGS(sys.argv)
    if FLAGS.evaluate_saved is not None:
        evaluate_saved()
    else:
        main()
