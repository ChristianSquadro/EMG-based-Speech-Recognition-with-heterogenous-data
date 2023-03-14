import os
import sys
import numpy as np
import logging
import subprocess
from ctcdecode import CTCBeamDecoder
import jiwer
import random
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
import torch.nn.functional as F

from read_emg import EMGDataset, SizeAwareSampler
from architecture import Model
from data_utils import combine_fixed_length, decollate_tensor, combine_fixed_length_tgt
from transformer import TransformerEncoderLayer

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

    blank_id = len(testset.text_transform.chars)
    decoder = CTCBeamDecoder(testset.text_transform.chars+'_', blank_id=blank_id, log_probs_input=True,
            model_path='lm.binary', alpha=1.5, beta=1.85)

    dataloader = torch.utils.data.DataLoader(testset, batch_size=1)
    references = []
    predictions = []
    with torch.no_grad():
        for example in dataloader:
            X = example['emg'].to(device)
            X_raw = example['raw_emg'].to(device)
            sess = example['session_ids'].to(device)

            pred  = F.log_softmax(model(X, X_raw, sess), -1)

            beam_results, beam_scores, timesteps, out_lens = decoder.decode(pred)
            pred_int = beam_results[0,0,:out_lens[0,0]].tolist()

            pred_text = testset.text_transform.int_to_text(pred_int)
            target_text = testset.text_transform.clean_text(example['text'][0])

            references.append(target_text)
            predictions.append(pred_text)

    model.train()
    #remove empty strings because I had an error in the calculation of WER function
    predictions = [predictions[i] for i in range(len(predictions)) if len(references[i]) > 0]
    references = [references[i] for i in range(len(references)) if len(references[i]) > 0]
    return jiwer.wer(references, predictions)


def train_model(trainset, devset, device, writer, n_epochs=200, report_every=10):
    #Define Dataloader
    dataloader_training = torch.utils.data.DataLoader(trainset, pin_memory=(device=='cuda'), num_workers=0, collate_fn=EMGDataset.collate_raw, batch_sampler=SizeAwareSampler(trainset, 128000))
    dataloader_evaluation = torch.utils.data.DataLoader(devset, batch_size=1)

    #Define model and loss function
    n_chars = len(devset.text_transform.chars)
    model = Model(devset.num_features, n_chars+1).to(device)
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
            X = combine_fixed_length(example['emg'], 200).to(device)
            X_raw = combine_fixed_length(example['raw_emg'], 200*8).to(device)
            sess = combine_fixed_length(example['session_ids'], 200).to(device)
            y = combine_fixed_length_tgt(example['text_int'], X_raw.shape[0]).to(device)

            #Shifting target for input decoder and loss
            tgt= y[:,:-1]
            target= y[:,1:]

            #Prediction
            pred = model(X, X_raw, tgt, sess)

            #Primary Loss
            pred=pred.permute(0,2,1)
            loss = loss_fn(pred, target)

            #Auxiliary Loss
            #pred = F.log_softmax(pred, 2)
            #pred = nn.utils.rnn.pad_sequence(decollate_tensor(pred, example['text_int_lengths']), batch_first=False) # seq first, as required by ctc
            #y = nn.utils.rnn.pad_sequence(example['text_int'], batch_first=True).to(device)
            #loss = F.ctc_loss(pred, y, example['text_int_lengths'], example['text_int_lengths'], blank=n_chars)
            losses.append(loss.item())
            train_loss += loss.item()

            loss.backward()
            if (batch_idx+1) % 2 == 0:
                optim.step()
                optim.zero_grad()

            if batch_idx % report_every == report_every - 2:     
                #Evaluation
                model.eval()
                with torch.no_grad():
                    for example, idx in zip(dataloader_evaluation, range(len(dataloader_evaluation))):
                        X_raw = example['raw_emg'].to(device)
                        sess = example['session_ids'].to(device)
                        y = example['text_int'].to(device)

                        #Shifting target for input decoder and loss
                        tgt= y[:,:-1]
                        target= y[:,1:]

                        #Prediction without the 197-th batch because of missing label
                        if idx != 197:
                            pred = model(X, X_raw, tgt, sess)
                            #Primary Loss
                            pred=pred.permute(0,2,1)
                            loss = loss_fn(pred, target)
                            eval_loss += loss.item()

                #Writing on tensorboard
                writer.add_scalar('Loss/Evaluation', eval_loss / batch_idx, batch_idx)
                writer.add_scalar('Loss/Training', train_loss / batch_idx, batch_idx) 
                train_loss= 0
                eval_loss= 0

            #Increment counter        
            batch_idx += 1

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
