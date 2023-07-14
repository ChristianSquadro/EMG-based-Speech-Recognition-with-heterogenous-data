import datetime
import os
import sys
import numpy as np
import logging
import jiwer
import PrefixTree
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
import torch.nn.functional as F

from read_emg import EMGDataset, DynamicBatchSampler
from architecture import Model
from BeamSearch import run_single_bs
from greedy_search import run_greedy
from greedy_search import run_greedy

from absl import flags
FLAGS = flags.FLAGS

#Settings
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_string('evaluate_saved_beam_search', None, 'run beam_evaluation on given model file')
flags.DEFINE_string('evaluate_saved_greedy_search', None, 'run greedy_evaluation on given model file')
flags.DEFINE_string('start_training_from', None, 'start training from this model')

#File model paths
flags.DEFINE_string('output_directory', 'output', 'where to save models and outputs')
flags.DEFINE_string('phonesSet', "descriptions/phonesSet", 'the set of all phones in the lexicon')
flags.DEFINE_string('vocabulary', "descriptions/vocabulary", 'the set of all words in the lexicon')
flags.DEFINE_string('dict', "descriptions/dgaddy-lexicon.txt", 'the pronunciation dictionary')
flags.DEFINE_string('lang_model', "descriptions/lm.binary", 'the language model')

#Parameters
flags.DEFINE_integer('pad', 42, 'Padding value according to the position on phoneme inventory')
flags.DEFINE_integer('report_PER', 1, "How many epochs to report PER")
flags.DEFINE_integer('report_loss', 50, "How many step train to report plots")

#Hyperparameters
flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
flags.DEFINE_integer('learning_rate_warmup', 1000, 'steps of linear warmup')
flags.DEFINE_float('l2', 0., 'weight decay')
flags.DEFINE_float('alpha_loss', 0.3, 'parameter alpha for the two losses')
flags.DEFINE_float('gradient_clipping', 1.0, 'parameter for tuning clipping')
flags.DEFINE_integer('steps_grad', 3, 'steps of gradient accumulation')
flags.DEFINE_integer('n_epochs', 100, 'number of epochs')
flags.DEFINE_integer('n_buckets', 32, 'number of buckets in the dataset')

def train_model(trainset, devset, device, writer):    
    def training_loop():
        nonlocal batch_idx, train_loss, train_dec_loss, train_enc_loss, run_train_steps

        #Warmup phase methods
        def set_lr(new_lr):
            for param_group in optim.param_groups:
                param_group['lr'] = new_lr
        target_lr = FLAGS.learning_rate
        def schedule_lr(iteration):
            iteration = iteration + 1
            if iteration <= FLAGS.learning_rate_warmup:
                set_lr(iteration*target_lr/FLAGS.learning_rate_warmup)
        
        #Training loop 
        model.train()   
        optim.zero_grad()    
        for step,example in enumerate(dataloader_training):
            
            #Schedule_lr to change learning rate during the warmup phase
            schedule_lr(batch_idx)
            
            #Preprosessing of the input and target for the model
            X_raw=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True, padding_value= FLAGS.pad).to(device)
            y = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True,  padding_value= FLAGS.pad).to(device)
            
            #To take into account the mean of the batch dimension
            print(len(X_raw))

            
            #Shifting target for input decoder and loss
            tgt= y[:,:-1]
            target= y[:,1:]

            #Prediction
            out_enc, out_dec = model(x_raw=X_raw, y=tgt)

            #Encoder Loss
            out_enc = F.log_softmax(out_enc, 2)
            out_enc = out_enc.transpose(1,0)
            loss_enc = F.ctc_loss(out_enc, y, example['lengths'], example['phonemes_int_lengths'], blank = n_phones) 
            
            
            #Decoder Loss
            out_dec=out_dec.permute(0,2,1)
            loss_dec = loss_fn(out_dec, target)

            #Combination the two losses
            loss = (1 - FLAGS.alpha_loss) * loss_dec + FLAGS.alpha_loss * loss_enc
            losses.append(loss.item())
            train_loss += loss.item()
            train_dec_loss += loss_dec.item()
            train_enc_loss += loss_enc.item()


            #Gradient Update
            loss.backward()
            #Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.gradient_clipping)
            if (step + 1) % FLAGS.steps_grad == 0:
                optim.step()
                optim.zero_grad()

            #Increment counter batch and counter for steps of losses   
            batch_idx += 1
            run_train_steps += 1
        
            if batch_idx % FLAGS.report_loss == 0:  
                evaluation_loop() 
                report_loss()
            
  
    def evaluation_loop():
        nonlocal eval_loss, eval_dec_loss, eval_enc_loss, run_eval_steps, predictions_eval, references_eval, predictions_train, references_train

        #Evaluation loop
        model.eval()
        with torch.no_grad():
            for step,example in enumerate(dataloader_evaluation):
                torch.cuda.empty_cache()
                #Collect the data
                X_raw=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True,  padding_value= FLAGS.pad).to(device)
                y = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True, padding_value=FLAGS.pad).to(device)
            
                #Forward Model using Greedy Approach not teacher forcing
                tgt= y[:,:-1]
                target= y[:,1:]
                out_enc, out_dec = model(x_raw=X_raw, y=tgt)

                #Encoder Loss
                out_enc = F.log_softmax(out_enc, 2)
                out_enc = out_enc.transpose(1,0)
                loss_enc = F.ctc_loss(out_enc, y, example['lengths'], example['phonemes_int_lengths'], blank = n_phones) 
                    
                #Decoder Loss
                out_dec=out_dec.permute(0,2,1)
                loss_dec = loss_fn(out_dec, target)

                #Combination the two losses
                loss = (1 - FLAGS.alpha_loss) * loss_dec + FLAGS.alpha_loss * loss_enc
                eval_loss += loss.item()
                eval_dec_loss += loss_dec.item()
                eval_enc_loss += loss_enc.item()
                run_eval_steps += 1

                #Pick up just 10 batches
                if step + 1 == 10:
                    break

        
    def report_loss():
        nonlocal batch_idx,train_loss,train_dec_loss,train_enc_loss,eval_loss,eval_dec_loss,eval_enc_loss,run_train_steps,run_eval_steps

        #Print training loss
        writer.add_scalar('Loss/Training', round(train_loss / run_train_steps,3), batch_idx)
        writer.add_scalar('Loss_Decoder/Training', round(train_dec_loss / run_train_steps,3), batch_idx)
        writer.add_scalar('Loss_Encoder/Training', round(train_enc_loss / run_train_steps,3), batch_idx)
        writer.flush()
        
        #Reset variables for training
        train_loss= 0
        train_dec_loss=0
        train_enc_loss=0
        run_train_steps = 0
                    
        #Writing on tensorboard
        writer.add_scalar('Loss/Evaluation', round (eval_loss / run_eval_steps, 3), batch_idx)
        writer.add_scalar('Loss_Decoder/Evaluation', round (eval_dec_loss / run_eval_steps, 3), batch_idx)
        writer.add_scalar('Loss_Encoder/Evaluation', round (eval_enc_loss / run_eval_steps, 3), batch_idx)

        
        #Reset variables for evaluation
        eval_loss = 0
        eval_dec_loss = 0
        eval_enc_loss = 0
        run_eval_steps=0

    def report_PER():
        nonlocal predictions_train,references_train,predictions_eval,references_eval

        #Calculation PER
        model.eval()
        #Greedy Search Training Set Subset
        for step,example in enumerate(dataloader_training):
            torch.cuda.empty_cache()
            X_raw=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True,  padding_value= FLAGS.pad).to(device)
            y = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True, padding_value=FLAGS.pad).to(device)
            target= y[:,1:]
            phones_seq = run_greedy(model, X_raw, target, n_phones, device)
            predictions_train += phones_seq
            references_train += example['phonemes']
            #Pick up just 10 batches
            if step + 1 == 10:
                break
        
        #Greedy Search Evaluation Set
        for step, example in enumerate(dataloader_evaluation):
            torch.cuda.empty_cache()
            #Collect the data
            X_raw=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True,  padding_value= FLAGS.pad).to(device)
            y = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True, padding_value=FLAGS.pad).to(device)
        
            #Forward Model using Greedy Approach not teacher forcing
            target= y[:,1:]
            phones_seq = run_greedy(model, X_raw, target, n_phones, device)
            
            #Append lists to calculate the PER
            predictions_eval += phones_seq
            references_eval += example['phonemes']

        #Reporting PER
        logging.info(f'Prediction: {predictions_eval[0]} ---> Reference: {references_eval[0]}  (PER: {jiwer.wer(predictions_eval[0], references_eval[0])})')
        writer.add_scalar('PhonemeErrorRate/Training', jiwer.wer(references_train, predictions_train), batch_idx)
        writer.add_scalar('PhonemeErrorRate/Evaluation', jiwer.wer(references_eval, predictions_eval), batch_idx)
        writer.flush()
        predictions_train=[]
        references_train=[]
        predictions_eval=[]
        references_eval=[]


    ################################################### TRAINING MODEL BELOW ########################################################################       

    ##INITIALIZATION##
    
    #Buffer variables initizialiation
    batch_idx = 0; train_loss= 0; train_dec_loss= 0; train_enc_loss= 0; eval_loss = 0; eval_dec_loss = 0; eval_enc_loss = 0; run_train_steps=0; run_eval_steps=0; predictions_train=[]; references_train=[]; predictions_eval=[]; references_eval=[]; losses = []

    #Define Dataloader
    dynamicBatchTrainingSampler=DynamicBatchSampler(trainset, 138000, FLAGS.n_buckets, shuffle=True, batch_ordering='random')
    dynamicBatchEvaluationSampler=DynamicBatchSampler(devset, 138000, FLAGS.n_buckets, shuffle=True, batch_ordering='random')
    dataloader_training = torch.utils.data.DataLoader(trainset, pin_memory=(device=='cuda'), num_workers=0,collate_fn=EMGDataset.collate_raw, batch_sampler= dynamicBatchTrainingSampler)
    dataloader_evaluation = torch.utils.data.DataLoader(devset, pin_memory=(device=='cuda'), num_workers=0,collate_fn=EMGDataset.collate_raw, batch_sampler= dynamicBatchEvaluationSampler)
    
    #Define model and loss function
    n_phones = len(devset.phone_transform.phoneme_inventory)
    model = Model(devset.num_features, n_phones + 1, n_phones, device) # plus 1 for the blank symbol of CTC loss in the encoder
    model=nn.DataParallel(model).to(device)
    loss_fn=nn.CrossEntropyLoss(ignore_index=FLAGS.pad)

    #If it is enable and you want to start from a pre-trained model
    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from)
        model.load_state_dict(state_dict, strict=False)

    #Define optimizer and scheduler for the learning rate
    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2)
    #lr_sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[5,10,15], gamma=.5)
    
    ##MODEL TRAINING##
    
    optim.zero_grad()
    for epoch_idx in range(FLAGS.n_epochs):
        losses = []
        torch.cuda.empty_cache()
        training_loop()  
        if epoch_idx % FLAGS.report_PER == 0:  
            report_PER()
         
        #Change learning rate
        #lr_sched.step()
        #Mean of the main loss and logging
        logging.info(f'finished epoch {epoch_idx+1} - training loss: {np.mean(losses):.4f}')
        #Save Model
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory,'model.pt'))
        #Random shift batches
        dynamicBatchTrainingSampler.set_epoch(epoch_idx + 1)
        dynamicBatchEvaluationSampler.set_epoch(epoch_idx + 1)
        #Stop if Training loss reaches convergence
        if round(np.mean(losses), 1) == 0.0:
            break

    return model


def evaluate_saved_beam_search():
    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    #testset = EMGDataset(test=True)
    testset = EMGDataset(dev=False,test=False)
    n_phones = len(testset.phone_transform.phoneme_inventory)
    model = Model(testset.num_features, n_phones + 1, n_phones, device) #plus 1 for the blank symbol of CTC loss in the encoder
    model=nn.DataParallel(model).to(device)
    tree = PrefixTree.init_tree(FLAGS.phonesSet,FLAGS.vocabulary,FLAGS.dict)
    language_model = PrefixTree.init_language_model(FLAGS.lang_model)
    model.load_state_dict(torch.load(FLAGS.evaluate_saved_beam_search))
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1)
    references = []
    predictions = []
     
    with torch.no_grad():
        for example in dataloader:
            X_raw=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True, padding_value= FLAGS.pad).to(device)
            tgt = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True, padding_value= FLAGS.pad).to(device)

            target= tgt[:,1:]
            pred=run_single_bs(model,X_raw,target,n_phones,tree,language_model,device)
 
            pred_text = testset.text_transform.clean_text(' '.join(pred[2]))
            target_text = testset.text_transform.clean_text(example['text'][0])
            references.append(target_text)
            predictions.append(pred_text)
            
            logging.info(f'Prediction:{pred_text} ---> Reference:{target_text}  (WER: {jiwer.wer(target_text, pred_text)})')
        
    print('WER:', jiwer.wer(references, predictions))

def evaluate_saved_greedy_search():
    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    testset = EMGDataset(test=True)
    #testset = EMGDataset(dev=False,test=False)
    n_phones = len(testset.phone_transform.phoneme_inventory)
    model = Model(testset.num_features, n_phones + 1, n_phones, device) #plus 1 for the blank symbol of CTC loss in the encoder
    model=nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(FLAGS.evaluate_saved_greedy_search))
    dataloader = torch.utils.data.DataLoader(testset, shuffle=True , batch_size=1)
    references = []
    predictions = []

    with torch.no_grad():
        for idx, example in enumerate(dataloader):
            #Collect the data
            X_raw=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True,  padding_value= FLAGS.pad).to(device)
            y = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True, padding_value=FLAGS.pad).to(device)
        
            #Forward Model
            target= y[:,1:]
            phones_seq = run_greedy(model, X_raw, target, n_phones, device)

            #Append lists to calculate the PER
            predictions += phones_seq
            references += example['phonemes']

    print('WER:', jiwer.wer(references, predictions))

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    trainset = EMGDataset(dev=False,test=False)
    devset = EMGDataset(dev=True)
    logging.info('train / dev split: %d %d',len(trainset),len(devset))

    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    log_dir="logs/run/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    train_model(trainset, devset ,device, writer)

if __name__ == '__main__':
    FLAGS(sys.argv)
    if FLAGS.evaluate_saved_beam_search is not None:
        evaluate_saved_beam_search()
    elif FLAGS.evaluate_saved_greedy_search is not None:
        evaluate_saved_greedy_search()
    else:
        main()
