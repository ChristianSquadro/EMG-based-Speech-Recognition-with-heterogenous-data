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
from LabelSmoothingLoss import LabelSmoothingLoss
from architecture import Model
from BeamSearch import run_single_bs
from greedy_search import run_greedy
from data_utils import load_dictionary,combine_fixed_length

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
flags.DEFINE_string('vocabulary', "descriptions/new_vocabulary", 'the set of all words in the lexicon')
flags.DEFINE_string('dict', "descriptions/new_dgaddy-lexicon.txt", 'the pronunciation dictionary')
flags.DEFINE_string('lang_model', "descriptions/lm.binary", 'the language model')

#Parameters
flags.DEFINE_integer('pad', 42, 'Padding value according to the position on phoneme inventory')
flags.DEFINE_integer('report_PER', 1, "How many epochs to report PER")
flags.DEFINE_integer('report_loss', 50, "How many step train to report plots")

#Hyperparameters
flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
flags.DEFINE_integer('learning_rate_warmup', 1500, 'steps of linear warmup')
flags.DEFINE_float('l2', 0., 'weight decay')
flags.DEFINE_float('threshold_alpha_loss', 0.05, 'threshold parameter alpha for the two losses')
flags.DEFINE_float('grad_clipping', 5.0 , 'parameter for gradient clipping')
flags.DEFINE_integer('batch_size_grad', 100, 'batch size for gradient accumulation')
flags.DEFINE_integer('n_epochs', 200, 'number of epochs')
flags.DEFINE_integer('n_buckets', 16, 'number of buckets in the dataset')
flags.DEFINE_integer('max_batch_length', 80000, 'maximum batch length')

def train_model(trainset, devset, device, writer):    
    def training_loop():
        nonlocal batch_idx, train_loss, train_dec_loss, train_enc_loss, run_train_steps, alpha_loss

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
        optim.zero_grad()  
        sum_batch_size=0  
        for step,example in enumerate(dataloader_training):
            #Set the model in train mode
            model.train()   
            
            #Schedule_lr to change learning rate during the warmup phase
            schedule_lr(batch_idx)
            
            #Preprosessing of the input and target for the model
            X=combine_fixed_length(example['raw_emg'], 200*8).to(device)
            #X=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True, padding_value= FLAGS.pad).to(device)
            y = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True,  padding_value= FLAGS.pad).to(device)
            
            #To take into account the length of the batch dimension and for gradient accumulation
            #print(len(X))
            sum_batch_size += len(X)

            
            #Shifting target for input decoder and loss
            tgt= y[:,:-1]
            target= y[:,1:]

            #Prediction
            out_enc, out_dec = model(example['lengths'], device, x_raw=X, y=tgt)

            #Encoder Loss
            out_enc = F.log_softmax(out_enc, 2)
            out_enc = out_enc.transpose(1,0)
            loss_enc = F.ctc_loss(out_enc, y, example['lengths'], example['phonemes_int_lengths'], blank = n_phones) 
            
            
            #Decoder Loss
            out_dec=out_dec.permute(0,2,1)
            loss_dec = loss_fn(out_dec, target)

                
            #Combination the two losses   
            loss = (1 - alpha_loss) * loss_dec + alpha_loss * loss_enc
            losses.append(loss.item())
            train_loss += loss.item()
            train_dec_loss += loss_dec.item()
            train_enc_loss += loss_enc.item()

            #Alternative Gradient Update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_clipping)
            if sum_batch_size >= FLAGS.batch_size_grad:
                optim.step()
                optim.zero_grad()
                sum_batch_size=0


            #Increment counter batch and counter for steps of losses   
            batch_idx += 1
            run_train_steps += 1
            
            #Run the model on evaluation set and report the loss
            if (step + 1) % FLAGS.report_loss == 0:  
                evaluation_loop()
                report_loss()
        
        
    def evaluation_loop():
        nonlocal eval_loss, eval_dec_loss, eval_enc_loss, run_eval_steps, predictions_eval, references_eval, predictions_train, references_train, alpha_loss

        #Evaluation loop
        model.eval()
        with torch.no_grad():
            for step,example in enumerate(dataloader_evaluation):
                torch.cuda.empty_cache()
                #Collect the data
                X=combine_fixed_length(example['raw_emg'], 200*8).to(device)
                #X=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True,  padding_value= FLAGS.pad).to(device)
                y = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True, padding_value=FLAGS.pad).to(device)
            
                #Forward Model 
                tgt= y[:,:-1]
                target= y[:,1:]
                out_enc, out_dec = model(example['lengths'] , device, x_raw=X, y=tgt)

                #Encoder Loss
                out_enc = F.log_softmax(out_enc, 2)
                out_enc = out_enc.transpose(1,0)
                loss_enc = F.ctc_loss(out_enc, y, example['lengths'], example['phonemes_int_lengths'], blank = n_phones) 
                    
                #Decoder Loss
                out_dec=out_dec.permute(0,2,1)
                loss_dec = loss_fn(out_dec, target)

                #Combination the two losses
                loss = (1 - alpha_loss) * loss_dec + alpha_loss * loss_enc
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
        nonlocal predictions_train,references_train,predictions_eval,references_eval,curr_eval_PER, text_eval, text_train
        running_total_train=0 ;running_correct_train=0; running_total_eval=0 ;running_correct_eval=0

        #Calculation PER
        model.eval()
        #Greedy Search Training Set Subset
        for step,example in enumerate(dataloader_training):
            torch.cuda.empty_cache()
            X=combine_fixed_length(example['raw_emg'], 200*8).to(device)
            #X=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True,  padding_value= FLAGS.pad).to(device)
            y = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True, padding_value=FLAGS.pad).to(device)
            
            target= y[:,1:]
            phones_seq, new_word_seq_idx = run_greedy(model, example['lengths'] ,X, target, n_phones, device)
            
            predictions_train += phones_seq
            references_train += example['phonemes']
            text_train += example['text']
            running_total_train += y.shape[0] * y.shape[1]
            running_correct_train += sum([sum(item) for item in torch.eq(new_word_seq_idx,y)]).item()
            #Pick up just 15 batches
            if step + 1 == 15:
                break
        
        #Greedy Search Evaluation Set
        for step, example in enumerate(dataloader_evaluation):
            torch.cuda.empty_cache()
            X=combine_fixed_length(example['raw_emg'], 200*8).to(device)
            #X=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True,  padding_value= FLAGS.pad).to(device)
            y = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True, padding_value=FLAGS.pad).to(device)
        
            #Forward Model using Greedy Approach not teacher forcing
            target= y[:,1:]
            phones_seq, new_word_seq_idx = run_greedy(model, example['lengths'] ,X, target, n_phones, device)
            
            #Append lists to calculate the PER
            predictions_eval += phones_seq
            references_eval += example['phonemes']
            text_eval += example['text']
            running_total_eval += y.shape[0] * y.shape[1]
            running_correct_eval += sum([sum(item) for item in torch.eq(new_word_seq_idx,y)]).item()

        #Reporting PER
        logging.info(f'----Prediction Evaluation-----')
        logging.info(f'Evaluation Prediction: {predictions_eval[0]} ---> \n  Evaluation Reference: {references_eval[0]}  (Evaluation PER: {jiwer.wer(references_eval[0], predictions_eval[0])}) \n Evaluation Reference Text: {text_eval[0]}')
        logging.info(f'----Prediction Training-----')
        logging.info(f'Training Prediction: {predictions_train[0]} ---> \n Training Reference: {references_train[0]}  (Training PER: {jiwer.wer(references_train[0], predictions_train[0])}) \n Training Reference Text: {text_train[0]}')
        writer.add_scalar('PhonemeErrorRate/Training', jiwer.wer(references_train, predictions_train), batch_idx)
        writer.add_scalar('PhonemeErrorRate/Evaluation', jiwer.wer(references_eval, predictions_eval), batch_idx)
        writer.add_scalar('PhonemeErrorRate_Epoch/Training', jiwer.wer(references_train, predictions_train), epoch_idx)
        writer.add_scalar('PhonemeErrorRate_Epoch/Evaluation', jiwer.wer(references_eval, predictions_eval), epoch_idx)
        writer.add_scalar('Accuracy_Epoch/Training', round(100 * running_correct_train / running_total_train,1), epoch_idx)
        writer.add_scalar('Accuracy_Epoch/Evaluation', round(100 * running_correct_eval / running_total_eval,1), epoch_idx)
        
        curr_eval_PER=jiwer.wer(references_eval, predictions_eval)
        writer.flush()
        predictions_train=[]
        references_train=[]
        predictions_eval=[]
        references_eval=[]
        text_train=[]
        text_eval=[]
        running_total_train = 0
        running_correct_train = 0
        running_total_eval = 0
        running_correct_eval = 0



    ################################################### TRAINING MODEL BELOW ########################################################################       

    ##INITIALIZATION##
    
    #Buffer variables initizialiation
    batch_idx = 0; train_loss= 0; train_dec_loss= 0; train_enc_loss= 0; eval_loss = 0; eval_dec_loss = 0; eval_enc_loss = 0; run_train_steps=0; run_eval_steps=0; predictions_train=[]; references_train=[]; predictions_eval=[]; references_eval=[]; losses = []; alpha_loss=0.25; best_eval_PER=10; curr_eval_PER=0; text_eval=[]; text_train=[]

    #Define Dataloader
    dynamicBatchTrainingSampler=DynamicBatchSampler(trainset, FLAGS.max_batch_length, FLAGS.n_buckets, shuffle=True, batch_ordering='random')
    dynamicBatchEvaluationSampler=DynamicBatchSampler(devset, FLAGS.max_batch_length, FLAGS.n_buckets, shuffle=True, batch_ordering='random')
    dataloader_training = torch.utils.data.DataLoader(trainset, pin_memory=(device=='cuda'), num_workers=0,collate_fn=EMGDataset.collate_raw, batch_sampler= dynamicBatchTrainingSampler)
    dataloader_evaluation = torch.utils.data.DataLoader(devset, pin_memory=(device=='cuda'), num_workers=0,collate_fn=EMGDataset.collate_raw, batch_sampler= dynamicBatchEvaluationSampler)
    
    #Define model and loss function
    n_phones = len(devset.phone_transform.phoneme_inventory)
    model = Model(devset.num_features, n_phones + 1, n_phones, device) # plus 1 for the blank symbol of CTC loss in the encoder
    model=nn.DataParallel(model).to(device)
    #loss_fn=nn.CrossEntropyLoss(ignore_index=FLAGS.pad)
    loss_fn = LabelSmoothingLoss(epsilon=0.1, num_classes=n_phones)

    #If it is enable and you want to start from a pre-trained model
    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from)
        model.load_state_dict(state_dict, strict=False)

    #Define optimizer and scheduler for the learning rate
    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[18], gamma=.1)
    
    ##MODEL TRAINING##
    
    optim.zero_grad()
    for epoch_idx in range(FLAGS.n_epochs):
        losses = []
        torch.cuda.empty_cache()
        training_loop()  
        #Random shift batches
        dynamicBatchTrainingSampler.set_epoch(epoch_idx + 1)
        #PER
        if epoch_idx % FLAGS.report_PER == 0:  
            report_PER()
        #Change learning rate
        #lr_sched.step()
        '''
        if epoch_idx % 3 == 0:
            alpha_loss += -0.1
            alpha_loss=max(alpha_loss,0.1)
        '''
        #Mean of the main loss and logging
        logging.info(f'-----finished epoch {epoch_idx+1} - training loss: {np.mean(losses):.4f}------')
        #Save the Best Model
        if curr_eval_PER < best_eval_PER:
            torch.save(model.state_dict(), os.path.join(FLAGS.output_directory,'model.pt'))
            best_eval_PER= curr_eval_PER
        #Stop if Training loss reaches convergence
        if round(np.mean(losses), 1) == 0.00:
            break

    return model


def evaluate_saved_beam_search():
    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    testset = EMGDataset(test=True)
    #testset = EMGDataset(dev=False,test=False)
    n_phones = len(testset.phone_transform.phoneme_inventory)
    model = Model(testset.num_features, n_phones + 1, n_phones, device) #plus 1 for the blank symbol of CTC loss in the encoder
    model=nn.DataParallel(model).to(device)
    tree = PrefixTree.init_tree(FLAGS.phonesSet,FLAGS.vocabulary,FLAGS.dict)
    language_model = PrefixTree.init_language_model(FLAGS.lang_model)
    model.load_state_dict(torch.load(FLAGS.evaluate_saved_beam_search))
    dataloader = torch.utils.data.DataLoader(testset,  collate_fn=EMGDataset.collate_raw,shuffle=False ,batch_size=1)
    references = []
    predictions = []
     
    model.eval() 
    with torch.no_grad():
        for example in dataloader:
            X=combine_fixed_length(example['raw_emg'], 200*8).to(device)
            #X=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True, padding_value= FLAGS.pad).to(device)
            tgt = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True, padding_value= FLAGS.pad).to(device)

            target= tgt[:,1:]
            pred=run_single_bs(model,X,target,n_phones,tree,language_model,device, example['lengths'])
            torch.cuda.empty_cache()
 
            pred_text = testset.text_transform.clean_text(' '.join(pred[2]))
            target_text = testset.text_transform.clean_text(example['text'][0])
            if len(target_text) is not 0:
                references.append(target_text)
                predictions.append(pred_text)  
                logging.info(f'Prediction:{pred_text} ---> Reference:{target_text}  (WER: {jiwer.wer(target_text, pred_text)})')
        
    logging.info(f'Final WER: {jiwer.wer(references, predictions)}')

def evaluate_saved_greedy_search():
    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    testset = EMGDataset(test=True)
    #testset = EMGDataset(dev=False,test=False)
    n_phones = len(testset.phone_transform.phoneme_inventory)
    model = Model(testset.num_features, n_phones + 1, n_phones, device) #plus 1 for the blank symbol of CTC loss in the encoder
    model=nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(FLAGS.evaluate_saved_greedy_search))
    dataloader = torch.utils.data.DataLoader(testset, collate_fn=EMGDataset.collate_raw, shuffle=False, batch_size=1)
    references = []
    predictions = []
    running_total=0
    running_correct=0

    model.eval() 
    with torch.no_grad():
        for idx, example in enumerate(dataloader):
            #Collect the data
            X=combine_fixed_length(example['raw_emg'], 200*8).to(device)
            #X=nn.utils.rnn.pad_sequence(example['emg'], batch_first=True,  padding_value= FLAGS.pad).to(device)
            y = nn.utils.rnn.pad_sequence(example['phonemes_int'], batch_first=True, padding_value=FLAGS.pad).to(device)
        
            #Forward Model
            target= y[:,1:]
            phones_seq, new_word_seq_idx = run_greedy(model, example['lengths'], X, target, n_phones, device)

            #Append lists to calculate the PER
            predictions += phones_seq
            references += example['phonemes']
            running_total += y.shape[0] * y.shape[1]
            running_correct += sum([sum(item) for item in torch.eq(new_word_seq_idx,y)]).item()
            logging.info(f'Prediction:{phones_seq} ---> Reference:{example["phonemes"]}  (PER: {jiwer.wer(phones_seq, example["phonemes"])} and Accuracy: {round(100 * sum([sum(item) for item in torch.eq(new_word_seq_idx,y)]).item() / (y.shape[0] * y.shape[1]),1)})')

    print(f'PER: {jiwer.wer(references, predictions)} and accuracy: {round(100 * running_correct / running_total,1)}')

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
    load_dictionary()
    if FLAGS.evaluate_saved_beam_search is not None:
        os.makedirs(FLAGS.output_directory, exist_ok=True)
        logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log_beam_search.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")
        evaluate_saved_beam_search()
    elif FLAGS.evaluate_saved_greedy_search is not None:
        os.makedirs(FLAGS.output_directory, exist_ok=True)
        logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log_greedy_search.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")
        evaluate_saved_greedy_search()
    else:
        main()
