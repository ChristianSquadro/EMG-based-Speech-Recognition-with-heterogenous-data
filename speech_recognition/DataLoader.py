import torch
import torch.utils.data as tud
import random
import scipy.io
import scipy.signal
import sacred
import numpy as np
import json
import os
import sys
import sqlite3
import re
import itertools
import glob
import pickle  
import pdb
import itertools
import BundledData  
sys.path.append('/home/mwand/projects/EMG/BeamSearch')    
import Dictionary

AudioCorpusDir = '/home/mwand/projects/EMG/EMG-UKA-Full-Corpus/raw_901_001/all'
CorpusDir = '/home/mwand/projects/EMG/EMG-UKA-Full-Corpus/901_001'
phonesSetFile = '/home/mwand/projects/EMG/Descriptions/PhonesSet-ReducedPhones' # only phones
updatedDict = '/home/mwand/projects/EMG-Silvia/End2EndSystem/Descriptions/EMG-Full-NoWB-ReducedPhones-WithSil-ButNoFiller.dict'
dctFile = '/home/mwand/projects/EMG/Descriptions/CombinedDict-ReducedPhones.dict' # words + phones non in forma di dict, ossia tipo CIAO CI IA O
# # EmgPath = '/home/mwand/projects/EMG/EMG-UKA-Full-Corpus/Processed/TD0-FAC-2019-Multises-NormalizedVar1-Audible'
db = '/home/mwand/projects/EMG/database/emg.db'
normSessions = '/home/mwand/projects/EMG/EMG-UKA-Full-Corpus/Processed/TD0-FAC-NormalizedVar1-Audible'
LabelsDir = 'MappedLabels'
WordsAligns = '/home/mwand/projects/EMG/EMG-UKA-Full-Corpus/Alignments'
TrainPath = '/home/mwand/projects/EMG/EMG-UKA-Full-Corpus/901_001/train_set'
TestPath = '/home/mwand/projects/EMG/EMG-UKA-Full-Corpus/901_001/eval_set'

ingr = sacred.Ingredient('Data')


@ingr.config
def cfg():
    Spkses = '901_001-001-001' # '551-010_013_015_016_017_018_019_020_021_022_024_026_027_028_029-030'
    Source = 'emg' 
    assert Source in ['audio','emg']
    Norm = ''
    FeatShift = 0.0 # DEBUG ONLY: add this value to all features
    BatchSize = 30
    
    ContextFrames = 1 # 0 = no stacking
    Shuffle = True
    AddNoise = 0.0
    UseBundled = False
    assert not UseBundled
    Tree = 'VOICED' # unused unless "UseBundled" is True

    ChannelDrop = {
        'TriggerProb': -1.0,
        'MinDrop': 0.0,
        'MaxDrop': 0.4,
        'TimeConsistent': True
            }
    TimeDrop = {
        'TriggerProb': -1.0,
        'DropCount': 4,
        'DropFrac': 0.05
        }
    
    # the fraction of training data which is used (from 1 to 100, integer)
    TrainFrac = 100

@ingr.capture
def make_preprocessor(Spkses,Norm,FeatShift,ContextFrames,ChannelDrop,TimeDrop,TrainFrac):
    return _EMGPreprocessor(Spkses,Norm,FeatShift,ContextFrames,ChannelDrop,TimeDrop,TrainFrac)

@ingr.capture
def make_loader(subset, Shuffle, preproc, num_workers, Spkses, BatchSize):
    assert subset in ['train','test']
    dataset = _EMGDataset(subset, preproc) 
    sampler = _BatchRandomSampler(dataset, BatchSize, shuffle=Shuffle)
    loader = tud.DataLoader(dataset,
                batch_size=BatchSize,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn = _zipper,
                drop_last=False) 

    # monkey patch loader!
    def set_sampling_mode(ldr,mode):
        ldr.sampler.set_mode(mode)

    tud.DataLoader.set_sampling_mode = set_sampling_mode
    return loader

def save_train_voc_file(resultInfo):
    words = set()
    
    for i in range(len(resultInfo)):
        phrase = resultInfo[i]['TEXT'].upper().split()
        for w in phrase:
            words.add(w)

    f = open('Descriptions/TrainVocabulary.txt','a')
    f.write(str(words))

def _zipper(args):
    return zip(*args) # takes iterables and aggregates them in a tuple, then return it

@ingr.capture
def collect_data(subset, phone_name_to_index, Spkses, AddNoise, UseBundled, Tree):
    assert subset in ['train','test']
    testVal = 1 if subset == 'test' else 0
    speaker, train_sessions, test_sessions = Spkses.split('-')
    pathInfix = speaker + '/' + train_sessions + '-' + test_sessions
    
    if testVal == 0:
        sessionsString = ' OR '.join(['SESSION_NB = "%s"' % ses for ses in train_sessions.split('_')])
    else:
        sessionsString = ' OR '.join(['SESSION_NB = "%s"' % ses for ses in test_sessions.split('_')])

    collected_data = []; collected_info = []; collected_phone_targets = []; collected_word_targets = []; collected_frame_targets = []
    max_length = -1
    dct = _load_dictionary()
    dct901 = load_updated_dict()

    if speaker == '901_001':
        if subset == 'train':
            train_file = open(TrainPath, 'r')
            train_data = train_file.read().split()
            utterances = [x.split('_')[2] for x in train_data ] 
            uttList = '-'.join(utterances)
        else:
            test_file = open(TestPath, 'r')
            test_data = test_file.read().split()
            utterances = [x.split('_')[2] for x in test_data]
            uttList = '-'.join(utterances)
        
        # Get data from file
        all_info = json.load(open("Descriptions/901_dict.txt"))

        utt = list(uttList.split('-'))
        
        for u in utt:
            
            thisEmgData = preProcessingWithPath(u)

            if thisEmgData.shape[0] > max_length:
                max_length = thisEmgData.shape[0]
            
            key = '901' + u
            
            for n in range(len(all_info[key])):
                for ch in ['.', ',', ':', ';', '?', ')']:
                    if ch in all_info[key][n]:
                        all_info[key][n] = all_info[key][n].replace(ch,'')
                    if '-' in all_info[key][n]:
                        all_info[key][n] = all_info[key][n].split('-')
                        all_info[key][n] = ' '.join(all_info[key][n])
                        
            text = ' '.join(all_info[key])
            
            thisInfo = (speaker, u, text)
            words = thisInfo[2].upper().split()

            mightProns = get_words_phones(words, dct901)
            thisTarget = [item for sublist in mightProns for item in sublist]

            # also get framewise targets
            frame_tgt_filename = CorpusDir + '/' + '901_001_%s.labels.txt' % u
            frame_tgt_data_1 = [ line.rstrip() for line in open(frame_tgt_filename,'r') ]
            frame_tgt_data_2 = [ re.sub(r'X([MNL])',r'\1',t) for t in frame_tgt_data_1 ] # this replaces phones XM, XN, XL with M, N, L (TODO to be generalized)
            frame_tgt_data_3 = [ re.sub(r'-[bme]','',t) for t in frame_tgt_data_2 ] # this removes the begin, middle, end part (TODO TODO XXX)
            frame_tgt_data = [ phone_name_to_index[p] for p in frame_tgt_data_3 ] # make data numeric
#             assert len(frame_tgt_data) >= thisEmgData.shape[0]
            frame_tgt_data = frame_tgt_data[:thisEmgData.shape[0]]
            
            if UseBundled:   
                raise Exception('do not do this')
                # get thisInfo 
                treeTarget = BundledData.getTargetFromTree(Spkses, Tree, text) # all this probably does not work
                collected_phone_targets.append(treeTarget)
            else:
                collected_phone_targets.append(thisTarget)    
                
            collected_data.append(thisEmgData)
            collected_info.append(thisInfo)
            collected_word_targets.append(words)
            collected_frame_targets.append(frame_tgt_data)
        
    if UseBundled: # encode data
        flat_list = list(itertools.chain(*collected_phone_targets)) 
        model_names = list(set(flat_list)) # get unique elements
        dct_b = dict((el,0) for el in model_names)
        for i in range(len(model_names)):
            dct_b[model_names[i]] = i 

        # encode data 
        bundResultTarget = encode_bundled_data(collected_phone_targets, dct_b)
        collected_phone_targets = bundResultTarget

    # Make masks and join
    mask = np.zeros((max_length,len(collected_data)),dtype=bool)
    
#     if AddNoise > 0.0 and subset == 'train':
#         for i in range(len(collected_data)):
#             noise = np.random.normal(0, AddNoise, collected_data[i].shape)
#             collected_data[i] = collected_data[i] + noise
    # this is done in preProcessingWithPath

    joined_result_data = np.zeros((max_length,len(collected_data),collected_data[0].shape[1]))
    for pos,seq in enumerate(collected_data):
        mask[0:seq.shape[0],pos] = True
        joined_result_data[0:seq.shape[0],pos,:] = collected_data[pos]
        
    # transpose to have 'batch' first
    joined_result_data = np.transpose(joined_result_data,(1,0,2))
    mask = np.transpose(mask, (1,0))
    
    data = {
        'DATA' : joined_result_data,
        'MASK' : mask,
        'WORDS' : collected_word_targets,
        'TARGET' : collected_phone_targets,
        'FRAMETARGET' : collected_frame_targets,
        'INFO' : collected_info,
    }

    return data

@ingr.capture
def preProcessingWithPath(utt,ContextFrames,AddNoise,Source):
   
    # doStacking: for each frame -> feature vector, in each step: stack more feature vectors
    def _doStacking(td0, stackHeight):
        tdx = np.empty(shape=[td0.shape[0],0])
        td0Stack=_repmat(td0[0],stackHeight)
        td0Stack=np.vstack((td0Stack,td0))
        td0Stack=np.vstack((td0Stack,_repmat(td0[-1],stackHeight)))
        for i in range(-stackHeight, (stackHeight+1)):
            start = stackHeight + i
            end = td0.shape[0] + stackHeight + i
            debug = td0Stack[start:end,0:td0Stack.shape[1]]
            tdx = np.hstack((tdx,td0Stack[start:end,0:td0Stack.shape[1]]))
        return tdx

    def _repmat(oneRowMatrix, stackHeight):
        result=np.empty(shape=[0,oneRowMatrix.shape[0]])
        for i in range(stackHeight):
            result=np.vstack((result,oneRowMatrix))
        return result

    emgFilename = CorpusDir + '/' + '901_001_%s.emg.npy' % utt
    emgMatrix = np.load(emgFilename)[:,1125:1200] # remove context from EMG matrix

    if AddNoise > 0.0:
        noise = np.random.normal(0, AddNoise, emgMatrix.shape)
        emgMatrix += noise
    
    emgMatrix = _doStacking(emgMatrix, ContextFrames) # 901: (375, 825), else: (, 275)
        

    audioFilename = AudioCorpusDir + '/' + '901_001_%s_audio_raw.npy' % utt
    rawAudioMatrix = np.load(audioFilename)[:,0] # remove marker
    processedAudioMatrix = np.log(scipy.signal.spectrogram(rawAudioMatrix,nperseg=200,noverlap=40)[2].T)

    assert processedAudioMatrix.shape[0] >= emgMatrix.shape[0]
#     print('Data sizes: audio %s, emg %s' % (processedAudioMatrix.shape,emgMatrix.shape))

    processedAudioMatrix = processedAudioMatrix[:emgMatrix.shape[0]]
    if Source == 'emg':
        return emgMatrix
    elif Source == 'audio':
        return processedAudioMatrix # only the actual spectrogram

def get_aligns_indexes(thisInfo):
    end_index = []
    end_phones = []
    aligns_path = WordsAligns + '/' + thisInfo['SPK_ID'] + '/' + thisInfo['SESSION_NB'] + '/'
    aligns_file = 'words_' + thisInfo['SPK_ID'] + '_' + thisInfo['SESSION_NB'] + '_' + thisInfo['UTT_NB'] + '.txt'   
    lines = open(aligns_path+aligns_file, 'r').readlines()
    
    for l in lines:
        ln = l.split()
        if not (ln[2] == '$' or ln[2] == 'SIL'):
            elem = (ln[1],ln[2]) # save also word to get target later
            end_index.append(elem) 

    return end_index

def load_updated_dict():
    dct = dict()
    f = open(updatedDict)
    for line in f:
        l = line.split()
        if l != []:
            dct[l[0]] = l[1:]
    return dct

def get_words_phones(sentence, dct):
    phones = []
    for n in sentence:
        p = dct[n]
        phones.append(p)
    return phones

def _load_dictionary():
    dct = Dictionary.Dictionary()
    dct.readPhonesSet(phonesSetFile)
    dct.readDictionary(dctFile,None)
    return dct

class _EMGPreprocessor():
    START = '<S>'
    END = '</S>'

    def __init__(self,Spkses,Norm,FeatShift,ContextFrames,ChannelDrop,TimeDrop,TrainFrac):
        
        self.Spkses = Spkses
        self.start_end = True
        self.dct = _load_dictionary()
        self.context_frames = ContextFrames
        self.channel_drop = ChannelDrop
        self.time_drop = TimeDrop

        # TODO FIXME XXX
        phone_name_to_index = { k: self.dct.lookupPhoneByName(k).idx for k in self.dct._phonesByName.keys() }

        self.train_data = collect_data('train', phone_name_to_index)
        self.test_data = collect_data('test', phone_name_to_index)

        if 'M' in Norm:
            print('Perfoming MEAN subtraction')

            flattened_train_array = self.train_data['DATA'][self.train_data['MASK']]
            train_means = np.mean(flattened_train_array,0)
            assert train_means.ndim == 1 
      
            # perform mean subtraction
            self.train_data['DATA'] -= train_means
            self.test_data['DATA'] -= train_means

        if 'V' in Norm:
            print('Perfoming STD removal')

            flattened_train_array = self.train_data['DATA'][self.train_data['MASK']]
            train_std = np.std(flattened_train_array,0)
            assert train_std.ndim == 1 
      
            # perform mean subtraction
            self.train_data['DATA'] /= train_std
            self.test_data['DATA'] /= train_std

        self.train_data['DATA'] += FeatShift
        self.test_data['DATA'] += FeatShift

        # subsample data based on TrainFrac
        if TrainFrac == 100:
            self.valid_train_idx = np.arange(self.train_data['DATA'].shape[0])
        else:
            full_size = self.train_data['DATA'].shape[0]
            target_size = int(full_size * (TrainFrac / 100))
            self.valid_train_idx = np.random.choice(self.train_data['DATA'].shape[0],target_size,replace=False)

        self.input_dim = self.train_data['DATA'].shape[2]
        self.train_seq_count = self.valid_train_idx.shape[0]
        self.test_seq_count = self.test_data['DATA'].shape[0]

        # encodings
        chars = list(self.dct._phonesByName.keys())
        if self.start_end:
            # START must be last so it can easily be excluded in the output classes of a model
            chars.extend([self.END, self.START])
        self.int_to_char = dict(enumerate(chars))
        self.char_to_int = {v : k for k, v in self.int_to_char.items()}
        self.vocab_size = len(self.int_to_char) 
        
        # the start token has the highest value so take that as vocab size
        if type(self.train_data['TARGET'][0][0]) is int:
            self.vocab_size = self.train_data['TARGET'][0][0] + 1
        
    def encode(self, text):
        text = list(text)
        if self.start_end:
            text = [self.START] + text + [self.END]
        return [self.char_to_int[t] for t in text]
    
    def decode(self, seq):
        text = [self.int_to_char[s] for s in seq]
        
        if not self.start_end:
            return text

        if text[0] == self.START:
            start_pos = 1
        else:
            start_pos = 0
        try:
            end_pos = text.index(self.END)
        except ValueError as e:
            end_pos = len(text)
        return text[start_pos:end_pos]

    def perform_channel_drop(self,data):
        def sample_channel_indices(num_channels, min_num_channels_percentage, max_num_channels_percentage): 
            min_num_channels = int(min_num_channels_percentage * num_channels)
            max_num_channels = min(num_channels, int(max_num_channels_percentage * num_channels))
            num_sampled_channels = random.randint(min_num_channels, max_num_channels)
            channel_idx_list = list(range(num_channels))
            selected_channels = random.sample(channel_idx_list, k=num_sampled_channels)
            return selected_channels

        trigger = (random.random() <= self.channel_drop['TriggerProb'])
        if trigger:
            if self.channel_drop['TimeConsistent']:
                channel_count = data.shape[1] // (2 * self.context_frames + 1)
            else:
                channel_count = data.shape[1]

            dropped_channels = np.array(sample_channel_indices(channel_count,self.channel_drop['MinDrop'],self.channel_drop['MaxDrop']))
            if self.channel_drop['TimeConsistent']:
                true_dropped_channel_list = [ dropped_channels + (channel_count * idx) for idx in range(2 * self.context_frames + 1) ]
                true_dropped_channels = np.concatenate(true_dropped_channel_list,0)
            else:
                true_dropped_channels = dropped_channels
            transformed_data = np.copy(data)
            for dropped_channel_idx in true_dropped_channels:
                transformed_data[:, dropped_channel_idx] = 0.0
        else:
            transformed_data = data
       
        return transformed_data
        
    def perform_time_drop(self,data):
        trigger = (random.random() <= self.time_drop['TriggerProb'])
        if trigger:
            transformed_data = np.copy(data)
            for cnt in range(self.time_drop['DropCount']):
                start = np.random.randint(0,transformed_data.shape[0])
                len_frac = np.random.uniform(low=0.0, high=self.time_drop['DropFrac'])
                len_int = int(transformed_data.shape[0] * len_frac)
                transformed_data[start:start+len_int] = 0.0
        else:
            transformed_data = data
       
        return transformed_data

    
    def getItem(self, subset, id):
        if subset == 'train':
            # take into account reduced training set
            mapped_id = self.valid_train_idx[id]
            if self.channel_drop:
                processed_data = self.perform_time_drop(self.perform_channel_drop(self.train_data['DATA'][mapped_id]))
            else:
                processed_data = self.train_data['DATA'][mapped_id]
            return processed_data, self.train_data['MASK'][mapped_id], self.train_data['WORDS'][mapped_id], self.train_data['TARGET'][mapped_id], self.train_data['FRAMETARGET'][mapped_id], self.train_data['INFO'][mapped_id] 
        elif subset == 'test':
            # no augmentation here
            return self.test_data['DATA'][id], self.test_data['MASK'][id], self.test_data['WORDS'][id], self.test_data['TARGET'][id], self.test_data['FRAMETARGET'][id],self.test_data['INFO'][id]
            

class _EMGDataset(tud.Dataset):
    
    def __init__(self, subset, preproc): # do I need Spkses
        super().__init__()
        self.size = preproc.train_seq_count if subset == 'train' else preproc.test_seq_count
        self.subset = subset
        self.preproc = preproc

        # compute lengths!
        sizes = []
        for idx in range(len(self)):
            this_mask = self.preproc.getItem(self.subset,idx)[1]
            this_size = np.count_nonzero(this_mask)
            sizes.append(this_size)
        self.sizes = np.array(sizes)

        refs = [1,2,3,4,5,6,7,8,9,10]
        self.quantiles = { q: np.quantile(self.sizes,q/10) for q in refs }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.preproc.getItem(self.subset,idx)


class _BatchRandomSampler(tud.sampler.Sampler):
    def __init__(self, data_source, batch_size, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = 'normal'

    def __iter__(self):
#         print('Call to BatchRandomSampler.__iter__ with mode',self.mode)
        all_nums = np.arange(len(self.data_source))

        if self.mode != 'normal':
            # must be 0.1 or so
            quantile = int(self.mode)
            threshold = self.data_source.quantiles[quantile]
            flt = (all_nums < threshold)
            all_nums = all_nums[flt]
        if self.shuffle:
            np.random.shuffle(all_nums)
        batches = np.array_split(all_nums,len(self.data_source) / self.batch_size)
        return (i for b in batches for i in b)

    def __len__(self):
        return len(self.data_source)

    def set_mode(self,mode):
#         print('setting sapling mode to', mode)
        self.mode = mode
