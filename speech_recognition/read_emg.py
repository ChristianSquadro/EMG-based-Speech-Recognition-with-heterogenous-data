import re
import os
import numpy as np
from typing import List
import random
import scipy
import json
import copy
import sys
import pickle
import string
import logging
from scipy.stats import lognorm
from functools import lru_cache
from copy import copy

import librosa
import soundfile as sf

import torch

from data_utils import load_audio, get_emg_features, FeatureNormalizer, phoneme_inventory, read_phonemes, TextTransform, PhoneTransform

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_list('remove_channels', [], 'channels to remove')
flags.DEFINE_list('silent_data_directories', ['./emg_data/silent_parallel_data'], 'silent data locations')
flags.DEFINE_list('voiced_data_directories', ['./emg_data/voiced_parallel_data','./emg_data/nonparallel_data'], 'voiced data locations')
flags.DEFINE_string('testset_file', 'testset_largedev.json', 'file with testset indices')
flags.DEFINE_string('text_align_directory', 'text_alignments', 'directory with alignment files')

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)

def load_utterance(base_dir, index, limit_length=False, debug=False):
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f'{index}_emg.npy'))
    before = os.path.join(base_dir, f'{index-1}_emg.npy')
    after = os.path.join(base_dir, f'{index+1}_emg.npy')
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0,raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0,raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0]-raw_emg_after.shape[0],:]
    emg_orig = apply_to_all(subsample, x, 689.06, 1000)
    x = apply_to_all(subsample, x, 516.79, 1000)
    emg = x

    for c in FLAGS.remove_channels:
        emg[:,int(c)] = 0
        emg_orig[:,int(c)] = 0

    emg_features = get_emg_features(emg)

    mfccs = load_audio(os.path.join(base_dir, f'{index}_audio_clean.flac'),
            max_frames=min(emg_features.shape[0], 800 if limit_length else float('inf')))

    if emg_features.shape[0] > mfccs.shape[0]:
        emg_features = emg_features[:mfccs.shape[0],:]
    assert emg_features.shape[0] == mfccs.shape[0]
    emg = emg[6:6+6*emg_features.shape[0],:]
    emg_orig = emg_orig[8:8+8*emg_features.shape[0],:]
    assert emg.shape[0] == emg_features.shape[0]*6

    with open(os.path.join(base_dir, f'{index}_info.json')) as f:
        info = json.load(f)

    sess = os.path.basename(base_dir)
    
    phonemes=read_phonemes(info['text'])
    
    return mfccs, emg_features, info['text'], (info['book'],info['sentence_index']), phonemes, emg_orig.astype(np.float32)

class EMGDirectory(object):
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory

class SizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch = []
        batch_length = 0
        for idx in indices:
            directory_info, file_idx = self.dataset.example_indices[idx]
            with open(os.path.join(directory_info.directory, f'{file_idx}_info.json')) as f:
                info = json.load(f)
            if not np.any([l in string.ascii_letters for l in info['text']]):
                continue
            length = sum([emg_len for emg_len, _, _ in info['chunks']])
            if length > self.max_len:
                logging.warning(f'Warning: example {idx} cannot fit within desired batch length')
            if length + batch_length > self.max_len:
                yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length
        # dropping last incomplete batch

class DynamicBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        max_batch_length: int,
        num_buckets: int = None,
        shuffle: bool = True,
        batch_ordering: str = "random",
        max_batch_ex: int = None,
        bucket_boundaries: List[int] = [],
        seed: int = 42,
        epoch: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
    ):
        self._dataset = dataset
        self._ex_lengths = {}
        self.verbose = verbose
        self.lengths_list=[]
        
        indices = list(range(len(dataset)))
        for idx in indices:
            directory_info, file_idx = dataset.example_indices[idx]
            with open(os.path.join(directory_info.directory, f'{file_idx}_info.json')) as f:
                info = json.load(f)
            self.lengths_list.append(sum([emg_len for emg_len, _, _ in info['chunks']]))

        if self.lengths_list is not None:
            # take length of examples from this argument and bypass length_key
            for indx in range(len(self.lengths_list)):
                self._ex_lengths[str(indx)] = self.lengths_list[indx]

        if len(bucket_boundaries) > 0:
            if not all([x >= 0 for x in bucket_boundaries]):
                raise ValueError(
                    "All elements in bucket boundaries should be non-negative (>= 0)."
                )
            if not len(set(bucket_boundaries)) == len(bucket_boundaries):
                raise ValueError(
                    "Bucket_boundaries should not contain duplicates."
                )
            np.testing.assert_array_equal(
                np.array(bucket_boundaries),
                np.array(sorted(bucket_boundaries)),
                err_msg="The arg bucket_boundaries should be an ascending sorted list of non negative values values!",
            )
            self._bucket_boundaries = np.array(sorted(bucket_boundaries))
        else:
            # use num_buckets
            self._bucket_boundaries = np.array(
                self._get_boundaries_through_warping(
                    max_batch_length=max_batch_length,
                    num_quantiles=num_buckets,
                )
            )

        self._max_batch_length = max_batch_length
        self._shuffle_ex = shuffle
        self._batch_ordering = batch_ordering
        self._seed = seed
        self._drop_last = drop_last
        if max_batch_ex is None:
            max_batch_ex = np.inf
        self._max_batch_ex = max_batch_ex
        # Calculate bucket lengths - how often does one bucket boundary fit into max_batch_length?
        self._bucket_lens = [
            max(1, int(max_batch_length / self._bucket_boundaries[i]))
            for i in range(len(self._bucket_boundaries))
        ] + [1]
        self._epoch = epoch
        self._generate_batches()

    def get_durations(self, batch):
        """Gets durations of the elements in the batch."""
        return [self._ex_lengths[str(idx)] for idx in batch]

    def _get_boundaries_through_warping(
        self, max_batch_length: int, num_quantiles: int,
    ) -> List[int]:

        # NOTE: the following lines do not cover that there is only one example in the dataset
        # warp frames (duration) distribution of train data
        # linspace set-up
        num_boundaries = num_quantiles + 1
        # create latent linearly equal spaced buckets
        latent_boundaries = np.linspace(
            1 / num_boundaries, num_quantiles / num_boundaries, num_quantiles,
        )
        # get quantiles using lognormal distribution
        quantiles = lognorm.ppf(latent_boundaries, 1)
        # scale up to to max_batch_length
        bucket_boundaries = quantiles * max_batch_length / quantiles[-1]
        
        return list(sorted(bucket_boundaries))

    def _permute_batches(self):

        if self._batch_ordering == "random":
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(
                len(self._batches), generator=g).tolist()  # type: ignore
            tmp = []
            for idx in sampler:
                tmp.append(self._batches[idx])
            self._batches = tmp

        elif self._batch_ordering == "ascending":
            self._batches = sorted(
                self._batches,
                key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
            )
        elif self._batch_ordering == "descending":
            self._batches = sorted(
                self._batches,
                key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
                reverse=True,
            )
        else:
            raise NotImplementedError

    def _generate_batches(self):
        if self._shuffle_ex:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(len(self._dataset), generator=g).tolist()  # type: ignore
        else:
            # take examples as they are: e.g. they have been sorted
            sampler = range(len(self._dataset))  # type: ignore

        self._batches = []
        bucket_batches = [[] for i in self._bucket_lens]

        stats_tracker = [
            {"min": np.inf, "max": -np.inf, "tot": 0, "n_ex": 0}
            for i in self._bucket_lens
        ]

        for idx in sampler:
            directory_info, file_idx = self._dataset.example_indices[idx]
            with open(os.path.join(directory_info.directory, f'{file_idx}_info.json')) as f:
                info = json.load(f)
            if not np.any([l in string.ascii_letters for l in info['text']]):
                continue
            # length of pre-sampled audio
            item_len = self._ex_lengths[str(idx)]
            # bucket to fill up most padding
            bucket_id = np.searchsorted(self._bucket_boundaries, item_len)
            # fill audio's duration into that bucket
            bucket_batches[bucket_id].append(idx)

            stats_tracker[bucket_id]["min"] = min(
                stats_tracker[bucket_id]["min"], item_len
            )
            stats_tracker[bucket_id]["max"] = max(
                stats_tracker[bucket_id]["max"], item_len
            )
            stats_tracker[bucket_id]["tot"] += item_len
            stats_tracker[bucket_id]["n_ex"] += 1
            # track #samples - why not duration/#frames; rounded up?
            # keep track of durations, if necessary

            if (
                len(bucket_batches[bucket_id]) >= self._bucket_lens[bucket_id]
                or len(bucket_batches[bucket_id]) >= self._max_batch_ex
            ):
                self._batches.append(bucket_batches[bucket_id])
                bucket_batches[bucket_id] = []
                # keep track of durations

        # Dump remaining batches
        if not self._drop_last:
            for batch in bucket_batches:
                if batch:
                    self._batches.append(batch)

        self._permute_batches()  # possibly reorder batches

    def __iter__(self):
        for batch in self._batches:
            yield batch

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self._epoch = epoch
        if self._shuffle_ex:  # re-generate examples if ex_ordering == "random"
            self._generate_batches()

    def __len__(self):
        return len(self._batches)

class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir=None, limit_length=False, dev=False, test=False, no_testset=False, no_normalizers=False):

        self.text_align_directory = FLAGS.text_align_directory

        if no_testset:
            devset = []
            testset = []
        else:
            with open(FLAGS.testset_file) as f:
                testset_json = json.load(f)
                devset = testset_json['dev']
                testset = testset_json['test']

        directories = []
        if base_dir is not None:
            directories.append(EMGDirectory(0, base_dir, False))
        else:
            for sd in FLAGS.silent_data_directories:
                for session_dir in sorted(os.listdir(sd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))

            has_silent = len(FLAGS.silent_data_directories) > 0
            for vd in FLAGS.voiced_data_directories:
                for session_dir in sorted(os.listdir(vd)):
                    directories.append(EMGDirectory(len(directories), os.path.join(vd, session_dir), False, exclude_from_testset=has_silent))

        self.example_indices = []
        self.voiced_data_locations = {} # map from book/sentence_index to directory_info/index
        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                m = re.match(r'(\d+)_info.json', fname)
                if m is not None:
                    idx_str = m.group(1)
                    with open(os.path.join(directory_info.directory, fname)) as f:
                        info = json.load(f)
                        if info['sentence_index'] >= 0: # boundary clips of silence are marked -1
                            location_in_testset = [info['book'], info['sentence_index']] in testset
                            location_in_devset = [info['book'], info['sentence_index']] in devset
                            if (test and location_in_testset and not directory_info.exclude_from_testset) \
                                    or (dev and location_in_devset and not directory_info.exclude_from_testset) \
                                    or (not test and not dev and not location_in_testset and not location_in_devset):
                                self.example_indices.append((directory_info,int(idx_str)))

                            if not directory_info.silent:
                                location = (info['book'], info['sentence_index'])
                                self.voiced_data_locations[location] = (directory_info,int(idx_str))

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            self.mfcc_norm, self.emg_norm = pickle.load(open(FLAGS.normalizers_file,'rb'))

        sample_mfccs, sample_emg, _, _, _, _ = load_utterance(self.example_indices[0][0].directory, self.example_indices[0][1])
        self.num_speech_features = sample_mfccs.shape[1]
        self.num_features = sample_emg.shape[1]
        self.limit_length = limit_length
        self.num_sessions = len(directories)

        self.text_transform = TextTransform()
        self.phone_transform = PhoneTransform()

    def silent_subset(self):
        result = copy(self)
        silent_indices = []
        for example in self.example_indices:
            if example[0].silent:
                silent_indices.append(example)
        result.example_indices = silent_indices
        return result

    def subset(self, fraction):
        result = copy(self)
        result.example_indices = self.example_indices[:int(fraction*len(self.example_indices))]
        return result

    def __len__(self):
        return len(self.example_indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        mfccs, emg, text, book_location, phonemes, raw_emg = load_utterance(directory_info.directory, idx, self.limit_length)
        raw_emg = raw_emg / 20
        raw_emg = 50*np.tanh(raw_emg/50.)

        if not self.no_normalizers:
            mfccs = self.mfcc_norm.normalize(mfccs)
            emg = self.emg_norm.normalize(emg)
            emg = 8*np.tanh(emg/8.)

        session_ids = np.full(emg.shape[0], directory_info.session_index, dtype=np.int64)
        audio_file = f'{directory_info.directory}/{idx}_audio_clean.flac'

        words= [word for word in text]
        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

        result = {'audio_features':torch.from_numpy(mfccs).pin_memory(), 'emg':torch.from_numpy(emg).pin_memory(), 'text':text, 'words': words,'text_int': torch.from_numpy(text_int).pin_memory(), 'file_label':idx, 'session_ids':torch.from_numpy(session_ids).pin_memory(), 'book_location':book_location, 'silent':directory_info.silent, 'raw_emg':torch.from_numpy(raw_emg).pin_memory()}

        if directory_info.silent:
            voiced_directory, voiced_idx = self.voiced_data_locations[book_location]
            voiced_mfccs, voiced_emg, _, _, phonemes, _= load_utterance(voiced_directory.directory, voiced_idx, False)

            if not self.no_normalizers:
                voiced_mfccs = self.mfcc_norm.normalize(voiced_mfccs)
                voiced_emg = self.emg_norm.normalize(voiced_emg)
                voiced_emg = 8*np.tanh(voiced_emg/8.)

            result['parallel_voiced_audio_features'] = torch.from_numpy(voiced_mfccs).pin_memory()
            result['parallel_voiced_emg'] = torch.from_numpy(voiced_emg).pin_memory()

            audio_file = f'{voiced_directory.directory}/{voiced_idx}_audio_clean.flac'
        
        result['phonemes'] = ' '.join(phonemes)
        phonemes=np.array(self.phone_transform.phone_to_int(phonemes), dtype=np.int64)
        result['phonemes_int'] = torch.from_numpy(phonemes).pin_memory() # either from this example if vocalized or aligned example if silent
        result['audio_file'] = audio_file

        return result

    @staticmethod
    def collate_raw(batch):
        batch_size = len(batch)
        audio_features = []
        audio_feature_lengths = []
        parallel_emg = []
        raw_emg_augmented = []
        lengths_augmented=[]
        for ex in batch:
            if ex['silent']:
                audio_features.append(ex['parallel_voiced_audio_features'])
                audio_feature_lengths.append(ex['parallel_voiced_audio_features'].shape[0])
                parallel_emg.append(ex['parallel_voiced_emg'])
                '''
                generator = torch.Generator().manual_seed(0)
                silence = torch.zeros((torch.randint(200, 201,(1,), generator=generator).item(), 8))
                raw_emg_augmented.append(torch.concatenate([ex['raw_emg'], silence, ex['silent_raw_emg']], axis=0))
                phonemes_int_augmented.append(torch.concatenate([ex['phonemes_int'][:-1], ex['phonemes_int'][1:]], axis=0)) # to exclude </S> and <S> in the middle
                lengths_augmented.append(ex['emg'].shape[0] + (silence.shape[0] // 8) + ex['parallel_voiced_emg'].shape[0])
                '''
            else:
                audio_features.append(ex['audio_features'])
                audio_feature_lengths.append(ex['audio_features'].shape[0])
                parallel_emg.append(np.zeros(1))
                '''
                raw_emg_augmented.append(ex['raw_emg'])
                phonemes_int_augmented.append(ex['phonemes_int'])
                lengths_augmented.append(ex['emg'].shape[0])
                '''
            raw_emg_augmented.append(torch.roll(ex['raw_emg'], shifts=random.randint(-2,2), dims=1))
            lengths_augmented.append(ex['emg'].shape[0])

        phonemes = [ex['phonemes'] for ex in batch]
        phonemes_int = [ex['phonemes_int'] for ex in batch]
        phonemes_lengths = [ex['phonemes_int'].shape[0] for ex in batch]
        emg = [ex['emg'] for ex in batch]
        raw_emg = [ex['raw_emg'] for ex in batch]
        session_ids = [ex['session_ids'] for ex in batch]
        lengths = [ex['emg'].shape[0] for ex in batch]
        silent = [ex['silent'] for ex in batch]
        text= [ex['text'] for ex in batch]
        text_ints = [ex['text_int'] for ex in batch]
        text_lengths = [ex['text_int'].shape[0] for ex in batch]

        result = {'audio_features':audio_features,
                  'audio_feature_lengths':audio_feature_lengths,
                  'emg':emg,
                  'raw_emg':raw_emg,
                  'raw_emg_augmented': raw_emg_augmented,
                  'parallel_voiced_emg':parallel_emg,
                  'phonemes':phonemes,
                  'phonemes_int':phonemes_int,
                  'phonemes_int_lengths': phonemes_lengths,
                  'session_ids':session_ids,
                  'lengths':lengths,
                  'lengths_augmented': lengths_augmented,
                  'silent':silent,
                  'text':text,
                  'text_int':text_ints,
                  'text_int_lengths':text_lengths}
        return result

def make_normalizers():
    dataset = EMGDataset(no_normalizers=True)
    mfcc_samples = []
    emg_samples = []
    for d in dataset:
        mfcc_samples.append(d['audio_features'])
        emg_samples.append(d['emg'])
        if len(emg_samples) > 50:
            break
    mfcc_norm = FeatureNormalizer(mfcc_samples, share_scale=True)
    emg_norm = FeatureNormalizer(emg_samples, share_scale=False)
    pickle.dump((mfcc_norm, emg_norm), open(FLAGS.normalizers_file, 'wb'))

if __name__ == '__main__':
    FLAGS(sys.argv)
    d = EMGDataset()
    for i in range(1000):
        d[i]

