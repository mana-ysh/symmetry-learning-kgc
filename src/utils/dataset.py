"""
TODO
- writing Item class for supporting fancy indexing in PathQueryDataset
"""

import itertools
import numpy as np
import os
import pickle
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

np.random.seed(46)


class Dataset(object):
    def __init__(self, samples):
        assert type(samples) == list or type(samples) == np.ndarray
        self.samples = samples if type(samples) == np.ndarray else np.array(samples)

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)

    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        pass


class TripletDataset(Dataset):
    def __init__(self, samples):
        super(TripletDataset, self).__init__(samples)

    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        samples = []
        with open(data_path) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                samples.append((ent_vocab[sub], rel_vocab[rel], ent_vocab[obj]))
        return TripletDataset(samples)


class LabeledTripletDataset(Dataset):
    def __init__(self, samples):
        super(LabeledTripletDataset, self).__init__(samples)
        assert self.samples.shape[1] == 4

    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        samples = []
        with open(data_path) as f:
            for line in f:
                sub, rel, obj, label = line.strip().split('\t')
                samples.append((ent_vocab[sub], rel_vocab[rel], ent_vocab[obj], int(label)))
        return LabeledTripletDataset(samples)


class PathQueryDataset(Dataset):
    def __init__(self, samples):
        super(PathQueryDataset, self).__init__(samples)

    @classmethod
    def load(cls, data_path):
        raise NotImplementedError


class Vocab(object):
    def __init__(self):
        self.id2word = []
        self.word2id = {}

    def add(self, word):
        if word not in self.id2word:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, word):
        return self.word2id[word]

    @classmethod
    def load(cls, vocab_path):
        root, ext = os.path.splitext(vocab_path)
        if ext == '.pkl':
            with open(vocab_path, 'rb') as fw:
                v = pickle.load(fw)
        else:
            v = Vocab()
            with open(vocab_path) as f:
                for word in f:
                    v.add(word.strip())
        return v


def batch_iter(dataset, batchsize, rand_flg=True):
    n_sample = len(dataset)
    idxs = np.random.permutation(n_sample) if rand_flg else np.arange(n_sample)
    for start_idx in range(0, n_sample, batchsize):
        yield dataset[idxs[start_idx:start_idx+batchsize]]


def generate_synthetic(n_ent, n_rel, pos_ratio, sym_ratio=0.5, split=False):
    """
    NOTE:
      - Currently, fixed number of positive triplets in each relation. is it OK?
    """
    samples = []
    n_triplet_each_rel = int(n_ent * n_ent * pos_ratio)
    candidates = list(itertools.combinations([i for i in range(n_ent)], 2))
    for r in range(n_rel):
        if int(n_rel*sym_ratio)<=r: # asymmetry relation
            rand_idxs = np.random.permutation(len(candidates))
            for idx in rand_idxs[:n_triplet_each_rel]:
                if np.random.randint(2):
                    samples.append([candidates[idx][0], r, candidates[idx][1]])
                else:
                    samples.append([candidates[idx][1], r, candidates[idx][0]])
        else: # symmetry relation
            rand_idxs = np.random.permutation(len(candidates))
            for idx in rand_idxs[:n_triplet_each_rel//2]:
                samples.append([candidates[idx][0], r, candidates[idx][1]])
                samples.append([candidates[idx][1], r, candidates[idx][0]])
    if split:
        train, dev_test = train_test_split(samples, test_size=0.2)
        dev, test = train_test_split(dev_test, test_size=0.5)
        return train, dev, test
    else:
        return samples


def generate_labeled_synthetic(n_ent, n_rel, pos_ratio, sym_ratio=0.5, split=False):
    samples = []
    n_triplet_each_rel = int(n_ent * n_ent * pos_ratio)
    candidates = list(itertools.combinations([i for i in range(n_ent)], 2))
    for r in range(n_rel):
        if int(n_rel*sym_ratio)<=r: # asymmetry relation
            rand_idxs = np.random.permutation(len(candidates))
            for idx in rand_idxs[:n_triplet_each_rel]:
                if np.random.randint(2):
                    samples.append([candidates[idx][0], r, candidates[idx][1], 1])
                    samples.append([candidates[idx][1], r, candidates[idx][0], -1])
                else:
                    samples.append([candidates[idx][1], r, candidates[idx][0], 1])
                    samples.append([candidates[idx][0], r, candidates[idx][1], -1])
            for idx in rand_idxs[n_triplet_each_rel:]:
                samples.append([candidates[idx][0], r, candidates[idx][1], -1])
                samples.append([candidates[idx][1], r, candidates[idx][0], -1])
        else: # symmetry relation
            rand_idxs = np.random.permutation(len(candidates))
            for idx in rand_idxs[:n_triplet_each_rel//2]:
                samples.append([candidates[idx][0], r, candidates[idx][1], 1])
                samples.append([candidates[idx][1], r, candidates[idx][0], 1])
            for idx in rand_idxs[n_triplet_each_rel//2:]:
                samples.append([candidates[idx][0], r, candidates[idx][1], -1])
                samples.append([candidates[idx][1], r, candidates[idx][0], -1])
    if split:
        train, dev_test = train_test_split(samples, test_size=0.2)
        dev, test = train_test_split(dev_test, test_size=0.5)
        return train, dev, test
    else:
        return samples
