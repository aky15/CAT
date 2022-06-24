import numpy as np
import h5py
import torch
import kaldi_io
import kaldiio
from torch.utils.data import Dataset, DataLoader
import sys

class SpeechDataset(Dataset):
    def __init__(self, h5py_path,augment=None):
        self.h5_path = h5py_path
        self.augment = augment

    def __len__(self):
        with h5py.File(self.h5_path,'r') as record:
            keys = list(record.keys())
        return len(keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path,'r') as record:
            dataset = record[list(record.keys())[idx]]
            mat = dataset[()]
            label = dataset.attrs['label']
            weight = dataset.attrs['weight']
        if self.augment:
            #print("augment!")
            #print(mat)
            #print(self.augment(torch.FloatTensor(mat).transpose(0,1)).transpose(0,1))
            return self.augment(torch.FloatTensor(mat).transpose(0,1)).transpose(0,1), torch.IntTensor(label), torch.FloatTensor(weight)
        else:
            #print("no augment!")
            return torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)

class SpeechDatasetMem(Dataset):
    def __init__(self, h5py_path, augment=None):
        hdf5_file = h5py.File(h5py_path, 'r')
        keys = hdf5_file.keys()
        self.data_batch = []
        for key in keys:
          dataset = hdf5_file[key]
          mat = dataset[()]
          label = dataset.attrs['label']
          weight = dataset.attrs['weight']
          self.data_batch.append([torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)])
        
        self.augment = augment
        hdf5_file.close()
        print("read all data into memory")

    def __len__(self):
        return len(self.data_batch)

    def __getitem__(self, idx):
        #print(self.data_batch[idx])
        #return self.data_batch[idx]
        if self.augment:
            return self.augment(self.data_batch[idx][0].transpose(0,1)).transpose(0,1), self.data_batch[idx][1], self.data_batch[idx][2]
        else:
            return self.data_batch[idx]

if sys.version > '3':
    import pickle
else:
    import cPickle as pickle

class FeatureReader:
    def __init__(self) -> None:
        self._opened_fd = {}

    def __call__(self, arkname: str):
        return kaldiio.load_mat(arkname, fd_dict=self._opened_fd)

    def __del__(self):
        for f in self._opened_fd.values():
            f.close()
        del self._opened_fd

class Augmentor(object):
    def __init__(self):
        super(Augmentor, self).__init__()

    def __call__(self, input):
        raise NotImplementedError

class SpeechDatasetPickle(Dataset):
    def __init__(self, pickle_path, augment=None):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)
        self.freader = FeatureReader()
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key, feature_path, label, weight = self.dataset[idx]
        #print(feature_path)
        #mat = np.asarray(kaldi_io.read_mat(feature_path))
        #return torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)
        mat = self.freader(feature_path)
        if self.augment:
            return self.augment(torch.tensor(mat, dtype=torch.float).transpose(0,1)).transpose(0,1), torch.IntTensor(label), torch.tensor(weight, dtype=torch.float)
        else:
            return torch.tensor(mat, dtype=torch.float), torch.IntTensor(label), torch.tensor(weight, dtype=torch.float)

class SpeechDatasetMemPickle(Dataset):
    def __init__(self, pickle_path, augment=None):
        with open(pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.data_batch = []

        for data in self.dataset:
            key, feature_path, label, weight = data
            mat = np.asarray(kaldi_io.read_mat(feature_path))
            self.data_batch.append(
                [torch.FloatTensor(mat), torch.IntTensor(label), torch.FloatTensor(weight)])
        self.augment = augment

    def __len__(self):
        return len(self.data_batch)

    def __getitem__(self, idx):
        if self.augment:
            return self.augment(self.data_batch[idx][0].transpose(0,1)).transpose(0,1), self.data_batch[idx][1], self.data_batch[idx][2]
        else:
            return self.data_batch[idx]

def pad_tensor(t, pad_to_length, dim):
    pad_size = list(t.shape)
    pad_size[dim] = pad_to_length - t.size(dim)
    return torch.cat([t, torch.zeros(*pad_size).type_as(t)], dim=dim)

class PadCollate:
    def __init__(self):
        pass
    def __call__(self, batch):
        # batch: list of (mat, label, weight)
        # return: logits, input_lengths, label_padded, label_lengths, weights
        input_lengths = map(lambda x: x[0].size(0), batch)
        if sys.version > '3':
            input_lengths = list(input_lengths)
        max_input_length = max(input_lengths)
        label_lengths = map(lambda x: x[1].size(0), batch)
        if sys.version > '3':
            label_lengths = list(label_lengths)
        max_label_length = max(label_lengths)
        input_batch = map(lambda x:pad_tensor(x[0], max_input_length, 0), batch)
        label_batch = map(lambda x:pad_tensor(x[1], max_label_length, 0), batch)
        if sys.version > '3':
            input_batch = list(input_batch)
            label_batch = list(label_batch)
        logits = torch.stack(input_batch, dim=0)
        label_padded = torch.stack(label_batch, dim=0)
        input_lengths = torch.IntTensor(input_lengths)
        label_lengths = torch.IntTensor(label_lengths)
        weights = torch.FloatTensor([x[2] for x in batch])
        return logits, input_lengths, label_padded, label_lengths, weights

class SpeechDatasetAPC(Dataset):
    def __init__(self, scp_path, chunk_size, percent=1.0, normalize=True):
        self.data_batch = []
        self.chunk_size = chunk_size
        self.percent = percent
        self.normalize = normalize

        with open(scp_path) as f:
            for line in f.readlines():
                #print(line)
                key, value = line.split()
                mat = np.asarray(kaldi_io.read_mat(value))
                mat_segments = self._apply_feat_preprocess(mat)
                for seg in mat_segments:
                    self.data_batch.append([torch.FloatTensor(seg[0])])

        print("read all data into memory")

    def __len__(self):
        return int(self.percent * len(self.data_batch))

    def __getitem__(self, idx):
        return self.data_batch[idx]

    def _apply_feat_preprocess(self, feature):
        feat_normX =  _gaussin_normalize(feature) if self.normalize else feature
        feat_segmentsX = self._chunk_slice(feat_normX) if self.chunk_size > 0 else [[feat_normX]]
        return feat_segmentsX

    def _chunk_slice(self, feat):
        chunk_size = self.chunk_size
        n_frames = feat.shape[0]
        split_chunks = []

        for i in range(0, n_frames, chunk_size):
            feat_seg = feat[i:i + chunk_size, :]
            if feat_seg.shape[0] < chunk_size:
                continue
            split_chunks.append([feat_seg])

        return split_chunks

    def pad_collate_fn(self, batch):
        feature_lengths = map(lambda x: x[0].size(0), batch)
        if sys.version > '3':
            feature_lengths = list(feature_lengths)
        max_input_length = max(feature_lengths)

        input_batch = map(lambda x:pad_tensor(x[0], max_input_length, 0), batch)
        if sys.version > '3':
            input_batch = list(input_batch)
        features = torch.stack(input_batch, dim=0)
        feature_lengths = torch.IntTensor(feature_lengths)

        return features, feature_lengths
