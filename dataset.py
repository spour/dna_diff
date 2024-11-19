# dna_diffusion/dataset.py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class DNASequenceDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores

        # Convert sequences to one-hot encoding
        self.one_hot_sequences = [self.one_hot_encode(seq) for seq in self.sequences]
        # Convert scores to tensors
        self.scores = torch.tensor(self.scores, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.one_hot_sequences[idx]
        score = self.scores[idx]
        return seq, score

    def one_hot_encode(self, seq):
        # Map nucleotides to integers
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq_int = [mapping[nuc] for nuc in seq]
        seq_tensor = torch.tensor(seq_int, dtype=torch.long)
        # One-hot encode
        one_hot = F.one_hot(seq_tensor, num_classes=4)  # shape: [seq_len, 4]
        return one_hot.float()



class ProgressiveDifficultyDataset(Dataset):
    def __init__(self, sequences, scores, initial_high_score_ratio=0.1):
        self.sequences = sequences
        self.scores = scores
        # Identify high and low scoring indices based on percentile
        self.high_score_indices = np.where(scores >= np.percentile(scores, 90))[0]  # Top 10%
        self.low_score_indices = np.where(scores < np.percentile(scores, 90))[0]
        self.high_score_ratio = initial_high_score_ratio  # Initial ratio of high-scoring examples

    def set_high_score_ratio(self, ratio):
        self.high_score_ratio = ratio

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Decide whether to sample a high or low scoring example
        if np.random.rand() < self.high_score_ratio and len(self.high_score_indices) > 0:
            idx = np.random.choice(self.high_score_indices)
        else:
            idx = np.random.choice(self.low_score_indices)

        sequence = self.sequences[idx]
        score = self.scores[idx]

        # One-hot encode the sequence
        sequence_encoded = self.one_hot_encode_sequence(sequence)

        return torch.tensor(sequence_encoded, dtype=torch.float32), torch.tensor(score, dtype=torch.float32)
    
    def one_hot_encode_sequence(self, sequence):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        sequence_int = [mapping[nuc] for nuc in sequence]
        sequence_tensor = torch.tensor(sequence_int, dtype=torch.long)
        one_hot = F.one_hot(sequence_tensor, num_classes=4)
        return one_hot.float()
