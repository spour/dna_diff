# dna_diffusion/utils.py
import torch 

def decode_sequences(x):
    indices = x.argmax(dim=-1)
    reverse_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    sequences = []
    for idx_seq in indices:
        seq = ''.join([reverse_mapping[int(idx)] for idx in idx_seq])
        sequences.append(seq)
    return sequences

def one_hot_encode(sequences, seq_len):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    x = torch.zeros(len(sequences), seq_len, 4)
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            x[i, j, mapping[base]] = 1
    return x