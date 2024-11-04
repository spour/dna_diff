# dna_diffusion/data_generator.py

import numpy as np
import pandas as pd
import random

class DummyDataGenerator:
    def __init__(self, num_sequences, sequence_length):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.data = self.generate_data()

    def generate_sequence(self):
        return ''.join(random.choices(['A', 'C', 'G', 'T'], k=self.sequence_length))

    def compute_gc_content(self, sequence):
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)

    def generate_data(self):
        sequences = []
        scores = []
        for _ in range(self.num_sequences):
            seq = self.generate_sequence()
            score = self.compute_gc_content(seq)
            sequences.append(seq)
            scores.append(score)
        dataframe = pd.DataFrame({'sequences': sequences, 'scores': scores})
        return dataframe

class DummyMotifCounter:
    def __init__(self, num_sequences, sequence_length, motif):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.motif = motif
        self.data = self.generate_data()
        
    def generate_sequence(self):
        # create distribution of sequences that have 0-3 motifs
        num_motifs = np.random.choice([0, 0, 0, 10], p=[0.4, 0.3, 0.25, 0.05])
        num_motifs = min(num_motifs, self.sequence_length // len(self.motif))
        sequence = ''
        for _ in range(num_motifs):
            sequence += self.motif
        sequence += ''.join(random.choices(['A', 'C', 'G', 'T'], k=self.sequence_length - len(sequence)))
        return sequence
    
    def count_motif(self, sequence, motif):
        return sequence.count(motif) / (len(sequence) -  len(motif) + 1) 
    
    def generate_data(self):
        sequences = []
        counts = []
        for _ in range(self.num_sequences):
            seq = self.generate_sequence()
            count = self.count_motif(seq, self.motif)
            sequences.append(seq)
            counts.append(count)
        dataframe = pd.DataFrame({'sequences': sequences, 'scores': counts})
        return dataframe
    
    
class DummyPWMScorer:
    def __init__(self, num_sequences, sequence_length, pwm):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.pwm = pwm
        self.consensus = self.get_consensus_sequence()
        self.data = self.generate_data()
        
    def generate_sequence(self, p_match=0.4, p_mismatch=0.3, p_random=0.3):
        """Generate a sequence with varying probabilities for matching the consensus sequence,
        mismatching by one position, or being fully random."""
        
        sequence = ''
        
        for i in range(self.sequence_length):
            prob = random.random()
            if prob < p_match:
                # Match consensus nucleotide at position i if i < len(consensus)
                sequence += self.consensus[i % len(self.consensus)]
            elif prob < p_match + p_mismatch:
                # Mismatch by choosing a random nucleotide different from consensus at position i
                mismatch_choices = [nt for nt in self.pwm.keys() if nt != self.consensus[i % len(self.consensus)]]
                sequence += random.choice(mismatch_choices)
            else:
                # Choose any random nucleotide
                sequence += random.choice(list(self.pwm.keys()))
                
        return sequence

    def score_sequence(self, sequence):
        """Calculate the score of the sequence by sliding the PWM across it."""
        pwm_length = len(next(iter(self.pwm.values())))
        score = 0
        
        # Slide the PWM across the sequence
        for i in range(len(sequence) - pwm_length + 1):
            window_score = 0
            for j, nucleotide in enumerate(sequence[i:i + pwm_length]):
                window_score += self.pwm[nucleotide][j]
            score += window_score
            
        return score
    
    def get_consensus_sequence(self):
        """Get the consensus sequence of the PWM."""
        consensus = ''
        for i in range(len(next(iter(self.pwm.values())))):
            max_score = -float('inf')
            max_nucleotide = None
            for nucleotide, scores in self.pwm.items():
                if scores[i] > max_score:
                    max_score = scores[i]
                    max_nucleotide = nucleotide
            consensus += max_nucleotide
        return consensus
    
    def generate_data(self):
        """Generate sequences and their scores, returning a DataFrame."""
        sequences = []
        scores = []
        for _ in range(self.num_sequences):
            seq = self.generate_sequence()
            score = self.score_sequence(seq)
            sequences.append(seq)
            scores.append(score)
        dataframe = pd.DataFrame({'sequences': sequences, 'scores': scores})
        # normalize scores to be between 0 and 1
        dataframe['scores'] = (dataframe['scores'] - dataframe['scores'].min()) / (dataframe['scores'].max() - dataframe['scores'].min())
        return dataframe
