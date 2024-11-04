import torch

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vizsequence.viz_sequence import plot_weights

# import sample from ../diffusion/sample.py
import sys
sys.path.append('/home/spour98/projects/def-thughes/spour98/ledidi/ledidi/ledidi/dna_diffusion')

from sample import sample
from utils import decode_sequences

class DiffusionModelInterpreter:
    def __init__(self, model, diffusion, data_generator, device='cpu'):
        self.model = model
        self.diffusion = diffusion
        self.data_generator = data_generator
        self.device = device

    def generate_sequences(self, desired_scores, seq_len, num_samples):
        """
        Generates sequences based on specified target scores, evaluates them, and decodes them.
        """
        generated_x0 = sample(self.model, self.diffusion, seq_len, desired_scores, self.device, num_samples=num_samples)
        generated_sequences = decode_sequences(generated_x0.cpu())
        
        # Analyze generated sequences for motifs, GC content, and scores
        result_df = self.analyze_sequences(generated_sequences)
        return result_df

    def analyze_sequences(self, sequences):
        """
        Analyzes each sequence for motifs, GC content, PWM scores, and relevant metrics.
        Returns a DataFrame with analysis results.
        """
        sequence_data = []
        
        for seq in sequences:
            gc_content = (seq.count("G") + seq.count("C")) / len(seq)
            motif_count = seq.count(self.data_generator.consensus) / (len(seq) - len(self.data_generator.consensus) + 1)
            pwm_score = self.data_generator.score_sequence(seq)

            sequence_data.append({
                'Sequence': seq,
                'GC_Content': gc_content,
                'Motif_Count': motif_count,
                'PWM_Score': pwm_score
            })
        
        return pd.DataFrame(sequence_data)

    def visualize_gradients(self, sequence, t, score):
        """
        Calculates gradients for each nucleotide position in a given sequence based on the model output.
        Visualizes these gradients to highlight which nucleotides are important for scoring.
        """
        sequence_tensor = torch.tensor(sequence, dtype=torch.float).unsqueeze(0).to(self.device).requires_grad_(True)
        t_tensor = torch.tensor([t], device=self.device)
        score_tensor = torch.tensor([score], device=self.device)
        
        output = self.model(sequence_tensor, t_tensor, score_tensor)
        output.sum().backward()
        
        gradients = sequence_tensor.grad.squeeze().cpu().numpy()
        plot_weights(gradients, figsize=(15, 2))
        plt.title("Gradient Visualization for Sequence Importance")
        plt.savefig('/home/spour98/projects/def-thughes/spour98/ledidi/ledidi/ledidi/figs/gradient_visualization_logo.png')
        plt.show()
        plt.imshow(gradients, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.xlabel("Nucleotide (A, C, G, T)")
        plt.ylabel("Position in Sequence")
        plt.title("Gradient Visualization for Sequence Importance")
        plt.savefig('/home/spour98/projects/def-thughes/spour98/ledidi/ledidi/ledidi/figs/gradient_visualization.png')
        plt.show()

    def interpret_self_conditioning_effect(self, sequence, t, score):
        """
        Analyzes the effect of self-conditioning on sequence generation by generating sequences
        with and without self-conditioning.
        """
        sequence_tensor = torch.tensor(sequence, dtype=torch.float).unsqueeze(0).to(self.device)
        t_tensor = torch.tensor([t], device=self.device)
        score_tensor = torch.tensor([score], device=self.device)
        
        # Without self-conditioning
        output_no_self_cond = self.model(sequence_tensor, t_tensor, score_tensor, x_self_cond=None)
        
        # With self-conditioning
        with torch.no_grad():
            x_0_pred = self.model(sequence_tensor, t_tensor, score_tensor, x_self_cond=None).softmax(dim=-1)
        output_with_self_cond = self.model(sequence_tensor, t_tensor, score_tensor, x_self_cond=x_0_pred)

        diff = (output_with_self_cond - output_no_self_cond).abs().mean().item()
        print(f"Difference in outputs (with vs without self-conditioning): {diff:.4f}")

        return output_no_self_cond, output_with_self_cond

    def compute_pwm_importance(self, generated_sequences, pwm):
        """
        Analyzes the importance of each position in the Position Weight Matrix (PWM) by computing
        sequence scores and visualizing the positional importance.
        """
        pwm_length = len(next(iter(pwm.values())))
        pwm_scores = []
        
        for seq in generated_sequences:
            pwm_score = []
            for i in range(len(seq) - pwm_length + 1):
                score = 0
                for j, nucleotide in enumerate(seq[i:i + pwm_length]):
                    score += pwm[nucleotide][j]
                pwm_score.append(score)
            pwm_scores.append(np.mean(pwm_score))

        plt.plot(pwm_scores)
        plt.title("Average Positional PWM Scores Across Generated Sequences")
        plt.xlabel("Sequence Position")
        plt.ylabel("Average PWM Score")
        # save to /project/6000369/spour98/ledidi/ledidi/ledidi/figs
        plt.savefig('/home/spour98/projects/def-thughes/spour98/ledidi/ledidi/ledidi/figs/pwm_scores.png')
        plt.show()


    def motif_importance_analysis(self, sequence, motif):
        """
        Checks if the motif exists within generated sequences and its contribution to the score.
        """
        motif_len = len(motif)
        motif_counts = [seq.count(motif) for seq in sequence]
        avg_motif_count = np.mean(motif_counts)
        
        print(f"Average Motif Count in Generated Sequences: {avg_motif_count}")
        return motif_counts
    
    def deep_shap_attributions(self, input_sequence, baseline_sequences, t, score, num_steps=50):
        input_sequence = input_sequence.to(self.device).requires_grad_(True)
        baseline_sequences = baseline_sequences.to(self.device)
        t = t.to(self.device)
        score = score.to(self.device)
        self.model.to(self.device)
        self.model.eval()

        num_baselines = baseline_sequences.shape[0]
        attributions = torch.zeros_like(input_sequence[0], device=self.device).unsqueeze(0)
        alphas = torch.linspace(0, 1, steps=num_steps).to(self.device).view(num_steps, 1, 1)

        for baseline in baseline_sequences:
            delta_X = input_sequence - baseline.unsqueeze(0)
            interpolated_inputs = baseline.unsqueeze(0) + alphas * delta_X
            interpolated_inputs.requires_grad_(True)

            t_batch = t.repeat(num_steps)
            score_batch = score.repeat(num_steps)
            
            outputs = self.model(interpolated_inputs, t_batch, score_batch).sum(dim=(1, 2))

            # Calculate gradients and add debugging to check gradient flow
            grads = torch.autograd.grad(outputs.sum(), interpolated_inputs)[0]
            if grads.abs().sum().item() == 0:
                print("Warning: Zero gradients detected. Check model and inputs.")
            
            avg_grads = grads.mean(dim=0, keepdim=True)
            attributions += avg_grads * delta_X

        attributions /= num_baselines
        return attributions.detach().cpu()


    def generate_baseline_sequences(self, seq_len, num_classes, num_baselines):
        """
        Generate baseline sequences for DeepSHAP.

        Args:
            seq_len (int): Length of the sequences.
            num_classes (int): Number of classes (nucleotides).
            num_baselines (int): Number of baseline sequences to generate.

        Returns:
            baseline_sequences (torch.Tensor): Tensor of shape [num_baselines, seq_len, num_classes].
        """
        # uniformbaseline should be random one-hot sequences
        baseline_seqs = []
        for _ in range(num_baselines):
            baseline_seq = torch.randint(0, num_classes, (seq_len,))
            baseline_one_hot = F.one_hot(baseline_seq, num_classes).float()
            baseline_seqs.append(baseline_one_hot)
        baseline_sequences = torch.stack(baseline_seqs)
        return baseline_sequences

    def visualize_attributions(self, attributions, sequence):
        """
        Visualize attributions as a heatmap.

        Args:
            attributions (torch.Tensor): Attributions tensor of shape [seq_len, num_classes].
            sequence (str): Original sequence as a string.
        """
        plot_weights(attributions.numpy(), figsize=(15, 2))
        plt.title("Attributions for Sequence Importance")
        plt.xlabel("Nucleotide (A, C, G, T)")
        plt.ylabel("Position in Sequence")
        plt.savefig('/home/spour98/projects/def-thughes/spour98/ledidi/ledidi/ledidi/figs/attributions.png')

    def interpret_sequence(self, sequence_str, t, score, num_baselines=5, num_samples=10):
        """
        Interpret a sequence using the DeepSHAP-inspired method.

        Args:
            sequence_str (str): Input sequence as a string (e.g., "ACGT...").
            t (int): Time step.
            score (float): Score value.
            num_baselines (int): Number of baseline sequences.
            num_samples (int): Number of samples for approximation.

        Returns:
            attributions (torch.Tensor): Attributions tensor of shape [seq_len, num_classes].
        """
        # One-hot encode the input sequence
        input_sequence = self.one_hot_encode_sequence(sequence_str).unsqueeze(0)  # Shape: [1, seq_len, num_classes]

        # Generate baseline sequences
        seq_len, num_classes = input_sequence.shape[1], input_sequence.shape[2]
        baseline_sequences = self.generate_baseline_sequences(seq_len, num_classes, num_baselines)

        # Prepare tensors
        t_tensor = torch.tensor([t], dtype=torch.long)
        score_tensor = torch.tensor([score], dtype=torch.float32)

        # Compute attributions
        attributions = self.deep_shap_attributions(input_sequence, baseline_sequences, t_tensor, score_tensor, num_steps=50)

        # Visualize attributions
        self.visualize_attributions(attributions, sequence_str)

        return attributions

    def one_hot_encode_sequence(self, sequence_str):
        """
        One-hot encode a DNA sequence string.

        Args:
            sequence_str (str): DNA sequence (e.g., "ACGT").

        Returns:
            one_hot_tensor (torch.Tensor): One-hot encoded tensor of shape [seq_len, num_classes].
        """
        nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq_len = len(sequence_str)
        num_classes = 4  # A, C, G, T

        one_hot_array = np.zeros((seq_len, num_classes), dtype=np.float32)
        for i, nucleotide in enumerate(sequence_str):
            index = nucleotide_to_index.get(nucleotide, -1)
            if index >= 0:
                one_hot_array[i, index] = 1.0
            else:
                # Handle unknown nucleotides (e.g., N)
                one_hot_array[i] = np.ones(num_classes) / num_classes

        one_hot_tensor = torch.tensor(one_hot_array)
        return one_hot_tensor



# desired_scores = torch.linspace(0, 1, steps=5, device=device)  # Adjust steps as needed
# num_samples_per_score = 200  # Number of sequences per score

# all_generated_sequences = []

# for score_value in desired_scores:
#     desired_score = torch.full((num_samples_per_score,), score_value, device=device)
#     generated_x0 = sample(model, diffusion, seq_len, desired_score, device, num_samples=num_samples_per_score)
#     sequences = decode_sequences(generated_x0.cpu())
#     all_generated_sequences.append((score_value.item(), sequences))

# # Step 2-3: Compute PFMs and PWMs
# def compute_pfm(sequences, seq_len):
#     nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
#     pfm = np.zeros((seq_len, 4))
#     for seq in sequences:
#         for i, nucleotide in enumerate(seq):
#             if nucleotide in nucleotide_to_index:
#                 idx = nucleotide_to_index[nucleotide]
#                 pfm[i, idx] += 1
#     pfm /= len(sequences)
#     return pfm

# def compute_pwm(pfm, background_freq=None):
#     if background_freq is None:
#         background_freq = np.array([0.25, 0.25, 0.25, 0.25])
#     pwm = np.log2((pfm + 1e-6) / background_freq)
#     return pwm

# pwms = []
# for score_value, sequences in all_generated_sequences:
#     pfm = compute_pfm(sequences, seq_len)
#     pwm = compute_pwm(pfm)
#     pwms.append((score_value, pwm))