# main.py

from dna_diffusion.dataset import DNASequenceDataset, ProgressiveDifficultyDataset
from torch.utils.data import DataLoader
from dna_diffusion.diffusion import DiscreteDiffusion
from dna_diffusion.model import DiffusionModel
from dna_diffusion.train import train, train_progressive
from dna_diffusion.sample import sample
from dna_diffusion.utils import decode_sequences, one_hot_encode
from dna_diffusion.data_generator import DummyDataGenerator, DummyMotifCounter, DummyPWMScorer, BEDFileDataGenerator
from dna_diffusion.interpret import DiffusionModelInterpreter
import torch
import torch.optim as optim
from captum.attr import IntegratedGradients
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import pyfaidx

def collate_fn(batch):
    sequences, scores = zip(*batch) 

    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)  
    scores = torch.stack(scores) 

    return padded_sequences, scores

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in reversed(seq)])

def main():
    # dummy data generator
    MOTIF = 'GACGTT'
    PWM = {
        'A': [0.1, 0.2, 0.1, 0.8, 0.8],
        'C': [0.6, 0.6, 0.7, 0.1, 0.1],
        'G': [0.2, 0.1, 0.1, 0.1, 0.1],
        'T': [0.1, 0.1, 0.1, 0.0, 0.0]
    }
    MODE = "NOT PROGRESSIVE"
    num_sequences = 5000  
    sequence_length = 90  
    # data_generator = DummyDataGenerator(num_sequences, sequence_length)
    # data_generator = DummyMotifCounter(num_sequences, sequence_length, motif=MOTIF)
    data_generator = DummyPWMScorer(num_sequences, sequence_length, pwm=PWM)
    dataframe = data_generator.data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = sequence_length
    ## round 2 of different length
    sequence_length2 = 100
    data_generator2 = DummyPWMScorer(num_sequences, sequence_length2, pwm=PWM)
    dataframe2 = data_generator2.data
    
    dataframe = pd.concat([dataframe, dataframe2], ignore_index=True) 
     
    # bed version 
    dataframe = pd.read_csv('/project/6000369/spour98/ledidi/data/ENCFF612ZUY.bedout', sep='\t', header=None)
    # dataframe = dataframe[[4, 10]]
    # dataframe = dataframe.rename(columns={4: 'scores', 10: 'sequences'})
    # # sequences to uppercase
    # dataframe['sequences'] = dataframe['sequences'].str.upper()
    # using BEDFileDataGenerator
    data_generator = BEDFileDataGenerator('/project/6000369/spour98/ledidi/data/ENCFF612ZUY.bedout', 10000)
    dataframe = data_generator.data
    num_classes = 4
    hidden_size = 128
    T = 100
    epochs = 2
    sequences = dataframe['sequences'].values
    scores = dataframe['scores'].values
    # model and diffusion
    # model = DiffusionModel(seq_len=seq_len, num_classes=num_classes, hidden_size=hidden_size).to(device)
    model = DiffusionModel(num_classes=num_classes, hidden_size=hidden_size, num_layers=6, num_heads=8)
    diffusion = DiscreteDiffusion(num_classes=num_classes, T=T)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    if not MODE == "PROGRESSIVE":
        dataset = DNASequenceDataset(sequences, scores)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        train(model, diffusion, dataloader, optimizer, epochs, device)
    elif MODE == "PROGRESSIVE":
        initial_ratio = 0.1
        final_ratio = 0.5
        high_score_ratio_schedule = np.linspace(initial_ratio, final_ratio, epochs)
        dataset = ProgressiveDifficultyDataset(sequences, scores)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        train_progressive(model, diffusion, dataloader, optimizer, device, epochs, high_score_ratio_schedule)



    # trainingggg

    # sample sequences and generate
    # try multiple desired_score_values
    desired_score = torch.tensor([0.1, 0.3, 0.6, 1], device=device)
    generated_x0 = sample(model, diffusion, seq_len, desired_score, device, num_samples=len(desired_score))
    generated_sequences = decode_sequences(generated_x0.cpu())
    print("Generated sequences:")
    for seq in generated_sequences:
        print(seq)
        print(f"GC content: {seq.count('G') + seq.count('C')}/{len(seq)}")
        print(f"Motif count: {seq.count(MOTIF) / (len(seq) -  len(MOTIF) + 1)}")
        # print(f"Score: {data_generator.score_sequence(seq)}")
    
    # random sequence
    random_onehot = torch.randint(0, 4, (1, seq_len), device=device)
    random_onehot = torch.nn.functional.one_hot(random_onehot, num_classes=num_classes).float()
    sequence = random_onehot.requires_grad_(True)

    t = torch.tensor([0]).cuda()
    score = torch.tensor([217]).float().cuda()
    
    target_indices = torch.argmax(sequence, dim=-1)

    def model_forward(input_seq):
        logits = model(input_seq, t, score)
        batch_size = logits.size(0)
        seq_len = logits.size(1)
        num_classes = logits.size(2)
        # Debugging prints
        print(f"input_seq shape: {input_seq.shape}")
        print(f"logits shape: {logits.shape}")
        print(f"batch_size: {batch_size}, seq_len: {seq_len}, num_classes: {num_classes}")
        print(f"target_indices shape before expansion: {target_indices.shape}")
        expanded_target_indices = target_indices.repeat(batch_size, 1)
        print(f"expanded_target_indices shape: {expanded_target_indices.shape}")  
        logits_reshaped = logits.view(-1, num_classes)
        target_indices_flat = expanded_target_indices.contiguous().view(-1)
        print(f"logits_reshaped shape: {logits_reshaped.shape}") 
        print(f"target_indices_flat shape: {target_indices_flat.shape}")  
        loss = F.cross_entropy(logits_reshaped, target_indices_flat, reduction='sum')
        return -loss.unsqueeze(0)  # Return scalar output

    ig = IntegratedGradients(model_forward)
    baseline = torch.zeros_like(sequence)

    attributions, delta = ig.attribute(
        inputs=sequence,
        baselines=baseline,
        target=None,
        return_convergence_delta=True,
        # additional_forward_args=additional_args
    )


    attr = attributions[0].cpu().detach().numpy()  
    attr_magnitude = np.abs(attr)
    # sum across columns
    attr_sum = np.average(attr, axis=0)
    
    # interpret

    breakpoint()
    interpreter = DiffusionModelInterpreter(model, diffusion, data_generator, device=device)
    desired_scores = torch.tensor([0.0, 0.3, 0.6, 1.0], device=device)  # Define desired scores
    # 10 sequences of score 1000
    # desired_scores = torch.tensor([100.0], device=device) * torch.ones(10, device=device)
    generated_df = interpreter.generate_sequences(desired_scores, 100, num_samples=len(desired_scores))
    # count CCA**AGGGGGCG in generated sequences USING COUNT
    # generated_df['MOTIF_count'] = generated_df['Sequence'].apply(lambda x: len(re.findall('GGGGG', x)))
    # dataframe['CCAC count'] = dataframe['sequences'].apply(lambda x: len(re.findall('GGGGG', x)))
    print("Generated Sequences and Analysis:")
    print(generated_df)
    import grelu.interpret.motifs
    comparison = grelu.interpret.motifs.compare_motifs(
        ref_seq = generated_df["Sequence"][0],
        alt_seq = generated_df["Sequence"].tolist()[-1],
        motifs="consensus",
        pthresh=5e-5,
        rc=True, # Scan both strands of the sequence
    )
    comparison[comparison["motif"].str.contains("CTCF")]
    breakpoint()
    example_sequence = dataframe['sequences'].iloc[-1]  # Example sequence from dataset
    example_encoded = one_hot_encode([example_sequence], len(example_sequence))[0]
    interpreter.visualize_gradients(example_encoded, t=5, score=1.0)  # Example t and score

    # selc ondition
    output_no_self_cond, output_with_self_cond = interpreter.interpret_self_conditioning_effect(example_encoded, t=50, score=0.5)
    print(f"Output without self-conditioning:\n{output_no_self_cond}")
    print(f"Output with self-conditioning:\n{output_with_self_cond}")
    
    interpreter.compute_pwm_importance(generated_df['Sequence'].tolist(), PWM)
    
    motif = data_generator.consensus 
    motif_counts = interpreter.motif_importance_analysis(generated_df['Sequence'].tolist(), motif)
    print(f"Motif occurrence counts in generated sequences:\n{motif_counts}")
    
    attr = interpreter.interpret_sequence(example_sequence, t, score, num_baselines=50, num_samples=50)
    # softmax over attr
    attr_softmax = F.softmax(attr, dim=-1)
    interpreter.visualize_attributions(attr_softmax, example_sequence)
    
    breakpoint()
if __name__ == '__main__':
    main()
