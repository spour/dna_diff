# dna_diffusion/sample.py

import torch
import torch.nn.functional as F

@torch.no_grad()
def sample(model, diffusion, seq_len, score, device, num_samples=1):
    # Start from random noise
    batch_size = num_samples
    x_t = torch.full((batch_size, seq_len, diffusion.num_classes), 1.0 / diffusion.num_classes, device=device)

    for t in reversed(range(diffusion.T)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        logits = model(x_t, t_tensor, score)
        probs = F.softmax(logits, dim=-1)
        x_t = torch.distributions.Categorical(probs=probs).sample()
        x_t = F.one_hot(x_t, num_classes=diffusion.num_classes).float()

    x_0 = x_t
    return x_0
