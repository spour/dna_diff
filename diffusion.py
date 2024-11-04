# dna_diffusion/diffusion.py

import torch
import torch.nn.functional as F

def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

class DiscreteDiffusion:
    def __init__(self, num_classes, T, self_conditioning=True):
        self.num_classes = num_classes
        self.T = T
        self.betas = get_beta_schedule(T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.self_conditioning = self_conditioning

    def q_sample(self, x_0, t):
        batch_size, seq_len, num_classes = x_0.shape

        # cum alpha_t
        # put on same device as t
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        alpha_t = self.alphas_cumprod[t].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]

        #uniform dist over classes
        uniform_dist = torch.ones_like(x_0) / self.num_classes

        # calcualte probabilities
        probs = alpha_t * x_0 + (1 - alpha_t) * uniform_dist

        x_t = torch.distributions.Categorical(probs=probs).sample()
        # one hot
        x_t_one_hot = F.one_hot(x_t, num_classes=self.num_classes).float()

        return x_t_one_hot

    def p_loss(self, model, x_0, t, score):
        x_t = self.q_sample(x_0, t)

        if self.self_conditioning:
            # With 50% probability, use self-conditioning
            use_self_cond = torch.rand([]) < 0.5
            if use_self_cond:
                # Predict x_0 from x_t (without self-conditioning)
                with torch.no_grad():
                    x_0_pred = model(x_t, t, score, x_self_cond=None).softmax(dim=-1)
                    x_self_cond = x_0_pred
            else:
                x_self_cond = torch.zeros_like(x_t)
        else:
            x_self_cond = None

        logits = model(x_t, t, score, x_self_cond=x_self_cond)

        # Reshape for loss
        batch_size, seq_len, num_classes = logits.shape
        logits = logits.view(batch_size * seq_len, num_classes)
        target = x_0.argmax(dim=2).view(batch_size * seq_len)

        loss = F.cross_entropy(logits, target)

        return loss
