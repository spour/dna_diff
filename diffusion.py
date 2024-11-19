# # dna_diffusion/diffusion.py


import torch
import torch.nn.functional as F
from functools import partial
import math
import numpy as np


def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    alpha_bar = torch.cos(((steps / timesteps) + s) / (1 + s) * (torch.pi / 2)) ** 2
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, min=0.0001, max=0.02)

def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02,  s = 0.008, type='linear',):
    if type == 'linear':
        return torch.linspace(beta_start, beta_end, T)
    elif type == 'cosine':
        return cosine_beta_schedule(T, s=s)



def vp_noise_schedule(timesteps):
    betas = torch.linspace(1e-4, 0.02, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sigma = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)  # Variance-preserving noise
    return betas, sigma

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape, device=None):
    if device:
        a = a.to(device)
        t = t.to(device)
    out = a.gather(-1, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))

class DiscreteDiffusion:
    def __init__(self, model, timesteps, num_classes=4, self_conditioning=True):
        self.model = model.cuda()
        self.num_classes = num_classes
        self.T = timesteps
        self.betas = get_beta_schedule(timesteps).cuda()
        self.timesteps = timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.betas.device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.self_conditioning = self_conditioning

        # Define alphas_cumprod_prev and posterior_variance
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)


    @property
    def device(self):
        return self.betas.device

    @torch.no_grad()
    def sample(self, scores, shape, cond_weight):
        return self.p_sample_loop(scores=scores, shape=shape, cond_weight=cond_weight)

    @torch.no_grad()
    def p_sample_loop(self, scores, shape, cond_weight):
        device = self.device
        img = torch.randn(shape, device=device)  # Initialize with random noise
        imgs = []
        x_self_cond = None  # Initialize self-conditioning variable as None for the first step

        batch_size = shape[0]
        context_mask = torch.ones(batch_size, device=device)

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample_guided(
                x=img,
                t=t,
                t_index=i,
                scores=scores,
                cond_weight=cond_weight,
                context_mask=context_mask,
                x_self_cond=x_self_cond  # Pass the current self-conditioned input
            )
            imgs.append(img.cpu())

            # Update self-conditioning with the current prediction
            x_self_cond = img.detach()  # Detach to avoid computational graph buildup

        return imgs

    @torch.no_grad()
    def p_sample_guided(self, x, scores, t, t_index, context_mask, cond_weight, x_self_cond=None):
        batch_size = x.shape[0]
        device = self.device

        # Ensure scores, t, and context_mask have correct shapes
        scores = scores.view(batch_size, -1)
        t = t.view(batch_size)
        context_mask = context_mask.view(batch_size)

        # Double batch for classifier-free guidance
        x_double = torch.cat([x, x], dim=0)  # [2*batch_size, seq_len, num_classes]
        t_double = torch.cat([t, t], dim=0)
        scores_double = torch.cat([scores, scores], dim=0)
        context_mask_double = torch.cat([context_mask, torch.zeros_like(context_mask)], dim=0)

        # Handle self-conditioning for doubled batch
        if x_self_cond is not None:
            x_self_cond_double = torch.cat([x_self_cond, x_self_cond], dim=0)
        else:
            x_self_cond_double = None

        # Model prediction
        predicted_noise = self.model(
            x_double,
            t_double,
            scores_double,
            x_self_cond=x_self_cond_double,  # Pass the self-conditioning
            attention_mask=None
        )

        # Split predictions
        eps_cond, eps_uncond = predicted_noise.chunk(2, dim=0)

        # Classifier-free guidance
        x_t = eps_uncond + cond_weight * (eps_cond - eps_uncond)

        # Compute model mean
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * x_t / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise



    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, torch.randn_like(x_start))
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def p_losses(self, x_start, t, scores, mask = None, noise=None, loss_type="huber", p_uncond=0.1):
        noise = default(noise, torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Classifier-free guidance masking
        context_mask = torch.bernoulli(torch.ones(scores.shape[0], device=self.device) * (1 - p_uncond))

        # Mask scores
        scores_masked = scores.clone()
        scores_masked[context_mask == 0] = 0

        # Ensure scores_masked has shape [batch_size, 1]
        if scores_masked.dim() == 1:
            scores_masked = scores_masked.unsqueeze(-1)

        # Model prediction
        predicted_noise = self.model(x_noisy, t, scores_masked, attention_mask = mask)

        # Loss computation
        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, scores, mask=None):
        x = x.float()
        scores = scores.float().unsqueeze(-1)
        b = x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        return self.p_losses(x, t, scores, mask = mask)
    
    def __call__(self, x, scores, mask=None):
        return self.forward(x, scores, mask)
    





# import torch
# import torch.nn.functional as F

# def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
#     return torch.linspace(beta_start, beta_end, T)

# class DiscreteDiffusion:
#     def __init__(self, num_classes, T, self_conditioning=True):
#         self.num_classes = num_classes
#         self.T = T
#         self.betas = get_beta_schedule(T)
#         self.alphas = 1.0 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
#         self.self_conditioning = self_conditioning

#     # def q_sample(self, x_0, t):
#     #     batch_size, seq_len, num_classes = x_0.shape

#     #     # cum alpha_t
#     #     # put on same device as t
#     #     self.alphas_cumprod = self.alphas_cumprod.to(t.device)
#     #     alpha_t = self.alphas_cumprod[t].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]

#     #     #uniform dist over classes
#     #     uniform_dist = torch.ones_like(x_0) / self.num_classes

#     #     # calcualte probabilities
#     #     probs = alpha_t * x_0 + (1 - alpha_t) * uniform_dist

#     #     x_t = torch.distributions.Categorical(probs=probs).sample()
#     #     # one hot
#     #     x_t_one_hot = F.one_hot(x_t, num_classes=self.num_classes).float()

#     #     return x_t_one_hot
    
#     def q_sample(self, x_0, t):
#         batch_size, seq_len, num_classes = x_0.shape

#         # Ensure alphas_cumprod is on the correct device
#         self.alphas_cumprod = self.alphas_cumprod.to(t.device)
#         alpha_t = self.alphas_cumprod[t].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]

#         # Compute probabilities
#         probs = x_0 * alpha_t + (1 - x_0) * ((1 - alpha_t) / (self.num_classes - 1))

#         # Sample x_t from the categorical distribution defined by probs
#         x_t = torch.distributions.Categorical(probs=probs).sample()
#         x_t_one_hot = F.one_hot(x_t, num_classes=self.num_classes).float()

#         return x_t_one_hot


#     def p_loss(self, model, x_0, t, score):
#         x_t = self.q_sample(x_0, t)

#         if self.self_conditioning:
#             # With 50% probability, use self-conditioning
#             use_self_cond = torch.rand([]) < 0.5
#             if use_self_cond:
#                 # Predict x_0 from x_t (without self-conditioning)
#                 with torch.no_grad():
#                     x_0_pred = model(x_t, t, score, x_self_cond=None).softmax(dim=-1)
#                     x_self_cond = x_0_pred
#             else:
#                 x_self_cond = torch.zeros_like(x_t)
#         else:
#             x_self_cond = None

#         logits = model(x_t, t, score, x_self_cond=x_self_cond)

#         # Reshape for loss
#         batch_size, seq_len, num_classes = logits.shape
#         logits = logits.view(batch_size * seq_len, num_classes)
#         target = x_0.argmax(dim=2).view(batch_size * seq_len)

#         loss = F.cross_entropy(logits, target)

#         return loss
