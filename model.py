# # dna_diffusion/model.py
# dna_diffusion/model.py

import torch
import torch.nn as nn
import math

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ).cuda()
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        t_emb = self.mlp(emb)
        return t_emb

class ScoreEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ).cuda()

    def forward(self, score):
        score_emb = self.mlp(score)
        return score_emb

class RelativeMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_relative_position=16):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.max_relative_position = max_relative_position

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Relative positional embeddings
        self.relative_position_bias = nn.Parameter(
            torch.zeros((2 * max_relative_position + 1, num_heads))
        )

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        # breakpoint()
        batch_size, seq_len, _ = x.size()

        # Compute queries, keys, and values
        q = self.query(x)  # [batch_size, seq_len, embed_dim]
        k = self.key(x)
        v = self.value(x)

        # Split into heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # Compute relative positional bias
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        relative_positions = position_ids.unsqueeze(0) - position_ids.unsqueeze(1)  # [seq_len, seq_len]
        relative_positions = relative_positions.clamp(
            -self.max_relative_position, self.max_relative_position
        ) + self.max_relative_position  # Shift values to be >= 0

        # Get relative position biases
        relative_bias = self.relative_position_bias[relative_positions]  # [seq_len, seq_len, num_heads]
        relative_bias = relative_bias.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]

        # Add relative position bias to attention scores
        attn_scores = attn_scores + relative_bias.unsqueeze(0)  # [batch_size, num_heads, seq_len, seq_len]

        # Compute attention probabilities
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)  # [batch_size, seq_len, embed_dim]

        return attn_output

class DiffusionTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, max_relative_position=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = RelativeMultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, max_relative_position=max_relative_position)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 4),
        ).cuda()

    def forward(self, x, t_emb, score_emb):
        c = torch.cat([t_emb, score_emb], dim=-1)
        # breakpoint()
        shift1, scale1, shift2, scale2 = self.adaLN_modulation(c).chunk(4, dim=-1)

        x_modulated = modulate(self.norm1(x), shift1, scale1)
        attn_out = self.attn(x_modulated)
        x = x + attn_out

        x = x + self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x

class DiffusionModel(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers=6, num_heads=8, self_conditioning=True, max_relative_position=16):
        super(DiffusionModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.self_conditioning = self_conditioning

        # Adjust input size based on self-conditioning
        input_size = num_classes * 2 if self_conditioning else num_classes

        self.input_embedding = nn.Linear(input_size, hidden_size).cuda()

        # Timestep and score embeddings
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.score_embedder = ScoreEmbedder(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_size, num_heads, max_relative_position).cuda() for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes).cuda()

    def forward(self, x_t, t, score, x_self_cond=None):
        """
        x_t: [batch_size, seq_len, num_classes]
        t: [batch_size]
        score: [batch_size]
        x_self_cond: [batch_size, seq_len, num_classes] (Optional)
        """
        if self.self_conditioning:
            if x_self_cond is None:
                # Set x_self_cond to zeros with the same shape as x_t
                x_self_cond = torch.zeros_like(x_t)
            x_input = torch.cat([x_t, x_self_cond], dim=-1)
        else:
            x_input = x_t
        # breakpoint()
        x = self.input_embedding(x_input)

        # Timestep and score embeddings
        t_emb = self.timestep_embedder(t)
        score_emb = self.score_embedder(score.unsqueeze(1))

        # Expand embeddings to match sequence length
        seq_len = x.size(1)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
        score_emb = score_emb.unsqueeze(1).expand(-1, seq_len, -1)

        # Transformer blocks
        for block in self.blocks:
            # breakpoint()
            x = block(x, t_emb, score_emb)

        # Output layer
        logits = self.output_layer(x)
        return logits


# import torch
# import torch.nn as nn
# import math

# def modulate(x, shift, scale):
#     return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# class TimestepEmbedder(nn.Module):
#     def __init__(self, hidden_size, frequency_embedding_size=256):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(frequency_embedding_size, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size),
#         )
#         self.frequency_embedding_size = frequency_embedding_size

#     def forward(self, t):
#         half_dim = self.frequency_embedding_size // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
#         emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
#         emb = torch.cat([emb.sin(), emb.cos()], dim=1)
#         t_emb = self.mlp(emb)
#         return t_emb

# class ScoreEmbedder(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, hidden_size),
#             nn.SiLU(),
#             nn.Linear(hidden_size, hidden_size),
#         )

#     def forward(self, score):
#         score_emb = self.mlp(score)
#         return score_emb

# class DiffusionTransformerBlock(nn.Module):
#     def __init__(self, hidden_size, num_heads):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
#         self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 4),
#             nn.GELU(),
#             nn.Linear(hidden_size * 4, hidden_size),
#         )
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size * 2, hidden_size * 4),
#         )

#     def forward(self, x, t_emb, score_emb):
#         c = torch.cat([t_emb, score_emb], dim=-1)
#         shift1, scale1, shift2, scale2 = self.adaLN_modulation(c).chunk(4, dim=-1)

#         x_modulated = modulate(self.norm1(x), shift1, scale1)
#         attn_out, _ = self.attn(x_modulated, x_modulated, x_modulated)
#         x = x + attn_out

#         x = x + self.mlp(modulate(self.norm2(x), shift2, scale2))
#         return x

# class DiffusionModel(nn.Module):
#     def __init__(self, seq_len, num_classes, hidden_size, num_layers=6, num_heads=8, self_conditioning=True):
#         super(DiffusionModel, self).__init__()
#         self.seq_len = seq_len
#         self.num_classes = num_classes
#         self.hidden_size = hidden_size
#         self.self_conditioning = self_conditioning

#         # Adjust input size based on self-conditioning
#         input_size = num_classes * 2 if self_conditioning else num_classes

#         self.input_embedding = nn.Linear(input_size, hidden_size)

#         # Timestep and score embeddings
#         self.timestep_embedder = TimestepEmbedder(hidden_size)
#         self.score_embedder = ScoreEmbedder(hidden_size)

#         # Positional embedding
#         self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=False)
#         self.initialize_positional_embedding()

#         # Transformer blocks
#         self.blocks = nn.ModuleList([
#             DiffusionTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
#         ])

#         # Output layer
#         self.output_layer = nn.Linear(hidden_size, num_classes)

#     def initialize_positional_embedding(self):
#         position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * (-math.log(10000.0) / self.hidden_size))
#         pe = torch.zeros(1, self.seq_len, self.hidden_size)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.pos_embedding.data.copy_(pe)

#     def forward(self, x_t, t, score, x_self_cond=None):
#         """
#         x_t: [batch_size, seq_len, num_classes]
#         t: [batch_size]
#         score: [batch_size]
#         x_self_cond: [batch_size, seq_len, num_classes] (Optional)
#         """
#         if self.self_conditioning:
#             if x_self_cond is None:
#                 # Set x_self_cond to zeros with the same shape as x_t
#                 x_self_cond = torch.zeros_like(x_t)
#             x_input = torch.cat([x_t, x_self_cond], dim=-1)
#         else:
#             x_input = x_t
#         x = self.input_embedding(x_input) + self.pos_embedding
#         # Timestep and score embeddings
#         t_emb = self.timestep_embedder(t)
#         score = score.unsqueeze(1)
#         score_emb = self.score_embedder(score)

#         # Transformer blocks
#         for block in self.blocks:
#             x = block(x, t_emb, score_emb)

#         # Output layer
#         logits = self.output_layer(x)
#         return logits

