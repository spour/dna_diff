# # dna_diffusion/model.py
# dna_diffusion/model.py
# # dna_diffusion/model.py
# dna_diffusion/model.py

import torch
import torch.nn as nn
import math

def modulate(x, shift, scale):
    return x * (1 + scale) + shift
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=6, kernel_size=5, dilation=1):
        super(ConvEncoder, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=input_dim if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2 * dilation,
                    dilation=dilation
                )
            )
            layers.append(nn.ReLU())
            input_dim = hidden_dim  # Update input_dim for next layer
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, seq_len]
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, hidden_dim]
        return x


    
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
    def __init__(self, input_dim, hidden_size):
        """
        Args:
        - input_dim: Dimensionality of input scores (1 for scalar, >1 for multi-head scores)
        - hidden_size: Dimensionality of the output embedding
        """
        super(ScoreEmbedder, self).__init__()
        
        # Linear transformation for scores
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        # Normalization layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Learnable shift and scale parameters
        self.shift = nn.Parameter(torch.zeros(hidden_size))
        self.scale = nn.Parameter(torch.ones(hidden_size))
        
        # Activation
        self.activation = nn.GELU()

    def forward(self, scores):
        """
        Forward pass for score embedding.
        
        Args:
        - scores: Input tensor of shape [batch_size, input_dim]
        
        Returns:
        - score_embedding: Tensor of shape [batch_size, hidden_size]
        """
        # Pass scores through linear layers and activation
        x = self.activation(self.linear1(scores))  # Shape: [batch_size, hidden_size]
        x = self.activation(self.linear2(x))       # Shape: [batch_size, hidden_size]
        
        # Apply normalization
        x = self.layer_norm(x)  # Shape: [batch_size, hidden_size]
        
        # Apply learnable shift and scale
        x = x * self.scale + self.shift
        
        return x



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
    def __init__(self, hidden_size, num_heads, dropout=0.1, stochastic_depth_p=0.1):
        super(DiffusionTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)

        self.stochastic_depth_p = stochastic_depth_p  # Probability of skipping
        
    def forward(self, x, attn_mask=None):


        # Stochastic depth per sample
        if self.training:
            survival_prob = 1.0 - self.stochastic_depth_p
            batch_size = x.size(1)
            # Adjust the shape of random_tensor and binary_mask
            random_tensor = torch.rand(1, batch_size, 1, device=x.device) + survival_prob
            binary_mask = torch.floor(random_tensor)
            # Now binary_mask has shape [1, batch_size, 1], which can be broadcast over x
            x = x * binary_mask  # Apply mask to skip the block for some samples
        else:
            survival_prob = 1.0

        x_residual = x
        x2, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        x = x_residual + self.dropout1(x2)
        x = self.norm1(x)

        x_residual = x
        x2 = self.linear2(self.activation(self.linear1(x)))
        x = x_residual + self.dropout2(x2)
        x = self.norm2(x)

        # Scale output during training
        if self.training:
            x = x / survival_prob

        return x




class DiffusionModel(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers=6, num_heads=8, self_conditioning=True, max_seq_len=512):
        super(DiffusionModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.self_conditioning = self_conditioning

        # Adjust input size based on self-conditioning
        input_size = num_classes * 2 if self_conditioning else num_classes

        # Initialize the convolutional encoder
        self.conv_encoder = ConvEncoder(input_dim=input_size, hidden_dim=hidden_size).cuda()

        # Timestep and score embeddings
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        # self.score_embedder = nn.Linear(1, hidden_size) 
        self.score_embedder = ScoreEmbedder(1, hidden_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(self.get_sinusoid_encoding_table(max_seq_len, hidden_size), requires_grad=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_size, num_heads).cuda() for _ in range(num_layers)
        ])

        # Output layer (decoder)
        self.output_layer = nn.Linear(hidden_size, num_classes).cuda()

    def get_sinusoid_encoding_table(self, max_seq_len, hidden_size):
        def get_angle(pos, i_hidden):
            return pos / (10000 ** (2 * (i_hidden // 2) / hidden_size))
        sinusoid_table = torch.zeros(max_seq_len, hidden_size)
        for pos in range(max_seq_len):
            for i_hidden in range(hidden_size):
                angle = get_angle(pos, i_hidden)
                if i_hidden % 2 == 0:
                    sinusoid_table[pos, i_hidden] = math.sin(angle)
                else:
                    sinusoid_table[pos, i_hidden] = math.cos(angle)
        return sinusoid_table.unsqueeze(0)  # Shape: [1, max_seq_len, hidden_size]

    def forward(self, x_t, t, score, x_self_cond=None, attention_mask=None):
        if self.self_conditioning:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x_t)
            x_input = torch.cat([x_t, x_self_cond], dim=-1)  # Shape: [batch_size, seq_len, input_size]
        else:
            x_input = x_t  # Shape: [batch_size, seq_len, num_classes]

        # Apply convolutional encoder
        x = self.conv_encoder(x_input)  # Shape: [batch_size, seq_len, hidden_size]

        # Timestep and score embeddings
        t_emb = self.timestep_embedder(t)  # Shape: [batch_size, hidden_size]
        # score_emb = self.score_embedder(score)  # Shape: [batch_size, hidden_size]
        # Ensure score has shape [batch_size, 1]
        if score.dim() == 1:
            score = score.unsqueeze(-1)  # Shape: [batch_size, 1]
        elif score.shape[1] != 1:
            score = score[:, :1]  # Ensures the second dimension is 1 if there are extra dimensions

        # Apply score embedder
        score_emb = self.score_embedder(score)  # This should now have the correct input shape

        # Score embedding
        score_emb = self.score_embedder(score)

        # Combine embeddings and expand to sequence length
        seq_len = x.size(1)
        conditioning = t_emb + score_emb  # Shape: [batch_size, hidden_size]
        conditioning = conditioning.unsqueeze(1)  # Shape: [batch_size, seq_len, hidden_size]

        # Add conditioning to the sequence embeddings
        # x = x + conditioning
        x = x + conditioning.expand(-1, x.size(1), -1)

        # Add positional encoding
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc

        # Prepare input for transformer
        x = x.transpose(0, 1)  # Shape: [seq_len, batch_size, hidden_size]
        
        # Adjust the attention mask
        if attention_mask is not None:
            if attention_mask.dim() > 2:
                attention_mask = attention_mask.squeeze(-1)
            # Convert attention_mask to BoolTensor if not already
            attention_mask = attention_mask.bool()
            # key_padding_mask expects True for positions to be masked (padding positions)
            key_padding_mask = ~attention_mask  # Invert mask
        else:
            key_padding_mask = None

        # Apply transformer blocks

        for block in self.blocks:
            x = block(x, attn_mask=key_padding_mask)

        # Output layer
        x = x.transpose(0, 1)  # Shape: [batch_size, seq_len, hidden_size]
        logits = self.output_layer(x)  # Shape: [batch_size, seq_len, num_classes]

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

