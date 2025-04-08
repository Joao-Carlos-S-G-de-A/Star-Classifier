import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, hamming_loss
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
import warnings
import json
import gc  # For garbage collection
import argparse  # For command line arguments
import pickle

# Suppress warnings
warnings.filterwarnings("ignore")

# Set the plotting style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

# Define custom colors for consistency across plots
MODEL_COLORS = {
    'Gated CNN (MAMBAOut)': '#4C72B0',  # Blue
    'MAMBA': '#55A868',  # Green
    'Transformer': '#C44E52',  # Red
}

# Define token sizes for consistent ordering
TOKEN_SIZES = ['1 Token', 'Balanced', 'Max Tokens']

# Define output directories
OUTPUT_DIR = "evaluation_results"
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# Create necessary directories
for directory in [OUTPUT_DIR, METRICS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# ===== MODEL ARCHITECTURE DEFINITIONS =====

# Import what we need for DropPath in MambaOut models
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with different dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class FeatureTokenizer(nn.Module):
    """
    Tokenizes input features into sequences for transformer/mamba processing.
    Converts a flat feature vector into a sequence of token embeddings.
    """
    def __init__(self, input_dim, token_dim, d_model):
        """
        Args:
            input_dim (int): Dimension of input features
            token_dim (int): Dimension of each token (features are chunked into tokens)
            d_model (int): Output embedding dimension for each token
        """
        super().__init__()
        
        # Calculate number of tokens based on input_dim and token_dim
        # If input_dim is not divisible by token_dim, we'll pad
        self.n_tokens = (input_dim + token_dim - 1) // token_dim
        self.token_dim = token_dim
        self.input_dim = input_dim
        
        # Projection layer from token_dim to d_model
        self.projection = nn.Linear(token_dim, d_model)
        
        # Positional embedding for tokens
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_tokens, d_model))
        
        # Initialize
        nn.init.normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            token_embeddings: [batch_size, n_tokens, d_model]
        """
        batch_size = x.shape[0]
        
        # If necessary, pad the input to be divisible by token_dim
        if self.input_dim % self.token_dim != 0:
            padding_size = self.n_tokens * self.token_dim - self.input_dim
            x = F.pad(x, (0, padding_size), "constant", 0)
        
        # Reshape into tokens: [batch_size, n_tokens, token_dim]
        x = x.reshape(batch_size, self.n_tokens, self.token_dim)
        
        # Project each token to d_model dimension
        token_embeddings = self.projection(x)  # [batch_size, n_tokens, d_model]
        
        # Add positional embeddings
        token_embeddings = token_embeddings + self.pos_embedding
        
        return token_embeddings

class GatedCNNBlock(nn.Module):
    """Simplified and fixed GatedCNNBlock that preserves sequence length"""
    def __init__(self, dim, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(expand * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        
        # Properly calculate padding to ensure output length matches input length
        padding = (d_conv - 1) // 2
        
        # Single convolution with proper padding
        self.conv = nn.Conv1d(
            in_channels=hidden,
            out_channels=hidden, 
            kernel_size=d_conv,
            padding=padding,
            groups=hidden
        )
        
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Input shape: [B, seq_len, dim]
        shortcut = x
        x = self.norm(x)
        
        # Split for gating
        x = self.fc1(x)
        g, c = torch.chunk(x, 2, dim=-1)
        
        # Check shapes before processing
        batch_size, seq_len, channels = c.shape
        
        # Apply convolution
        c_permuted = c.permute(0, 2, 1)  # [B, hidden, seq_len]
        c_conv = self.conv(c_permuted)
        
        # Ensure output sequence length matches input
        if c_conv.size(2) != seq_len:
            if c_conv.size(2) < seq_len:
                # Pad if shorter
                padding = torch.zeros(
                    batch_size, channels, seq_len - c_conv.size(2),
                    device=c_conv.device, dtype=c_conv.dtype
                )
                c_conv = torch.cat([c_conv, padding], dim=2)
            else:
                # Truncate if longer
                c_conv = c_conv[:, :, :seq_len]
        
        c_final = c_conv.permute(0, 2, 1)  # [B, seq_len, hidden]
        
        # Perform gating and output projection
        x = self.fc2(self.act(g) * c_final)
        x = self.drop_path(x)
        
        return x + shortcut

class SequenceMambaOut(nn.Module):
    """Adaptation of MambaOut for sequence data with a single stage"""
    def __init__(self, d_model, d_conv=4, expand=2, depth=1, drop_path=0.):
        super().__init__()
        
        # Create a sequence of GatedCNNBlocks
        self.blocks = nn.Sequential(
            *[GatedCNNBlock(
                dim=d_model,
                d_conv=d_conv,
                expand=expand,
                drop_path=drop_path
            ) for _ in range(depth)]
        )
    
    def forward(self, x):
        return self.blocks(x)

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block to attend from one modality to another.
    """
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True
        )
    
    def forward(self, x, context):
        """
        Args:
            x: Query tensor of shape [batch_size, seq_len_q, dim]
            context: Key/value tensor of shape [batch_size, seq_len_kv, dim]
        
        Returns:
            Output tensor of shape [batch_size, seq_len_q, dim]
        """
        x_norm = self.norm(x)
        attn_output, _ = self.attention(
            query=x_norm,
            key=context,
            value=context
        )
        return x + attn_output

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
        # Generate position embeddings once at initialization
        self._generate_embeddings()
        
    def _generate_embeddings(self):
        t = torch.arange(self.max_seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(self.max_seq_len, 1, -1)
        sin = emb.sin().view(self.max_seq_len, 1, -1)
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)
        
    def forward(self, seq_len):
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to q and k tensors."""
    # Handle the case where q and k have shape [batch_size, seq_len, head_dim]
    # or [batch_size, n_heads, seq_len, head_dim]
    if q.dim() == 3:
        # [batch_size, seq_len, head_dim] -> [batch_size, seq_len, 1, head_dim]
        q = q.unsqueeze(2)
        k = k.unsqueeze(2)
        # After this operation, we squeeze back
        squeeze_after = True
    else:
        squeeze_after = False
    
    # Reshape cos and sin for proper broadcasting
    # [seq_len, 1, head_dim] -> [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    
    # Apply rotation
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    if squeeze_after:
        q_rot = q_rot.squeeze(2)
        k_rot = k_rot.squeeze(2)
    
    return q_rot, k_rot

class RotarySelfAttention(nn.Module):
    """Self-attention with rotary position embeddings."""
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        
        # QKV projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Rotary positional embedding
        self.rope = RotaryEmbedding(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            output: Tensor of same shape as input
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        q = self.q_proj(x)  # [batch_size, seq_len, dim]
        k = self.k_proj(x)  # [batch_size, seq_len, dim]
        v = self.v_proj(x)  # [batch_size, seq_len, dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Get position embeddings
        cos, sin = self.rope(seq_len)
        
        # Apply rotary position embeddings to q and k
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Transpose for efficient batch matrix multiplication
        q = q.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        
        # Compute scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, n_heads, seq_len, seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)  # [batch_size, n_heads, seq_len, head_dim]
        
        # Reshape back to original format
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Apply output projection
        output = self.out_proj(output)
        
        return output

class TransformerBlock(nn.Module):
    """Transformer block with rotary self-attention and feed-forward network."""
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RotarySelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # FFN with residual connection
        x = x + self.ffn(self.norm2(x))
        
        return x

class TransformerFeatureExtractor(nn.Module):
    """Stack of transformer blocks for feature extraction."""
    def __init__(self, d_model, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Processed tensor of same shape
        """
        for layer in self.layers:
            x = layer(x)
        return x

# Define model classes
class StarClassifierFusionMambaOut(nn.Module):
    def __init__(
        self,
        d_model_spectra,
        d_model_gaia,
        num_classes,
        input_dim_spectra,
        input_dim_gaia,
        token_dim_spectra=64,  # New parameter for token size
        token_dim_gaia=2,      # New parameter for token size
        n_layers=6,
        use_cross_attention=True,
        n_cross_attn_heads=8,
        d_conv=4,
        expand=2,
    ):
        """
        Args:
            d_model_spectra (int): embedding dimension for the spectra MAMBA
            d_model_gaia (int): embedding dimension for the gaia MAMBA
            num_classes (int): multi-label classification
            input_dim_spectra (int): # of features for spectra
            input_dim_gaia (int): # of features for gaia
            token_dim_spectra (int): size of each token for spectra features
            token_dim_gaia (int): size of each token for gaia features
            n_layers (int): depth for each MAMBA
            use_cross_attention (bool): whether to use cross-attention
            n_cross_attn_heads (int): number of heads for cross-attention
        """
        super().__init__()

        # --- Feature Tokenizers ---
        self.tokenizer_spectra = FeatureTokenizer(
            input_dim=input_dim_spectra,
            token_dim=token_dim_spectra,
            d_model=d_model_spectra
        )
        
        self.tokenizer_gaia = FeatureTokenizer(
            input_dim=input_dim_gaia,
            token_dim=token_dim_gaia,
            d_model=d_model_gaia
        )

        # --- MambaOut for spectra ---
        self.mamba_spectra = nn.Sequential(
            *[SequenceMambaOut(
                d_model=d_model_spectra,
                d_conv=d_conv,
                expand=expand,
                depth=1,
                drop_path=0.1 if i > 0 else 0.0,
            ) for i in range(n_layers)]
        )

        # --- MambaOut for gaia ---
        self.mamba_gaia = nn.Sequential(
            *[SequenceMambaOut(
                d_model=d_model_gaia,
                d_conv=d_conv,
                expand=expand,
                depth=1,
                drop_path=0.1 if i > 0 else 0.0,
            ) for i in range(n_layers)]
        )

        # --- Cross Attention (Optional) ---
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn_block_spectra = CrossAttentionBlock(d_model_spectra, n_heads=n_cross_attn_heads)
            self.cross_attn_block_gaia = CrossAttentionBlock(d_model_gaia, n_heads=n_cross_attn_heads)

        # --- Final Classifier ---
        fusion_dim = d_model_spectra + d_model_gaia
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, num_classes)
        )
    
    def forward(self, x_spectra, x_gaia):
        """
        x_spectra : (batch_size, input_dim_spectra)
        x_gaia    : (batch_size, input_dim_gaia)
        """
        # Tokenize input features
        # From [batch_size, input_dim] to [batch_size, num_tokens, d_model]
        x_spectra = self.tokenizer_spectra(x_spectra)  # (B, num_tokens_spectra, d_model_spectra)
        x_gaia = self.tokenizer_gaia(x_gaia)           # (B, num_tokens_gaia, d_model_gaia)

        # --- MambaOut encoding (each modality separately) ---
        x_spectra = self.mamba_spectra(x_spectra)  # (B, num_tokens_spectra, d_model_spectra)
        x_gaia = self.mamba_gaia(x_gaia)           # (B, num_tokens_gaia, d_model_gaia)

        # Optionally, use cross-attention to fuse the representations
        if self.use_cross_attention:
            # Cross-attention from spectra -> gaia
            x_spectra_fused = self.cross_attn_block_spectra(x_spectra, x_gaia)
            # Cross-attention from gaia -> spectra
            x_gaia_fused = self.cross_attn_block_gaia(x_gaia, x_spectra)
            
            # Update x_spectra and x_gaia
            x_spectra = x_spectra_fused
            x_gaia = x_gaia_fused
        
        # --- Pool across sequence dimension ---
        x_spectra = x_spectra.mean(dim=1)  # (B, d_model_spectra)
        x_gaia = x_gaia.mean(dim=1)        # (B, d_model_gaia)

        # --- Late Fusion by Concatenation ---
        x_fused = torch.cat([x_spectra, x_gaia], dim=-1)  # (B, d_model_spectra + d_model_gaia)

        # --- Final classification ---
        logits = self.classifier(x_fused)  # (B, num_classes)
        return logits

class StarClassifierFusionTransformer(nn.Module):
    """Transformer-based feature extractor with tokenization for multi-modal fusion."""
    def __init__(
        self,
        d_model_spectra,
        d_model_gaia,
        num_classes,
        input_dim_spectra,
        input_dim_gaia,
        token_dim_spectra=64,  # Size of each token for spectra
        token_dim_gaia=2,      # Size of each token for gaia
        n_layers=6,
        n_heads=8,
        use_cross_attention=True,
        n_cross_attn_heads=8,
        dropout=0.1,
    ):
        """
        Args:
            d_model_spectra (int): embedding dimension for the spectra Transformer
            d_model_gaia (int): embedding dimension for the gaia Transformer
            num_classes (int): multi-label classification
            input_dim_spectra (int): # of features for spectra
            input_dim_gaia (int): # of features for gaia
            token_dim_spectra (int): size of each token for spectra features
            token_dim_gaia (int): size of each token for gaia features
            n_layers (int): depth for each Transformer
            n_heads (int): number of attention heads
            use_cross_attention (bool): whether to use cross-attention
            n_cross_attn_heads (int): number of heads for cross-attention
            dropout (float): dropout rate
        """
        super().__init__()

        # --- Feature Tokenizers ---
        self.tokenizer_spectra = FeatureTokenizer(
            input_dim=input_dim_spectra,
            token_dim=token_dim_spectra,
            d_model=d_model_spectra
        )
        
        self.tokenizer_gaia = FeatureTokenizer(
            input_dim=input_dim_gaia,
            token_dim=token_dim_gaia,
            d_model=d_model_gaia
        )

        # --- Transformer for spectra ---
        self.transformer_spectra = TransformerFeatureExtractor(
            d_model=d_model_spectra,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # --- Transformer for gaia ---
        self.transformer_gaia = TransformerFeatureExtractor(
            d_model=d_model_gaia,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # --- Cross Attention (Optional) ---
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn_block_spectra = CrossAttentionBlock(d_model_spectra, n_heads=n_cross_attn_heads)
            self.cross_attn_block_gaia = CrossAttentionBlock(d_model_gaia, n_heads=n_cross_attn_heads)

        # --- Final Classifier ---
        fusion_dim = d_model_spectra + d_model_gaia
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)
        )
    
    def forward(self, x_spectra, x_gaia):
        """
        Args:
            x_spectra: Spectra features of shape [batch_size, input_dim_spectra]
            x_gaia: Gaia features of shape [batch_size, input_dim_gaia]
            
        Returns:
            logits: Classification logits of shape [batch_size, num_classes]
        """
        # Tokenize input features
        # From [batch_size, input_dim] to [batch_size, num_tokens, d_model]
        x_spectra_tokens = self.tokenizer_spectra(x_spectra)
        x_gaia_tokens = self.tokenizer_gaia(x_gaia)
        
        # Process through transformers
        x_spectra = self.transformer_spectra(x_spectra_tokens)  # [batch_size, num_tokens_spectra, d_model]
        x_gaia = self.transformer_gaia(x_gaia_tokens)          # [batch_size, num_tokens_gaia, d_model]

        # Optional cross-attention
        if self.use_cross_attention:
            x_spectra = self.cross_attn_block_spectra(x_spectra, x_gaia)
            x_gaia = self.cross_attn_block_gaia(x_gaia, x_spectra)
        
        # Global pooling over sequence dimension
        x_spectra = x_spectra.mean(dim=1)  # [batch_size, d_model]
        x_gaia = x_gaia.mean(dim=1)        # [batch_size, d_model]

        # Concatenate for fusion
        x_fused = torch.cat([x_spectra, x_gaia], dim=-1)  # [batch_size, 2*d_model]

        # Final classification
        logits = self.classifier(x_fused)  # [batch_size, num_classes]
        
        return logits

# Note: The StarClassifierFusionMambaTokenized requires the Mamba2 implementation which might not be available
# We'll handle that case specially in the model loading function

# ===== CUSTOM DATASET CLASS =====

class MultiModalBalancedMultiLabelDataset(Dataset):
    """
    A balanced multi-label dataset that returns (X_spectra, X_gaia, y).
    It uses the same balancing strategy as `BalancedMultiLabelDataset`.
    """
    def __init__(self, X_spectra, X_gaia, y, limit_per_label=201):
        """
        Args:
            X_spectra (torch.Tensor): [num_samples, num_spectra_features]
            X_gaia (torch.Tensor): [num_samples, num_gaia_features]
            y (torch.Tensor): [num_samples, num_classes], multi-hot labels
            limit_per_label (int): limit or target number of samples per label
        """
        self.X_spectra = X_spectra
        self.X_gaia = X_gaia
        self.y = y
        self.limit_per_label = limit_per_label
        self.num_classes = y.shape[1]
        self.indices = self.balance_classes()

    def balance_classes(self):
        indices = []
        class_counts = torch.sum(self.y, axis=0)
        for cls in range(self.num_classes):
            cls_indices = np.where(self.y[:, cls] == 1)[0]
            if len(cls_indices) < self.limit_per_label:
                if len(cls_indices) == 0:
                    # No samples for this class
                    continue
                extra_indices = np.random.choice(
                    cls_indices, self.limit_per_label - len(cls_indices), replace=True
                )
                cls_indices = np.concatenate([cls_indices, extra_indices])
            elif len(cls_indices) > self.limit_per_label:
                cls_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
            indices.extend(cls_indices)
        indices = np.unique(indices)
        np.random.shuffle(indices)
        return indices

    def re_sample(self):
        self.indices = self.balance_classes()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        return (
            self.X_spectra[index],  # spectra features
            self.X_gaia[index],  # gaia features
            self.y[index],  # multi-hot labels
        )

def calculate_metrics(y_true, y_pred):
    metrics = {
        "micro_f1": f1_score(y_true, y_pred, average='micro'),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
        "micro_precision": precision_score(y_true, y_pred, average='micro', zero_division=1),
        "macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=1),
        "weighted_precision": precision_score(y_true, y_pred, average='weighted', zero_division=1),
        "micro_recall": recall_score(y_true, y_pred, average='micro'),
        "macro_recall": recall_score(y_true, y_pred, average='macro'),
        "weighted_recall": recall_score(y_true, y_pred, average='weighted'),
        "hamming_loss": hamming_loss(y_true, y_pred)
    }
    # Calculate accuracy as well (proportion of correctly predicted samples)
    metrics["accuracy"] = accuracy_score(y_true.flatten(), y_pred.flatten())
    
    return metrics

# ===== MODEL LOADING AND EVALUATION FUNCTIONS =====

def load_model(model_path, input_dim_spectra=3647, input_dim_gaia=18, num_classes=55):
    """
    Load a saved PyTorch model from a state dictionary.
    
    Args:
        model_path (str): Path to the saved model state dict file
        input_dim_spectra (int): Input dimension for spectra
        input_dim_gaia (int): Input dimension for gaia
        num_classes (int): Number of classes
        
    Returns:
        model: Loaded PyTorch model
    """
    try:
        # Determine model type from filename
        model_type = None
        if 'mamba_' in model_path.lower() or 'mamba.' in model_path.lower():
            if 'mambaout' not in model_path.lower():
                model_type = 'MAMBA'
            else:
                model_type = 'MambaOut'
        elif 'transformer' in model_path.lower():
            model_type = 'Transformer'
        elif 'gated_cnn' in model_path.lower() or 'mambaout' in model_path.lower():
            model_type = 'MambaOut'
        else:
            print(f"Unable to determine model type from filename: {model_path}")
            return None
        
        # Initialize parameters based on the model configurations in the original code
        n_heads = 8
        expand = 2
        
        # Determine token config from filename and set correct parameters
        if '1_token' in model_path.lower():
            token_config = '1 Token'
            
            if model_type == 'MambaOut':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 3647  # Exact value instead of input_dim_spectra
                token_dim_gaia = 18       # Exact value instead of input_dim_gaia
                n_layers = 20
                d_conv = 1  # MambaOut 1 Token has d_conv = 1
            
            elif model_type == 'MAMBA':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 3647  # Exact value
                token_dim_gaia = 18       # Exact value
                n_layers = 20
                d_state = 32
                d_conv = 2
            
            elif model_type == 'Transformer':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 3647  # Exact value
                token_dim_gaia = 18       # Exact value
                n_layers = 10
                
        elif 'balanced' in model_path.lower():
            token_config = 'Balanced'
            
            if model_type == 'MambaOut':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 192
                token_dim_gaia = 1
                n_layers = 20
                d_conv = 4
            
            elif model_type == 'MAMBA':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 192
                token_dim_gaia = 1
                n_layers = 20
                d_state = 32
                d_conv = 4
            
            elif model_type == 'Transformer':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 192
                token_dim_gaia = 1
                n_layers = 10
            
        else:  # max_tokens
            token_config = 'Max Tokens'
            
            if model_type == 'MambaOut':
                d_model_spectra = 1536
                d_model_gaia = 1536
                token_dim_spectra = 7
                token_dim_gaia = 1
                n_layers = 20
                d_conv = 4
            
            elif model_type == 'MAMBA':
                d_model_spectra = 1536
                d_model_gaia = 1536
                token_dim_spectra = 7
                token_dim_gaia = 1
                n_layers = 20
                d_state = 16
                d_conv = 4
            
            elif model_type == 'Transformer':
                d_model_spectra = 1536
                d_model_gaia = 1536
                token_dim_spectra = 7
                token_dim_gaia = 1
                n_layers = 10
        
        print(f"Creating model of type {model_type} with configuration: {token_config}")
        print(f"  d_model_spectra: {d_model_spectra}, d_model_gaia: {d_model_gaia}")
        print(f"  token_dim_spectra: {token_dim_spectra}, token_dim_gaia: {token_dim_gaia}")
        print(f"  n_layers: {n_layers}, n_heads: {n_heads}")
        if model_type == 'MAMBA':
            print(f"  d_state: {d_state}, d_conv: {d_conv}, expand: {expand}")
        elif model_type == 'MambaOut':
            print(f"  d_conv: {d_conv}, expand: {expand}")
        
        # Create model based on type
        if model_type == 'MAMBA':
            # Note: This might fail if the mamba_ssm package is not available
            try:
                from mamba_ssm import Mamba2
                
                # Define custom Mamba model class here since it requires the imported Mamba2
                class StarClassifierFusionMambaTokenized(nn.Module):
                    def __init__(
                        self,
                        d_model_spectra,
                        d_model_gaia,
                        num_classes,
                        input_dim_spectra,
                        input_dim_gaia,
                        token_dim_spectra=64,  # Size of each token for spectra
                        token_dim_gaia=2,      # Size of each token for gaia
                        n_layers=10,
                        use_cross_attention=True,
                        n_cross_attn_heads=8,
                        d_state=256,
                        d_conv=4,
                        expand=2,
                    ):
                        super().__init__()

                        # --- Feature Tokenizers ---
                        self.tokenizer_spectra = FeatureTokenizer(
                            input_dim=input_dim_spectra,
                            token_dim=token_dim_spectra,
                            d_model=d_model_spectra
                        )
                        
                        self.tokenizer_gaia = FeatureTokenizer(
                            input_dim=input_dim_gaia,
                            token_dim=token_dim_gaia,
                            d_model=d_model_gaia
                        )

                        # --- MAMBA 2 for spectra ---
                        self.mamba_spectra = nn.Sequential(
                            *[Mamba2(
                                d_model=d_model_spectra,
                                d_state=d_state,
                                d_conv=d_conv,
                                expand=expand,
                            ) for _ in range(n_layers)]
                        )

                        # --- MAMBA 2 for gaia ---
                        self.mamba_gaia = nn.Sequential(
                            *[Mamba2(
                                d_model=d_model_gaia,
                                d_state=d_state,
                                d_conv=d_conv,
                                expand=expand,
                            ) for _ in range(n_layers)]
                        )

                        # --- Cross Attention (Optional) ---
                        self.use_cross_attention = use_cross_attention
                        if use_cross_attention:
                            self.cross_attn_block_spectra = CrossAttentionBlock(d_model_spectra, n_heads=n_cross_attn_heads)
                            self.cross_attn_block_gaia = CrossAttentionBlock(d_model_gaia, n_heads=n_cross_attn_heads)

                        # --- Final Classifier ---
                        fusion_dim = d_model_spectra + d_model_gaia
                        self.classifier = nn.Sequential(
                            nn.LayerNorm(fusion_dim),
                            nn.Linear(fusion_dim, num_classes)
                        )
                    
                    def forward(self, x_spectra, x_gaia):
                        # Tokenize input features
                        x_spectra_tokens = self.tokenizer_spectra(x_spectra)
                        x_gaia_tokens = self.tokenizer_gaia(x_gaia)
                        
                        # Process through Mamba models
                        x_spectra = self.mamba_spectra(x_spectra_tokens)
                        x_gaia = self.mamba_gaia(x_gaia_tokens)          

                        # Optional cross-attention
                        if self.use_cross_attention:
                            x_spectra = self.cross_attn_block_spectra(x_spectra, x_gaia)
                            x_gaia = self.cross_attn_block_gaia(x_gaia, x_spectra)
                        
                        # Global pooling over sequence dimension
                        x_spectra = x_spectra.mean(dim=1)  # [batch_size, d_model]
                        x_gaia = x_gaia.mean(dim=1)        # [batch_size, d_model]

                        # Concatenate for fusion
                        x_fused = torch.cat([x_spectra, x_gaia], dim=-1)  # [batch_size, 2*d_model]

                        # Final classification
                        logits = self.classifier(x_fused)  # [batch_size, num_classes]
                        
                        return logits
                
                model = StarClassifierFusionMambaTokenized(
                    d_model_spectra=d_model_spectra,
                    d_model_gaia=d_model_gaia,
                    num_classes=num_classes,
                    input_dim_spectra=input_dim_spectra,
                    input_dim_gaia=input_dim_gaia,
                    token_dim_spectra=token_dim_spectra,
                    token_dim_gaia=token_dim_gaia,
                    n_layers=n_layers,
                    n_cross_attn_heads=n_heads,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
            except ImportError:
                print("Mamba2 module not found. Using MambaOut model as a fallback.")
                model_type = 'MambaOut'  # Fallback to MambaOut
        
        if model_type == 'Transformer':
            model = StarClassifierFusionTransformer(
                d_model_spectra=d_model_spectra,
                d_model_gaia=d_model_gaia,
                num_classes=num_classes,
                input_dim_spectra=input_dim_spectra,
                input_dim_gaia=input_dim_gaia,
                token_dim_spectra=token_dim_spectra,
                token_dim_gaia=token_dim_gaia,
                n_layers=n_layers,
                n_heads=n_heads
            )
        elif model_type == 'MambaOut':
            model = StarClassifierFusionMambaOut(
                d_model_spectra=d_model_spectra,
                d_model_gaia=d_model_gaia,
                num_classes=num_classes,
                input_dim_spectra=input_dim_spectra,
                input_dim_gaia=input_dim_gaia,
                token_dim_spectra=token_dim_spectra,
                token_dim_gaia=token_dim_gaia,
                n_layers=n_layers,
                d_conv=d_conv,
                expand=expand
            )
        
        # Load state dictionary
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if the state_dict is wrapped (common in distributed training)
        # If keys start with 'module.', remove that prefix
        if all(k.startswith('module.') for k in state_dict.keys()):
            print("Removing 'module.' prefix from state dict keys")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Now match the keys to load the state dict
        model.load_state_dict(state_dict)
        
        # Set model to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
    

def load_model(model_path, input_dim_spectra=3647, input_dim_gaia=18, num_classes=55):
    """
    Load a saved PyTorch model from a state dictionary.
    
    Args:
        model_path (str): Path to the saved model state dict file
        input_dim_spectra (int): Input dimension for spectra
        input_dim_gaia (int): Input dimension for gaia
        num_classes (int): Number of classes
        
    Returns:
        model: Loaded PyTorch model
    """
    try:
        # Determine model type from filename
        model_type = None
        if 'mamba_' in model_path.lower() or 'mamba.' in model_path.lower():
            if 'mambaout' not in model_path.lower():
                model_type = 'MAMBA'
            else:
                model_type = 'MambaOut'
        elif 'transformer' in model_path.lower():
            model_type = 'Transformer'
        elif 'gated_cnn' in model_path.lower() or 'mambaout' in model_path.lower():
            model_type = 'MambaOut'
        else:
            print(f"Unable to determine model type from filename: {model_path}")
            return None
        
        # Initialize parameters based on the model configurations in the original code
        n_heads = 8
        expand = 2
        
        # Determine token config from filename and set correct parameters
        if '1_token' in model_path.lower():
            token_config = '1 Token'
            
            if model_type == 'MambaOut':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 3647  # Exact value instead of input_dim_spectra
                token_dim_gaia = 18       # Exact value instead of input_dim_gaia
                n_layers = 20
                d_conv = 1  # MambaOut 1 Token has d_conv = 1
            
            elif model_type == 'MAMBA':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 3647  # Exact value
                token_dim_gaia = 18       # Exact value
                n_layers = 20
                d_state = 32
                d_conv = 2
            
            elif model_type == 'Transformer':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 3647  # Exact value
                token_dim_gaia = 18       # Exact value
                n_layers = 10
                
        elif 'balanced' in model_path.lower():
            token_config = 'Balanced'
            
            if model_type == 'MambaOut':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 192
                token_dim_gaia = 1
                n_layers = 20
                d_conv = 4
            
            elif model_type == 'MAMBA':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 192
                token_dim_gaia = 1
                n_layers = 20
                d_state = 32
                d_conv = 4
            
            elif model_type == 'Transformer':
                d_model_spectra = 2048
                d_model_gaia = 2048
                token_dim_spectra = 192
                token_dim_gaia = 1
                n_layers = 10
            
        else:  # max_tokens
            token_config = 'Max Tokens'
            
            if model_type == 'MambaOut':
                d_model_spectra = 1536
                d_model_gaia = 1536
                token_dim_spectra = 7
                token_dim_gaia = 1
                n_layers = 20
                d_conv = 4
            
            elif model_type == 'MAMBA':
                d_model_spectra = 1536
                d_model_gaia = 1536
                token_dim_spectra = 7
                token_dim_gaia = 1
                n_layers = 20
                d_state = 16
                d_conv = 4
            
            elif model_type == 'Transformer':
                d_model_spectra = 1536
                d_model_gaia = 1536
                token_dim_spectra = 7
                token_dim_gaia = 1
                n_layers = 10
        
        print(f"Creating model of type {model_type} with configuration: {token_config}")
        print(f"  d_model_spectra: {d_model_spectra}, d_model_gaia: {d_model_gaia}")
        print(f"  token_dim_spectra: {token_dim_spectra}, token_dim_gaia: {token_dim_gaia}")
        print(f"  n_layers: {n_layers}, n_heads: {n_heads}")
        if model_type == 'MAMBA':
            print(f"  d_state: {d_state}, d_conv: {d_conv}, expand: {expand}")
        elif model_type == 'MambaOut':
            print(f"  d_conv: {d_conv}, expand: {expand}")
        
        # Define a custom FeatureTokenizer compatible with saved models
        # This version uses token_embed instead of projection and doesn't use pos_embedding
        class CompatibleFeatureTokenizer(nn.Module):
            def __init__(self, input_dim, token_dim, d_model):
                super().__init__()
                self.n_tokens = (input_dim + token_dim - 1) // token_dim
                self.token_dim = token_dim
                self.input_dim = input_dim
                self.token_embed = nn.Linear(token_dim, d_model)
                
            def forward(self, x):
                batch_size = x.shape[0]
                if self.input_dim % self.token_dim != 0:
                    padding_size = self.n_tokens * self.token_dim - self.input_dim
                    x = F.pad(x, (0, padding_size), "constant", 0)
                x = x.reshape(batch_size, self.n_tokens, self.token_dim)
                return self.token_embed(x)
        
        # Create model based on type
        if model_type == 'MAMBA':
            # Note: This might fail if the mamba_ssm package is not available
            try:
                from mamba_ssm import Mamba2
                
                # Define custom Mamba model class with compatible tokenizer
                class StarClassifierFusionMambaTokenized(nn.Module):
                    def __init__(
                        self,
                        d_model_spectra,
                        d_model_gaia,
                        num_classes,
                        input_dim_spectra,
                        input_dim_gaia,
                        token_dim_spectra=64,
                        token_dim_gaia=2,
                        n_layers=10,
                        use_cross_attention=True,
                        n_cross_attn_heads=8,
                        d_state=256,
                        d_conv=4,
                        expand=2,
                    ):
                        super().__init__()

                        # Use compatible tokenizers
                        self.tokenizer_spectra = CompatibleFeatureTokenizer(
                            input_dim=input_dim_spectra,
                            token_dim=token_dim_spectra,
                            d_model=d_model_spectra
                        )
                        
                        self.tokenizer_gaia = CompatibleFeatureTokenizer(
                            input_dim=input_dim_gaia,
                            token_dim=token_dim_gaia,
                            d_model=d_model_gaia
                        )

                        # Rest of the implementation remains the same
                        self.mamba_spectra = nn.Sequential(
                            *[Mamba2(
                                d_model=d_model_spectra,
                                d_state=d_state,
                                d_conv=d_conv,
                                expand=expand,
                            ) for _ in range(n_layers)]
                        )

                        self.mamba_gaia = nn.Sequential(
                            *[Mamba2(
                                d_model=d_model_gaia,
                                d_state=d_state,
                                d_conv=d_conv,
                                expand=expand,
                            ) for _ in range(n_layers)]
                        )

                        self.use_cross_attention = use_cross_attention
                        if use_cross_attention:
                            self.cross_attn_block_spectra = CrossAttentionBlock(d_model_spectra, n_heads=n_cross_attn_heads)
                            self.cross_attn_block_gaia = CrossAttentionBlock(d_model_gaia, n_heads=n_cross_attn_heads)

                        fusion_dim = d_model_spectra + d_model_gaia
                        self.classifier = nn.Sequential(
                            nn.LayerNorm(fusion_dim),
                            nn.Linear(fusion_dim, num_classes)
                        )
                    
                    def forward(self, x_spectra, x_gaia):
                        x_spectra_tokens = self.tokenizer_spectra(x_spectra)
                        x_gaia_tokens = self.tokenizer_gaia(x_gaia)
                        
                        x_spectra = self.mamba_spectra(x_spectra_tokens)
                        x_gaia = self.mamba_gaia(x_gaia_tokens)          

                        if self.use_cross_attention:
                            x_spectra = self.cross_attn_block_spectra(x_spectra, x_gaia)
                            x_gaia = self.cross_attn_block_gaia(x_gaia, x_spectra)
                        
                        x_spectra = x_spectra.mean(dim=1)
                        x_gaia = x_gaia.mean(dim=1)

                        x_fused = torch.cat([x_spectra, x_gaia], dim=-1)
                        logits = self.classifier(x_fused)
                        
                        return logits
                
                model = StarClassifierFusionMambaTokenized(
                    d_model_spectra=d_model_spectra,
                    d_model_gaia=d_model_gaia,
                    num_classes=num_classes,
                    input_dim_spectra=input_dim_spectra,
                    input_dim_gaia=input_dim_gaia,
                    token_dim_spectra=token_dim_spectra,
                    token_dim_gaia=token_dim_gaia,
                    n_layers=n_layers,
                    n_cross_attn_heads=n_heads,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
            except ImportError:
                print("Mamba2 module not found. Using MambaOut model as a fallback.")
                model_type = 'MambaOut'
        
        if model_type == 'Transformer':
            # Modified Transformer with compatible tokenizer
            class CompatibleTransformer(nn.Module):
                def __init__(
                    self,
                    d_model_spectra,
                    d_model_gaia,
                    num_classes,
                    input_dim_spectra,
                    input_dim_gaia,
                    token_dim_spectra=64,
                    token_dim_gaia=2,
                    n_layers=10,
                    n_heads=8,
                    use_cross_attention=True,
                    n_cross_attn_heads=8,
                    dropout=0.1,
                ):
                    super().__init__()

                    # Use compatible tokenizers
                    self.tokenizer_spectra = CompatibleFeatureTokenizer(
                        input_dim=input_dim_spectra,
                        token_dim=token_dim_spectra,
                        d_model=d_model_spectra
                    )
                    
                    self.tokenizer_gaia = CompatibleFeatureTokenizer(
                        input_dim=input_dim_gaia,
                        token_dim=token_dim_gaia,
                        d_model=d_model_gaia
                    )

                    # Rest of the implementation same as StarClassifierFusionTransformer
                    self.transformer_spectra = TransformerFeatureExtractor(
                        d_model=d_model_spectra,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        dropout=dropout
                    )

                    self.transformer_gaia = TransformerFeatureExtractor(
                        d_model=d_model_gaia,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        dropout=dropout
                    )

                    self.use_cross_attention = use_cross_attention
                    if use_cross_attention:
                        self.cross_attn_block_spectra = CrossAttentionBlock(d_model_spectra, n_heads=n_cross_attn_heads)
                        self.cross_attn_block_gaia = CrossAttentionBlock(d_model_gaia, n_heads=n_cross_attn_heads)

                    fusion_dim = d_model_spectra + d_model_gaia
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(fusion_dim),
                        nn.Dropout(dropout),
                        nn.Linear(fusion_dim, num_classes)
                    )
                
                def forward(self, x_spectra, x_gaia):
                    x_spectra_tokens = self.tokenizer_spectra(x_spectra)
                    x_gaia_tokens = self.tokenizer_gaia(x_gaia)
                    
                    x_spectra = self.transformer_spectra(x_spectra_tokens)
                    x_gaia = self.transformer_gaia(x_gaia_tokens)

                    if self.use_cross_attention:
                        x_spectra = self.cross_attn_block_spectra(x_spectra, x_gaia)
                        x_gaia = self.cross_attn_block_gaia(x_gaia, x_spectra)
                    
                    x_spectra = x_spectra.mean(dim=1)
                    x_gaia = x_gaia.mean(dim=1)

                    x_fused = torch.cat([x_spectra, x_gaia], dim=-1)
                    logits = self.classifier(x_fused)
                    
                    return logits
            
            model = CompatibleTransformer(
                d_model_spectra=d_model_spectra,
                d_model_gaia=d_model_gaia,
                num_classes=num_classes,
                input_dim_spectra=input_dim_spectra,
                input_dim_gaia=input_dim_gaia,
                token_dim_spectra=token_dim_spectra,
                token_dim_gaia=token_dim_gaia,
                n_layers=n_layers,
                n_heads=n_heads
            )
            
        elif model_type == 'MambaOut':
            # Modified MambaOut with compatible tokenizer
            class CompatibleMambaOut(nn.Module):
                def __init__(
                    self,
                    d_model_spectra,
                    d_model_gaia,
                    num_classes,
                    input_dim_spectra,
                    input_dim_gaia,
                    token_dim_spectra=64,
                    token_dim_gaia=2,
                    n_layers=6,
                    use_cross_attention=True,
                    n_cross_attn_heads=8,
                    d_conv=4,
                    expand=2,
                ):
                    super().__init__()

                    # Use compatible tokenizers
                    self.tokenizer_spectra = CompatibleFeatureTokenizer(
                        input_dim=input_dim_spectra,
                        token_dim=token_dim_spectra,
                        d_model=d_model_spectra
                    )
                    
                    self.tokenizer_gaia = CompatibleFeatureTokenizer(
                        input_dim=input_dim_gaia,
                        token_dim=token_dim_gaia,
                        d_model=d_model_gaia
                    )

                    # MambaOut for spectra
                    self.mamba_spectra = nn.Sequential(
                        *[SequenceMambaOut(
                            d_model=d_model_spectra,
                            d_conv=d_conv,
                            expand=expand,
                            depth=1,
                            drop_path=0.1 if i > 0 else 0.0,
                        ) for i in range(n_layers)]
                    )

                    # MambaOut for gaia
                    self.mamba_gaia = nn.Sequential(
                        *[SequenceMambaOut(
                            d_model=d_model_gaia,
                            d_conv=d_conv,
                            expand=expand,
                            depth=1,
                            drop_path=0.1 if i > 0 else 0.0,
                        ) for i in range(n_layers)]
                    )

                    # Cross Attention (Optional)
                    self.use_cross_attention = use_cross_attention
                    if use_cross_attention:
                        self.cross_attn_block_spectra = CrossAttentionBlock(d_model_spectra, n_heads=n_cross_attn_heads)
                        self.cross_attn_block_gaia = CrossAttentionBlock(d_model_gaia, n_heads=n_cross_attn_heads)

                    # Final Classifier
                    fusion_dim = d_model_spectra + d_model_gaia
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(fusion_dim),
                        nn.Linear(fusion_dim, num_classes)
                    )
                
                def forward(self, x_spectra, x_gaia):
                    x_spectra = self.tokenizer_spectra(x_spectra)
                    x_gaia = self.tokenizer_gaia(x_gaia)

                    x_spectra = self.mamba_spectra(x_spectra)
                    x_gaia = self.mamba_gaia(x_gaia)

                    if self.use_cross_attention:
                        x_spectra_fused = self.cross_attn_block_spectra(x_spectra, x_gaia)
                        x_gaia_fused = self.cross_attn_block_gaia(x_gaia, x_spectra)
                        
                        x_spectra = x_spectra_fused
                        x_gaia = x_gaia_fused
                    
                    x_spectra = x_spectra.mean(dim=1)
                    x_gaia = x_gaia.mean(dim=1)

                    x_fused = torch.cat([x_spectra, x_gaia], dim=-1)
                    logits = self.classifier(x_fused)
                    
                    return logits
            
            model = CompatibleMambaOut(
                d_model_spectra=d_model_spectra,
                d_model_gaia=d_model_gaia,
                num_classes=num_classes,
                input_dim_spectra=input_dim_spectra,
                input_dim_gaia=input_dim_gaia,
                token_dim_spectra=token_dim_spectra,
                token_dim_gaia=token_dim_gaia,
                n_layers=n_layers,
                d_conv=d_conv,
                expand=expand
            )
        
        # Load state dictionary
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if the state_dict is wrapped
        if all(k.startswith('module.') for k in state_dict.keys()):
            print("Removing 'module.' prefix from state dict keys")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Now load the state dict
        model.load_state_dict(state_dict)
        
        # Set model to evaluation mode
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None
def evaluate_model(model, dataloader, device):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on (CPU or CUDA)
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():  # Disable gradient calculation
        for spectra, gaia, labels in tqdm(dataloader, desc="Evaluating"):
            spectra = spectra.to(device)
            gaia = gaia.to(device)
            
            # Forward pass with both inputs (for multi-modal models)
            try:
                # First try with both inputs (spectra and gaia)
                outputs = model(spectra, gaia)
            except Exception as e:
                # If that fails, check if it's a single-input model
                try:
                    # Try with just spectra 
                    outputs = model(spectra)
                    print("Model accepts only spectra input")
                except Exception:
                    try:
                        # Try with just gaia data
                        outputs = model(gaia)
                        print("Model accepts only gaia input")
                    except Exception:
                        # Try with concatenated input as last resort
                        try:
                            combined = torch.cat((spectra, gaia), dim=1)
                            outputs = model(combined)
                            print("Model accepts concatenated input")
                        except Exception as final_e:
                            raise ValueError(f"Failed to determine model input format: {final_e}")
            
            # Convert outputs to binary predictions based on threshold
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            # Move predictions and labels back to CPU for storage
            all_predictions.append(predictions.cpu().numpy())
            all_true_labels.append(labels.numpy())
    
    # Concatenate batch results
    all_predictions = np.vstack(all_predictions)
    all_true_labels = np.vstack(all_true_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(all_true_labels, all_predictions)
    
    # Calculate per-class metrics
    num_classes = all_true_labels.shape[1]
    
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    sample_sizes = []
    
    for i in range(num_classes):
        per_class_precision.append(precision_score(all_true_labels[:, i], all_predictions[:, i], zero_division=1))
        per_class_recall.append(recall_score(all_true_labels[:, i], all_predictions[:, i]))
        per_class_f1.append(f1_score(all_true_labels[:, i], all_predictions[:, i]))
        sample_sizes.append(np.sum(all_true_labels[:, i]))
    
    metrics['per_class_precision'] = per_class_precision
    metrics['per_class_recall'] = per_class_recall
    metrics['per_class_f1'] = per_class_f1
    metrics['sample_sizes'] = sample_sizes
    
    # Calculate test loss
    test_loss = torch.nn.BCEWithLogitsLoss()(
        torch.tensor(outputs.cpu().numpy(), dtype=torch.float32),
        torch.tensor(all_true_labels, dtype=torch.float32)
    ).item()
    metrics['test_loss'] = test_loss
    
    return metrics, all_true_labels, all_predictions

# ===== DATA LOADING FUNCTIONS =====

def prepare_data(data_path, batch_size=16):
    """
    Prepare data for model evaluation using the custom MultiModalBalancedMultiLabelDataset.
    
    Args:
        data_path (str): Path to the data directory
        batch_size (int): Batch size for evaluation
        
    Returns:
        DataLoader: DataLoader containing the evaluation data
    """
    try:
        # Set batch limit based on batch size
        batch_limit = int(batch_size / 2.5)
        
        # Load class names
        class_names_path = os.path.join(data_path, "Updated_List_of_Classes_ubuntu.pkl")
        alt_class_names_path = os.path.join(data_path, "Updated_list_of_Classes.pkl")
        
        if os.path.exists(class_names_path):
            with open(class_names_path, "rb") as f:
                classes = pickle.load(f)
        elif os.path.exists(alt_class_names_path):
            with open(alt_class_names_path, "rb") as f:
                classes = pickle.load(f)
        else:
            raise FileNotFoundError(f"Class names file not found at {class_names_path} or {alt_class_names_path}")
        
        # Load test data
        test_data_path = os.path.join(data_path, "test_data_transformed_ubuntu.pkl")
        alt_test_data_path = os.path.join(data_path, "test_data_transformed.pkl")
        
        if os.path.exists(test_data_path):
            with open(test_data_path, "rb") as f:
                X_test_full = pickle.load(f)
        elif os.path.exists(alt_test_data_path):
            with open(alt_test_data_path, "rb") as f:
                X_test_full = pickle.load(f)
        else:
            raise FileNotFoundError(f"Test data file not found at {test_data_path} or {alt_test_data_path}")
        
        # Extract labels
        y_test = X_test_full[classes]
        
        # Drop labels from test dataset
        X_test_full.drop(classes, axis=1, inplace=True)
        
        # Extract Gaia and spectral data
        gaia_columns = ["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", 
                        "pmra_error", "pmdec_error", "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", 
                        "phot_bp_mean_flux", "phot_rp_mean_flux", "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", 
                        "flagnoflux"]
        
        # Check if all Gaia columns exist in the data
        existing_gaia_columns = [col for col in gaia_columns if col in X_test_full.columns]
        if len(existing_gaia_columns) < len(gaia_columns):
            print(f"Warning: Some Gaia columns are missing. Using {len(existing_gaia_columns)} out of {len(gaia_columns)} columns.")
            gaia_columns = existing_gaia_columns
        
        # Handle otype and obsid columns if they exist
        columns_to_drop = []
        if "otype" in X_test_full.columns:
            columns_to_drop.append("otype")
        if "obsid" in X_test_full.columns:
            columns_to_drop.append("obsid")
        
        # Spectra data (everything that is not Gaia-related and not otype/obsid)
        X_test_spectra = X_test_full.drop(columns={*columns_to_drop, *gaia_columns}, errors='ignore')
        
        # Gaia data (only the selected columns)
        X_test_gaia = X_test_full[gaia_columns]
        
        # Free up memory
        del X_test_full
        gc.collect()
        
        # Convert to PyTorch tensors
        X_test_spectra_tensor = torch.tensor(X_test_spectra.values, dtype=torch.float32)
        X_test_gaia_tensor = torch.tensor(X_test_gaia.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        
        # Create dataset and dataloader
        test_dataset = MultiModalBalancedMultiLabelDataset(
            X_test_spectra_tensor, X_test_gaia_tensor, y_test_tensor, limit_per_label=batch_limit
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Loaded test dataset with {len(test_dataset)} samples")
        print(f"Spectra features: {X_test_spectra.shape[1]}, Gaia features: {X_test_gaia.shape[1]}, Classes: {len(classes)}")
        
        # Save class names for later visualization
        class_list = list(classes)
        with open(os.path.join(OUTPUT_DIR, "class_names.json"), 'w') as f:
            json.dump(class_list, f)
        
        return test_dataloader, class_list
    
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_class_names(path=None):
    """Load class names from a previously saved JSON file or from the data preparation step"""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "class_names.json")
        
    try:
        with open(path, 'r') as f:
            classes = json.load(f)
        return classes
    except FileNotFoundError:
        print(f"Warning: Class names file not found at {path}. Using generic class names.")
        return [f"Class {i}" for i in range(10)]  # Default to 10 classes

# ===== METRICS SAVING AND LOADING FUNCTIONS =====

def save_metrics(model_name, metrics, true_labels, predicted_labels):
    """
    Save metrics and labels to disk.
    
    Args:
        model_name (str): Name of the model
        metrics (dict): Dictionary of metrics
        true_labels (ndarray): Ground truth labels
        predicted_labels (ndarray): Predicted labels
    """
    # Create a clean filename from the model name
    filename_base = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    # Save metrics as JSON
    metrics_to_save = {k: v for k, v in metrics.items() if isinstance(v, (int, float, list, tuple))}
    
    # Convert numpy arrays in lists to lists
    for key, value in metrics_to_save.items():
        if isinstance(value, list):
            metrics_to_save[key] = [float(x) if isinstance(x, np.number) else x for x in value]
    
    with open(os.path.join(METRICS_DIR, f"{filename_base}_metrics.json"), 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Save labels and predictions as numpy arrays
    np.save(os.path.join(METRICS_DIR, f"{filename_base}_true_labels.npy"), true_labels)
    np.save(os.path.join(METRICS_DIR, f"{filename_base}_pred_labels.npy"), predicted_labels)
    
    print(f"Saved metrics and labels for {model_name}")

def load_metrics(model_name):
    """
    Load metrics and labels from disk.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        tuple: (metrics, true_labels, predicted_labels)
    """
    # Create a clean filename from the model name
    filename_base = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    metrics_path = os.path.join(METRICS_DIR, f"{filename_base}_metrics.json")
    true_labels_path = os.path.join(METRICS_DIR, f"{filename_base}_true_labels.npy")
    pred_labels_path = os.path.join(METRICS_DIR, f"{filename_base}_pred_labels.npy")
    
    # Check if all files exist
    if not all(os.path.exists(p) for p in [metrics_path, true_labels_path, pred_labels_path]):
        return None, None, None
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load labels
    true_labels = np.load(true_labels_path)
    predicted_labels = np.load(pred_labels_path)
    
    return metrics, true_labels, predicted_labels

def load_all_metrics():
    """
    Load all saved metrics.
    
    Returns:
        dict: Dictionary of model results
    """
    if not os.path.exists(METRICS_DIR):
        print(f"Metrics directory {METRICS_DIR} does not exist")
        return {}
    
    results = {}
    model_architectures = ['Gated CNN (MAMBAOut)', 'MAMBA', 'Transformer']
    token_configurations = ['1 Token', 'Balanced', 'Max Tokens']
    
    for architecture in model_architectures:
        for token_config in token_configurations:
            model_name = f"{architecture} ({token_config})"
            metrics, true_labels, predicted_labels = load_metrics(model_name)
            
            if metrics is not None:
                results[model_name] = metrics
                results[model_name]['true_labels'] = true_labels
                results[model_name]['predicted_labels'] = predicted_labels
                print(f"Loaded metrics for {model_name}")
    
    return results

# ===== PLOTTING FUNCTIONS =====

def plot_precision_recall_f1_by_sample_size(results, class_names, save_path=None):
    """
    Create a plot showing precision, recall, and F1 score vs sample size for different models.
    
    Args:
        results (dict): Dictionary of results for each model
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Comparison of Metrics Across Models (with Trend Lines)", fontsize=20)

    metrics = ["per_class_precision", "per_class_recall", "per_class_f1"]
    titles = ["Precision", "Recall", "F1 Score"]
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        for model_name, model_results in results.items():
            # Extract model type (without token size)
            model_type = model_name.split(' (')[0]
            color = MODEL_COLORS.get(model_type, 'gray')
            
            y_values = np.array(model_results[metric])
            sample_sizes = np.array(model_results['sample_sizes'])
            
            # Scatter plot
            ax.scatter(sample_sizes, y_values, color=color, label=model_name, 
                      s=100, edgecolors='k', alpha=0.7)

            # Logarithmic regression: Fit a line to log10(sample_size) vs metric
            # Add a small constant to avoid log(0)
            valid_indices = sample_sizes > 0
            if np.sum(valid_indices) > 1:  # Need at least two points with non-zero sample size
                log_sample_sizes = np.log10(sample_sizes[valid_indices] + 1)
                valid_y_values = y_values[valid_indices]
                
                if len(log_sample_sizes) > 1:  # Need at least two points for regression
                    coeffs = np.polyfit(log_sample_sizes, valid_y_values, 1)
                    trend_line = np.poly1d(coeffs)

                    # Generate smooth x values for the trend line
                    x_trend = np.linspace(log_sample_sizes.min(), log_sample_sizes.max(), 100)
                    y_trend = trend_line(x_trend)

                    # Convert back to original scale
                    x_trend_original = 10 ** x_trend - 1  # Convert log scale back to normal scale
                    ax.plot(x_trend_original, y_trend, color=color, linestyle='dashed', linewidth=2)

        ax.set_xscale("log")
        ax.set_xlabel("Log10 Sample Size (Total Number of Samples)")
        ax.set_ylabel(title)
        
        max_sample_size = max([max(results[model]['sample_sizes']) for model in results])
        ax.set_xlim(0.1, max_sample_size * 1.05)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Sample Size vs {title}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved sample size correlation plot to {save_path}")

    return fig

def plot_confusion_matrices(results, class_names, save_path=None):
    """
    Plot confusion matrices for all models.
    
    Args:
        results (dict): Dictionary of results for each model
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    n_models = len(results)
    n_cols = 3  # 3 columns
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
        
    # Flatten in case we have a single row
    axes_flat = axes.flatten()
    
    for i, (model_name, model_results) in enumerate(results.items()):
        if i < len(axes_flat):
            ax = axes_flat[i]
            
            # Compute confusion matrix from true and predicted labels
            true_labels = model_results['true_labels']
            pred_labels = model_results['predicted_labels']
            
            # For multi-label classification, we'll plot a heatmap of label co-occurrence
            # This is a simplified view for multiple labels
            cooccurrence = np.zeros((len(class_names), len(class_names)))
            for j in range(len(class_names)):
                for k in range(len(class_names)):
                    # Count instances where model predicted j and true label was k
                    cooccurrence[j, k] = np.sum((pred_labels[:, j] == 1) & (true_labels[:, k] == 1))
            
            # Only show a subset of classes if there are many
            max_classes_to_show = 20
            if len(class_names) > max_classes_to_show:
                # Find the most active classes
                class_activity = cooccurrence.sum(axis=0) + cooccurrence.sum(axis=1)
                top_indices = np.argsort(-class_activity)[:max_classes_to_show]
                cooccurrence = cooccurrence[top_indices, :][:, top_indices]
                display_names = [class_names[i] for i in top_indices]
            else:
                display_names = class_names
            
            sns.heatmap(cooccurrence, annot=False, cmap="Blues", ax=ax, 
                        xticklabels=display_names, yticklabels=display_names)
            ax.set_title(f"{model_name}")
            ax.set_ylabel("Predicted")
            ax.set_xlabel("True")
            
            # Rotate axis labels if we have many classes
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved confusion matrices to {save_path}")
        
    return fig

# Plots for thesis visualization
def create_bar_comparison(results, metric, title=None, save_path=None):
    """Create a bar chart comparing models on a specific metric"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Prepare data for plotting
    model_names = list(results.keys())
    model_types = [name.split(' (')[0] for name in model_names]
    token_sizes = [name.split(' (')[1].replace(')', '') for name in model_names]
    values = [results[name][metric] for name in model_names]
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'model': model_types,
        'tokens': token_sizes,
        'value': values,
        'display_name': [f"{m}\n({t})" for m, t in zip(model_types, token_sizes)]
    })
    
    # Create bar chart
    bars = sns.barplot(
        x='display_name', 
        y='value',
        hue='model',
        data=plot_data,
        palette=MODEL_COLORS,
        ax=ax,
        legend=False
    )
    
    # Find the best value and highlight it
    if 'loss' in metric.lower():
        best_idx = plot_data['value'].idxmin()
    else:
        best_idx = plot_data['value'].idxmax()
        
    # Add a star to the best value
    best_x = best_idx
    best_y = plot_data.iloc[best_idx]['value']
    ax.text(best_x, best_y, '★', ha='center', va='bottom', fontsize=16, color='black')
    
    # Customize the plot
    ax.set_title(title or f"{metric.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel(title or f"{metric.replace('_', ' ').title()}", fontsize=12)
    
    # Format y-axis for percentages
    if 'acc' in metric or 'f1' in metric or 'precision' in metric or 'recall' in metric:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add a light grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved bar comparison to {save_path}")
        
    return fig

def create_token_trend_comparison(results, metrics, title=None, save_path=None):
    """Create line plots showing how metrics change with token size"""
    if isinstance(metrics, str):
        metrics = [metrics]
        
    # Create a figure with subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), dpi=300)
    if len(metrics) == 1:
        axes = [axes]
        
    # Get unique model types
    model_names = list(results.keys())
    model_types = list(set([name.split(' (')[0] for name in model_names]))
    
    # For each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Prepare data for trend lines
        trend_data = pd.DataFrame(columns=['Token Size'] + model_types)
        for token_size in TOKEN_SIZES:
            row = {'Token Size': token_size}
            for model_type in model_types:
                model_name = f"{model_type} ({token_size})"
                if model_name in results:
                    row[model_type] = results[model_name][metric]
            trend_data = pd.concat([trend_data, pd.DataFrame([row])], ignore_index=True)
            
        trend_data = trend_data.set_index('Token Size')
        
        # Plot each model type as a line
        for model_type in model_types:
            if model_type in trend_data.columns:
                ax.plot(
                    trend_data.index,
                    trend_data[model_type],
                    marker='o',
                    label=model_type,
                    color=MODEL_COLORS.get(model_type, 'gray'),
                    linewidth=2,
                    markersize=8
                )
        
        # Highlight the best value at each token size
        for token_size in trend_data.index:
            token_data = trend_data.loc[token_size].dropna()
            if not token_data.empty:
                if 'loss' in metric.lower():
                    best_model = token_data.idxmin()
                    best_value = token_data.min()
                else:
                    best_model = token_data.idxmax()
                    best_value = token_data.max()
                    
                ax.scatter(
                    [token_size],
                    [best_value],
                    s=100,
                    color=MODEL_COLORS.get(best_model, 'gray'),
                    edgecolor='black',
                    zorder=10
                )
        
        # Add title and labels
        metric_title = f"{metric.replace('_', ' ').title()}"
        ax.set_title(metric_title, fontsize=14)
        ax.set_xlabel('Token Size', fontsize=12)
        ax.set_ylabel(metric_title, fontsize=12)
        
        # Format y-axis for percentages
        if 'acc' in metric or 'f1' in metric or 'precision' in metric or 'recall' in metric:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
        # Add legend
        ax.legend(loc='best', frameon=True, fontsize=10)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title if specified
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved token trend comparison to {save_path}")
        
    return fig

def create_heatmap_comparison(results, metrics=None, save_path=None):
    """Create a heatmap comparing all models across metrics"""
    # If no metrics specified, use common ones
    if metrics is None:
        metrics = ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1', 'test_loss']
        
    # Prepare data for heatmap
    model_names = list(results.keys())
    
    # Create DataFrame with metrics for each model
    data = []
    for model_name in model_names:
        row = {'Model': model_name}
        for metric in metrics:
            if metric in results[model_name]:
                row[metric.replace('_', ' ').title()] = results[model_name][metric]
        data.append(row)
        
    df = pd.DataFrame(data)
    df = df.set_index('Model')
    
    # Create heatmap
    plt.figure(figsize=(len(metrics)*1.2 + 2, len(model_names)*0.5 + 2), dpi=300)
    
    # Create the heatmap
    ax = sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn', cbar=False, linewidths=0.5)
    
    # Highlight the best value for each metric
    for j, col in enumerate(df.columns):
        if 'loss' in col.lower():
            best_idx = df[col].idxmin()
        else:
            best_idx = df[col].idxmax()
            
        i = df.index.get_loc(best_idx)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
    plt.title('Model Performance Comparison Across Metrics', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved heatmap comparison to {save_path}")
        
    return plt.gcf()

def create_radar_chart(results, metrics=None, token_size='Max Tokens', save_path=None):
    """Create a radar chart comparing models across multiple metrics"""
    # If no metrics specified, use common ones that work well on radar charts
    if metrics is None:
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    
    # Filter for models with the specified token size
    filtered_results = {}
    for model_name, model_results in results.items():
        if token_size in model_name:
            # Extract model type without token size
            model_type = model_name.split(' (')[0]
            filtered_results[model_type] = model_results
    
    if not filtered_results:
        print(f"No models found with token size: {token_size}")
        return None
    
    # Create the radar chart
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111, polar=True)
    
    # Number of metrics (spokes)
    N = len(metrics)
    
    # Create angles for each metric
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Set up the radar chart
    ax.set_theta_offset(np.pi/2)  # Start from top
    ax.set_theta_direction(-1)    # Clockwise
    
    # Set labels at the correct angles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics], fontsize=12)
    
    # Draw the grid lines (circles)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Draw axis lines
    for a in angles[:-1]:
        ax.plot([a, a], [0, 1], color='grey', alpha=0.7, linewidth=1, linestyle='--')
    
    # Plot each model
    for model_type, model_results in filtered_results.items():
        values = []
        for metric in metrics:
            if metric in model_results:
                values.append(model_results[metric])
            else:
                values.append(0)
                
        # Close the polygon by repeating the first value
        values += values[:1]
        
        # Plot values
        ax.plot(angles, values, color=MODEL_COLORS.get(model_type, 'gray'), linewidth=2, label=model_type)
        ax.fill(angles, values, color=MODEL_COLORS.get(model_type, 'gray'), alpha=0.25)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    
    # Set title
    ax.set_title(f"Model Comparison ({token_size})", fontsize=16, y=1.08)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved radar chart to {save_path}")
        
    return fig

def create_comprehensive_plot(results, save_path=None):
    """Create a comprehensive figure with multiple subplots for thesis presentation"""
    fig = plt.figure(figsize=(12, 15), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])
    
    # Plot 1: Bar chart comparing accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_bar_metric_direct(ax1, results, 'accuracy', 'Test Accuracy')
    
    # Plot 2: Bar chart comparing F1 scores
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_bar_metric_direct(ax2, results, 'weighted_f1', 'Weighted F1 Score')
    
    # Plot 3: Line plot showing token size impact on accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_token_trend_direct(ax3, results, 'accuracy', 'Test Accuracy')
    
    # Plot 4: Line plot showing token size impact on F1 scores
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_token_trend_direct(ax4, results, 'weighted_f1', 'Weighted F1 Score')
    
    # Plot 5: Radar chart comparing models
    ax5 = fig.add_subplot(gs[2, :], polar=True)
    _plot_radar_chart_direct(ax5, results)
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved comprehensive plot to {save_path}")
        
    return fig

def _plot_bar_metric_direct(ax, results, metric, title):
    """Helper function for direct plotting on an axis"""
    # Prepare data for plotting
    model_names = list(results.keys())
    model_types = [name.split(' (')[0] for name in model_names]
    token_sizes = [name.split(' (')[1].replace(')', '') for name in model_names]
    values = [results[name].get(metric, 0) for name in model_names]
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'model': model_types,
        'tokens': token_sizes,
        'value': values,
        'display_name': [f"{m}\n({t})" for m, t in zip(model_types, token_sizes)]
    })
    
    # Create bar chart
    colors = [MODEL_COLORS.get(model, 'gray') for model in model_types]
    bars = ax.bar(plot_data['display_name'], plot_data['value'], color=colors)
    
    # Find the best value and highlight it
    if 'loss' in metric.lower():
        best_idx = plot_data['value'].idxmin()
    else:
        best_idx = plot_data['value'].idxmax()
        
    # Add a star to the best value
    best_x = plot_data.iloc[best_idx]['display_name']
    best_y = plot_data.iloc[best_idx]['value']
    ax.text(best_idx, best_y, '★', ha='center', va='bottom', fontsize=16, color='black')
    
    # Customize the plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel(title, fontsize=12)
    
    # Format y-axis for percentages
    if 'acc' in metric or 'f1' in metric or 'precision' in metric or 'recall' in metric:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add a light grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return ax

def _plot_token_trend_direct(ax, results, metric, title=None, lower_is_better=False):
    """Helper function for direct token trend plotting on an axis"""
    # Get unique model types and token sizes
    model_names = list(results.keys())
    model_types = list(set([name.split(' (')[0] for name in model_names]))
    
    # Prepare data for trend lines
    trend_data = {}
    for model_type in model_types:
        trend_data[model_type] = {}
        for token_size in TOKEN_SIZES:
            model_name = f"{model_type} ({token_size})"
            if model_name in results and metric in results[model_name]:
                trend_data[model_type][token_size] = results[model_name][metric]
    
    # Plot each model type as a line
    for model_type, values in trend_data.items():
        if values:  # Only plot if we have data
            token_sizes = list(values.keys())
            metric_values = list(values.values())
            ax.plot(
                token_sizes,
                metric_values,
                marker='o',
                label=model_type,
                color=MODEL_COLORS.get(model_type, 'gray'),
                linewidth=2,
                markersize=8
            )
    
    # Highlight the best value at each token size
    for token_size in TOKEN_SIZES:
        token_values = {}
        for model_type, values in trend_data.items():
            if token_size in values:
                token_values[model_type] = values[token_size]
                
        if token_values:  # Only find best if we have data
            if lower_is_better:
                best_model = min(token_values, key=token_values.get)
                best_value = token_values[best_model]
            else:
                best_model = max(token_values, key=token_values.get)
                best_value = token_values[best_model]
                
            ax.scatter(
                [token_size],
                [best_value],
                s=100,
                color=MODEL_COLORS.get(best_model, 'gray'),
                edgecolor='black',
                zorder=10
            )
    
    # Add title and labels
    ax.set_title(title or f"{metric.replace('_', ' ').title()}", fontsize=14)
    ax.set_xlabel('Token Size', fontsize=12)
    ax.set_ylabel(title or f"{metric.replace('_', ' ').title()}", fontsize=12)
    
    # Format y-axis for percentages
    if 'acc' in metric or 'f1' in metric or 'precision' in metric or 'recall' in metric:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
    # Add legend
    ax.legend(loc='best', frameon=True, fontsize=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax

def _plot_radar_chart_direct(ax, results, token_size='Max Tokens'):
    """Helper function for direct radar chart plotting on an axis"""
    # Define metrics for radar chart
    metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    
    # Filter for models with the specified token size
    filtered_results = {}
    for model_name, model_results in results.items():
        if token_size in model_name:
            # Extract model type without token size
            model_type = model_name.split(' (')[0]
            filtered_results[model_type] = model_results
    
    if not filtered_results:
        ax.text(0, 0, f"No models found with token size: {token_size}", 
               ha='center', va='center', fontsize=12)
        return ax
    
    # Number of metrics (spokes)
    N = len(metrics)
    
    # Create angles for each metric
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Set up the radar chart
    ax.set_theta_offset(np.pi/2)  # Start from top
    ax.set_theta_direction(-1)    # Clockwise
    
    # Set labels at the correct angles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metric.replace('_', ' ').title() for metric in metrics], fontsize=12)
    
    # Draw the grid lines (circles)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Draw axis lines
    for a in angles[:-1]:
        ax.plot([a, a], [0, 1], color='grey', alpha=0.7, linewidth=1, linestyle='--')
    
    # Plot each model
    for model_type, model_results in filtered_results.items():
        values = []
        for metric in metrics:
            if metric in model_results:
                values.append(model_results[metric])
            else:
                values.append(0)
                
        # Close the polygon by repeating the first value
        values += values[:1]
        
        # Plot values
        ax.plot(angles, values, color=MODEL_COLORS.get(model_type, 'gray'), linewidth=2, label=model_type)
        ax.fill(angles, values, color=MODEL_COLORS.get(model_type, 'gray'), alpha=0.25)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    
    # Set title
    ax.set_title(f"Model Comparison ({token_size})", fontsize=14, y=1.08)
    
    return ax

# ===== SINGLE MODEL EVALUATION FUNCTION =====

def evaluate_single_model(model_path, model_name, data_path, force=False):
    """
    Evaluate a single model and save its metrics.
    
    Args:
        model_path (str): Path to the model file
        model_name (str): Name of the model
        data_path (str): Path to the data
        force (bool): Whether to force re-evaluation even if metrics exist
    
    Returns:
        bool: Whether evaluation was successful
    """
    # Check if metrics already exist
    metrics, true_labels, predicted_labels = load_metrics(model_name)
    if metrics is not None and not force:
        print(f"Metrics for {model_name} already exist. Use --force to re-evaluate.")
        return True
    
    # If model_path doesn't exist, try to construct it
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Attempting to locate model...")
        
        # Extract model architecture and token config from name
        parts = model_name.split(' (')
        if len(parts) == 2:
            architecture = parts[0]
            token_config = parts[1].replace(')', '')
            
            # Try different filename formats and extensions
            possible_filenames = [
                f"{architecture.lower().replace(' ', '_')}_{token_config.lower().replace(' ', '_')}.pth",
                f"{architecture.lower().replace(' ', '_')}_{token_config.lower().replace(' ', '_')}.pt",
                f"{architecture.lower()}_{token_config.lower().replace(' ', '_')}.pth",
                f"{architecture.lower()}_{token_config.lower().replace(' ', '_')}.pt",
                f"{architecture.lower().replace(' ', '')}.pth",
                f"{architecture.lower().replace(' ', '')}.pt"
            ]
            
            # Check if model folder is a directory or the parent folder
            model_dir = os.path.dirname(model_path)
            if not model_dir:
                model_dir = "."
                
            for filename in possible_filenames:
                potential_path = os.path.join(model_dir, filename)
                if os.path.exists(potential_path):
                    model_path = potential_path
                    print(f"Found model at {model_path}")
                    break
            
            if not os.path.exists(model_path):
                print(f"Could not find model file for {model_name}. Tried: {possible_filenames}")
                return False
        else:
            print(f"Invalid model name format: {model_name}")
            return False
    
    # Prepare data
    test_dataloader, class_names = prepare_data(data_path)
    if test_dataloader is None:
        print("Error: Failed to load test data. Exiting.")
        return False
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    if model is None:
        print(f"Failed to load model from {model_path}")
        return False
    
    # Move model to device
    model = model.to(device)
    
    # Evaluate model
    try:
        print(f"Evaluating {model_name}...")
        metrics, true_labels, predicted_labels = evaluate_model(model, test_dataloader, device)
        
        # Save metrics
        save_metrics(model_name, metrics, true_labels, predicted_labels)
        
        print(f"Evaluation completed for {model_name}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        
        # Clean up to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

# ===== VISUALIZATION GENERATION FUNCTION =====

def generate_visualizations(class_names_path=None):
    """
    Generate visualizations from saved metrics.
    
    Args:
        class_names_path (str, optional): Path to class names file
    """
    print("Loading all saved metrics...")
    results = load_all_metrics()
    
    if not results:
        print("No metrics found. Run evaluation first.")
        return
    
    print(f"Loaded metrics for {len(results)} models.")
    
    # Load class names
    class_names = load_class_names(class_names_path) if class_names_path else load_class_names()
    
    print("\nGenerating visualization plots...")
    
    # Generate plots
    sample_size_plot = plot_precision_recall_f1_by_sample_size(
        results, class_names, os.path.join(OUTPUT_DIR, "sample_size_metrics.pdf"))
    
    confusion_matrices = plot_confusion_matrices(
        results, class_names, os.path.join(OUTPUT_DIR, "confusion_matrices.pdf"))
    
    # Generate thesis-style plots
    bar_acc = create_bar_comparison(
        results, 'accuracy', 'Test Accuracy', os.path.join(OUTPUT_DIR, "bar_accuracy.pdf"))
    
    bar_f1 = create_bar_comparison(
        results, 'weighted_f1', 'Weighted F1 Score', os.path.join(OUTPUT_DIR, "bar_f1.pdf"))
    
    token_trends = create_token_trend_comparison(
        results, ['accuracy', 'macro_f1', 'weighted_f1'], 
        'Effect of Token Size on Model Performance', 
        os.path.join(OUTPUT_DIR, "token_trends.pdf"))
    
    heatmap = create_heatmap_comparison(
        results, save_path=os.path.join(OUTPUT_DIR, "heatmap.pdf"))
    
    radar = create_radar_chart(
        results, save_path=os.path.join(OUTPUT_DIR, "radar_chart.pdf"))
    
    comprehensive = create_comprehensive_plot(
        results, save_path=os.path.join(OUTPUT_DIR, "comprehensive_comparison.pdf"))
    
    print(f"\nAll plots saved to {OUTPUT_DIR}")
    
    # Create a summary CSV file with all metrics
    summary_data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):  # Only include scalar metrics
                row[metric] = value
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Metrics summary saved to {summary_path}")
    
    return True

# ===== MAIN SCRIPT =====

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate star classification models')
    parser.add_argument('--model_folder', type=str, default='Comparing_Mambas_Trans/',
                        help='Path to the folder containing saved models')
    parser.add_argument('--data_path', type=str, default='Pickles/',
                        help='Path to the data directory')
    parser.add_argument('--class_names_path', type=str, default=None,
                        help='Path to class names file')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Only generate visualizations from existing metrics')
    parser.add_argument('--eval_model', type=str, default=None,
                        help='Evaluate a specific model (format: "Architecture (Token Size)")')
    parser.add_argument('--force', action='store_true',
                        help='Force re-evaluation even if metrics exist')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.visualize_only:
        generate_visualizations(args.class_names_path)
    elif args.eval_model:
        # Evaluate a specific model
        model_architecture = args.eval_model.split(' (')[0]
        token_config = args.eval_model.split(' (')[1].replace(')', '')
        
        # Construct model filename (trying both .pth and .pt extensions)
        model_filename = f"{model_architecture.lower().replace(' ', '_')}_{token_config.lower().replace(' ', '_')}.pth"
        model_path = os.path.join(args.model_folder, model_filename)
        
        success = evaluate_single_model(model_path, args.eval_model, args.data_path, args.force)
        
        if success:
            print(f"Successfully evaluated {args.eval_model}")
            
            # Ask if user wants to generate visualizations
            generate = input("Generate visualizations? [y/N]: ").lower() == 'y'
            if generate:
                generate_visualizations(args.class_names_path)
    else:
        # Evaluate all models sequentially
        model_architectures = ['Gated CNN (MAMBAOut)', 'MAMBA', 'Transformer']
        token_configurations = ['1 Token', 'Balanced', 'Max Tokens']
        
        successful_evals = 0
        
        for architecture in model_architectures:
            for token_config in token_configurations:
                model_name = f"{architecture} ({token_config})"
                
                # Construct model filename with .pth extension
                model_filename = f"{architecture.lower().replace(' ', '_')}_{token_config.lower().replace(' ', '_')}.pth"
                model_path = os.path.join(args.model_folder, model_filename)
                
                success = evaluate_single_model(model_path, model_name, args.data_path, args.force)
                
                if success:
                    successful_evals += 1
        
        print(f"\nCompleted evaluation of {successful_evals} out of {len(model_architectures) * len(token_configurations)} models.")
        
        if successful_evals > 0:
            # Ask if user wants to generate visualizations
            generate = input("Generate visualizations? [y/N]: ").lower() == 'y'
            if generate:
                generate_visualizations(args.class_names_path)



# Rotary Position Embeddings implementation
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
        # Generate position embeddings once at initialization
        self._generate_embeddings()
        
    def _generate_embeddings(self):
        t = torch.arange(self.max_seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(self.max_seq_len, 1, -1)
        sin = emb.sin().view(self.max_seq_len, 1, -1)
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)
        
    def forward(self, seq_len):
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to q and k tensors."""
    # Handle the case where q and k have shape [batch_size, seq_len, head_dim]
    # or [batch_size, n_heads, seq_len, head_dim]
    if q.dim() == 3:
        # [batch_size, seq_len, head_dim] -> [batch_size, seq_len, 1, head_dim]
        q = q.unsqueeze(2)
        k = k.unsqueeze(2)
        # After this operation, we squeeze back
        squeeze_after = True
    else:
        squeeze_after = False
    
    # Reshape cos and sin for proper broadcasting
    # [seq_len, 1, head_dim] -> [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    
    # Apply rotation
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    if squeeze_after:
        q_rot = q_rot.squeeze(2)
        k_rot = k_rot.squeeze(2)
    
    return q_rot, k_rot

class RotarySelfAttention(nn.Module):
    """Self-attention with rotary position embeddings."""
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        
        # QKV projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Rotary positional embedding
        self.rope = RotaryEmbedding(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            output: Tensor of same shape as input
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        q = self.q_proj(x)  # [batch_size, seq_len, dim]
        k = self.k_proj(x)  # [batch_size, seq_len, dim]
        v = self.v_proj(x)  # [batch_size, seq_len, dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Get position embeddings
        cos, sin = self.rope(seq_len)
        
        # Apply rotary position embeddings to q and k
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Transpose for efficient batch matrix multiplication
        q = q.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        
        # Compute scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, n_heads, seq_len, seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)  # [batch_size, n_heads, seq_len, head_dim]
        
        # Reshape back to original format
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Apply output projection
        output = self.out_proj(output)
        
        return output

class TransformerBlock(nn.Module):
    """Transformer block with rotary self-attention and feed-forward network."""
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RotarySelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # FFN with residual connection
        x = x + self.ffn(self.norm2(x))
        
        return x

class TransformerFeatureExtractor(nn.Module):
    """Stack of transformer blocks for feature extraction."""
    def __init__(self, d_model, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Processed tensor of same shape
        """
        for layer in self.layers:
            x = layer(x)
        return x

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block to attend from one modality to another.
    """
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            batch_first=True
        )
    
    def forward(self, x, context):
        """
        Args:
            x: Query tensor of shape [batch_size, seq_len_q, dim]
            context: Key/value tensor of shape [batch_size, seq_len_kv, dim]
        
        Returns:
            Output tensor of shape [batch_size, seq_len_q, dim]
        """
        x_norm = self.norm(x)
        attn_output, _ = self.attention(
            query=x_norm,
            key=context,
            value=context
        )
        return x + attn_output


class StarClassifierFusionTransformer(nn.Module):
    """Transformer-based feature extractor with tokenization for multi-modal fusion."""
    def __init__(
        self,
        d_model_spectra,
        d_model_gaia,
        num_classes,
        input_dim_spectra,
        input_dim_gaia,
        token_dim_spectra=64,  # Size of each token for spectra
        token_dim_gaia=2,      # Size of each token for gaia
        n_layers=6,
        n_heads=8,
        use_cross_attention=True,
        n_cross_attn_heads=8,
        dropout=0.1,
    ):
        """
        Args:
            d_model_spectra (int): embedding dimension for the spectra Transformer
            d_model_gaia (int): embedding dimension for the gaia Transformer
            num_classes (int): multi-label classification
            input_dim_spectra (int): # of features for spectra
            input_dim_gaia (int): # of features for gaia
            token_dim_spectra (int): size of each token for spectra features
            token_dim_gaia (int): size of each token for gaia features
            n_layers (int): depth for each Transformer
            n_heads (int): number of attention heads
            use_cross_attention (bool): whether to use cross-attention
            n_cross_attn_heads (int): number of heads for cross-attention
            dropout (float): dropout rate
        """
        super().__init__()

        # --- Feature Tokenizers ---
        self.tokenizer_spectra = FeatureTokenizer(
            input_dim=input_dim_spectra,
            token_dim=token_dim_spectra,
            d_model=d_model_spectra
        )
        
        self.tokenizer_gaia = FeatureTokenizer(
            input_dim=input_dim_gaia,
            token_dim=token_dim_gaia,
            d_model=d_model_gaia
        )

        # --- Transformer for spectra ---
        self.transformer_spectra = TransformerFeatureExtractor(
            d_model=d_model_spectra,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # --- Transformer for gaia ---
        self.transformer_gaia = TransformerFeatureExtractor(
            d_model=d_model_gaia,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # --- Cross Attention (Optional) ---
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn_block_spectra = CrossAttentionBlock(d_model_spectra, n_heads=n_cross_attn_heads)
            self.cross_attn_block_gaia = CrossAttentionBlock(d_model_gaia, n_heads=n_cross_attn_heads)

        # --- Final Classifier ---
        fusion_dim = d_model_spectra + d_model_gaia
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)
        )
    
    def forward(self, x_spectra, x_gaia):
        """
        Args:
            x_spectra: Spectra features of shape [batch_size, input_dim_spectra]
            x_gaia: Gaia features of shape [batch_size, input_dim_gaia]
            
        Returns:
            logits: Classification logits of shape [batch_size, num_classes]
        """
        # Tokenize input features
        # From [batch_size, input_dim] to [batch_size, num_tokens, d_model]
        x_spectra_tokens = self.tokenizer_spectra(x_spectra)
        x_gaia_tokens = self.tokenizer_gaia(x_gaia)
        
        # Process through transformers
        x_spectra = self.transformer_spectra(x_spectra_tokens)  # [batch_size, num_tokens_spectra, d_model]
        x_gaia = self.transformer_gaia(x_gaia_tokens)          # [batch_size, num_tokens_gaia, d_model]

        # Optional cross-attention
        if self.use_cross_attention:
            x_spectra = self.cross_attn_block_spectra(x_spectra, x_gaia)
            x_gaia = self.cross_attn_block_gaia(x_gaia, x_spectra)
        
        # Global pooling over sequence dimension
        x_spectra = x_spectra.mean(dim=1)  # [batch_size, d_model]
        x_gaia = x_gaia.mean(dim=1)        # [batch_size, d_model]

        # Concatenate for fusion
        x_fused = torch.cat([x_spectra, x_gaia], dim=-1)  # [batch_size, 2*d_model]

        # Final classification
        logits = self.classifier(x_fused)  # [batch_size, num_classes]
        
        return logits