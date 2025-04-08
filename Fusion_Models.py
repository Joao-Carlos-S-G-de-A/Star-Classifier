import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
from mamba_ssm import Mamba2
from timm.models.layers import DropPath
from torch import nn
from torch.nn import functional as F




class CrossAttentionBlock(nn.Module):
    """
    A simple cross-attention block with a feed-forward sub-layer.
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x_q, x_kv):
        """
        Args:
            x_q  : (batch_size, seq_len_q, d_model)
            x_kv : (batch_size, seq_len_kv, d_model)
        """
        # Cross-attention
        attn_output, _ = self.cross_attn(query=x_q, key=x_kv, value=x_kv)
        x = self.norm1(x_q + attn_output)

        # Feed forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class FeatureTokenizer(nn.Module):
    """
    Splits input features into tokens of a specified dimension.
    """
    def __init__(self, input_dim, token_dim, d_model):
        """
        Args:
            input_dim: Dimension of the input features
            token_dim: Dimension of each token
            d_model: Model dimension that each token will be embedded to
        """
        super().__init__()
        self.input_dim = input_dim
        self.token_dim = token_dim
        
        # Calculate number of tokens based on input dimension and token dimension
        self.num_tokens = (input_dim + token_dim - 1) // token_dim  # Ceiling division
        
        # Padding to ensure input_dim is divisible by token_dim
        self.padded_dim = self.num_tokens * token_dim
        
        # Linear projection to embed each token to d_model
        self.token_embed = nn.Linear(token_dim, d_model)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tokenized tensor of shape [batch_size, num_tokens, d_model]
        """
        batch_size = x.shape[0]
        
        # Pad input if needed
        if self.input_dim < self.padded_dim:
            padding = torch.zeros(batch_size, self.padded_dim - self.input_dim, 
                                 dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Reshape into tokens
        x = x.view(batch_size, self.num_tokens, self.token_dim)
        
        # Embed each token to d_model
        x = self.token_embed(x)
        
        return x
    

class GatedCNNBlock(nn.Module):
    """Adaptation of GatedCNNBlock for sequence data with dynamic kernel size adaptation"""
    def __init__(self, dim, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(expand * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        
        # Store these for dynamic convolution sizing
        self.d_conv = d_conv
        self.hidden = hidden
        
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Use simpler approach for sequence length 1 (common case)
        # This avoids dynamic convolution creation
        if d_conv == 1:
            self.use_identity_for_length_1 = True

        
        # Cache for static convolution with kernel size 1 (for length 1 sequences)
        if d_conv == 1:
            self.conv1 = nn.Conv1d(
                in_channels=hidden,
                out_channels=hidden, 
                kernel_size=1,
                padding=0,
                groups=hidden
            )
        else:
            # Dynamic convolution for other lengths
            self.conv = nn.Conv1d(
                in_channels=hidden,
                out_channels=hidden, 
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                groups=hidden
            )

    def forward(self, x):
        # Input shape: [B, seq_len, dim]
        shortcut = x
        x = self.norm(x)
        
        # Split the channels for gating mechanism
        x = self.fc1(x)  # [B, seq_len, hidden*2]
        g, c = torch.chunk(x, 2, dim=-1)  # Each: [B, seq_len, hidden]
        
        # Get sequence length
        batch_size, seq_len, channels = c.shape
        
        # Apply gating mechanism
        c_permuted = c.permute(0, 2, 1)  # [B, hidden, seq_len]
        
        # Special case for sequence length 1 
        if seq_len == 1 and self.use_identity_for_length_1:
            # Use the pre-created kernel size 1 conv, which is like identity but keeps channels
            c_conv = self.conv1(c_permuted)
        else:
            # For other sequence lengths, fallback to kernel size 1 to avoid issues
            # The conv1 layer is already initialized and on the correct device
            c_conv = self.conv(c_permuted)
            c_conv = c_conv[:, :, :seq_len] # Ensure we only take the valid part
        
        c_final = c_conv.permute(0, 2, 1)  # [B, seq_len, hidden]
        
        # Gating mechanism
        x = self.fc2(self.act(g) * c_final)  # [B, seq_len, dim]
        
        x = self.drop_path(x)
        return x + shortcut
    
class GatedCNNBlock(nn.Module):
    """Simplified and fixed GatedCNNBlock that preserves sequence length"""
    def __init__(self, dim, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(expand * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        
        # Properly calculate padding to ensure output length matches input length
        # For kernel_size k, padding needed is (k-1)/2, rounded up for even kernels
        self.d_conv = d_conv
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
    def __init__(self, d_model,  d_conv=4, expand=2, depth=1, drop_path=0.):
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
        x: (B, seq_len_x, dim)
        context: (B, seq_len_context, dim)
        """
        x_norm = self.norm(x)
        attn_output, _ = self.attention(
            query=x_norm,
            key=context,
            value=context
        )
        return x + attn_output


    
class StarClassifierFusionMambaOut(nn.Module):
    def __init__(
        self,
        d_model_spectra,
        d_model_gaia,
        num_classes,
        input_dim_spectra,
        input_dim_gaia,
        token_dim_spectra,  # New parameter for token size
        token_dim_gaia,      # New parameter for token size
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
            d_state (int): state dimension for Mamba
            d_conv (int): convolution dimension for Mamba
            expand (int): expansion factor for Mamba
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
        
        # Process through Mamba models
        x_spectra = self.mamba_spectra(x_spectra_tokens)  # [batch_size, num_tokens_spectra, d_model]
        x_gaia = self.mamba_gaia(x_gaia_tokens)          # [batch_size, num_tokens_gaia, d_model]

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
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

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
    



    