import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import wandb
import gc
from sklearn.model_selection import train_test_split
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block

# Create a custom Vision Transformer model
# Create dataset classes (using your BalancedDataset approach) and training function
class BalancedDataset(Dataset):
    def __init__(self, X, y, limit_per_label=1600):
        self.X = X
        self.y = y
        self.limit_per_label = limit_per_label
        self.classes = np.unique(y)
        self.indices = self.balance_classes()

    def balance_classes(self):
        indices = []
        for cls in self.classes:
            cls_indices = np.where(self.y == cls)[0]
            if len(cls_indices) > self.limit_per_label:
                cls_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
            indices.extend(cls_indices)
        np.random.shuffle(indices)
        return indices

    def re_sample(self):
        self.indices = self.balance_classes()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.X[index], self.y[index]
# Custom Dataset for validation with limit per class
class BalancedValidationDataset(Dataset):
    def __init__(self, X, y, limit_per_label=400):
        self.X = X
        self.y = y
        self.limit_per_label = limit_per_label
        self.classes = np.unique(y)
        self.indices = self.balance_classes()

    def balance_classes(self):
        indices = []
        for cls in self.classes:
            cls_indices = np.where(self.y == cls)[0]
            if len(cls_indices) > self.limit_per_label:
                cls_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
            indices.extend(cls_indices)
        np.random.shuffle(indices)
        return indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.X[index], self.y[index]
 
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class VisionTransformer1D(nn.Module):
    def __init__(self, input_size=3748, num_classes=4, patch_sizes=[20, 40], overlap=0.5, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.2):
        super(VisionTransformer1D, self).__init__()
        self.num_branches = len(patch_sizes)
        self.dim = dim
        self.overlap = overlap
        self.branches = nn.ModuleList()
        
        # Set up branches for different patch sizes
        for patch_size in patch_sizes:
            stride = int(patch_size * (1 - overlap))
            max_patches = (input_size - patch_size) // stride + 1
            patch_embed = nn.Linear(patch_size, dim)
            pos_embedding = nn.Embedding(max_patches, dim)  # Changed to nn.Embedding to act as a module
            transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout), depth
            )
            self.branches.append(nn.ModuleDict({
                'patch_embed': patch_embed,
                'pos_embedding': pos_embedding,
                'transformer': transformer
            }))
        
        # Cross-Attention for fusion of multiple patch sizes
        self.cross_attention = CrossAttentionBlock(dim, heads)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        batch_size, channels, seq_len = x.shape  # Assuming x has 3 dimensions
        x = x.squeeze(1) if channels == 1 else x  # Remove channel dimension if it's 1
        branch_outputs = []

        
        # Extract patches, embed, and process with transformer for each branch
        for branch in self.branches:
            patch_size = branch['patch_embed'].in_features
            stride = int(patch_size * (1 - self.overlap))
            num_patches = (seq_len - patch_size) // stride + 1
            patches = [x[:, i * stride : i * stride + patch_size] for i in range(num_patches)]
            x_branch = torch.stack(patches, dim=1)
            x_branch = branch['patch_embed'](x_branch) + branch['pos_embedding'](torch.arange(num_patches, device=x.device)).unsqueeze(0)
            x_branch = branch['transformer'](x_branch)
            branch_outputs.append(x_branch)

        # Apply cross-attention to combine the representations from each branch
        x_fused = torch.cat(branch_outputs, dim=1)
        x_fused = self.cross_attention(x_fused)

        # Classification based on the first token representation
        x = self.fc(x_fused[:, 0])
        return x


# Training function with learning rate scheduler
def train_model_vit(model, train_loader, val_loader, test_loader, num_epochs=500, lr=1e-4, max_patience=20, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(max_patience/5), verbose=True)
    criterion = nn.CrossEntropyLoss()
    best_test_loss = float('inf')
    patience = max_patience
    
    for epoch in range(num_epochs):
        train_loader.dataset.re_sample()
        model.train()
        train_loss, train_accuracy = 0.0, 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_accuracy += (outputs.argmax(dim=1) == y_batch).float().mean().item()
        
        # Validation phase
        model.eval()
        val_loss, val_accuracy = 0.0, 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
                val_accuracy += (outputs.argmax(dim=1) == y_val).float().mean().item()
        
        # Test phase
        test_loss, test_accuracy = 0.0, 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                outputs = model(X_test)
                loss = criterion(outputs, y_test)
                test_loss += loss.item() * X_test.size(0)
                test_accuracy += (outputs.argmax(dim=1) == y_test).float().mean().item()
                y_true.extend(y_test.cpu().numpy())
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
        
        # Scheduler step
        scheduler.step(test_loss / len(test_loader.dataset))

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader.dataset),
            "val_loss": val_loss / len(val_loader.dataset),
            "train_accuracy": train_accuracy / len(train_loader),
            "val_accuracy": val_accuracy / len(val_loader),
            "learning_rate": optimizer.param_groups[0]['lr'],
            "test_loss": test_loss / len(test_loader.dataset),
            "test_accuracy": test_accuracy / len(test_loader),
            "confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                y_true=y_true, preds=y_pred, class_names=np.unique(y_true)),
            "classification_report": classification_report(y_true, y_pred, target_names=label_mapping.keys())
        })
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience = max_patience
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    return model

# Set fixed hyperparameters
batch_size = 128
num_classes = 4
patience = 30
num_epochs = 500

# Example usage
if __name__ == "__main__":
    # Load and preprocess your data (example from original script)
    # Load and preprocess data
    X = pd.read_pickle("Pickles/trainv2.pkl")
    y = X["label"]
    label_mapping = {'star': 0, 'binary_star': 1, 'galaxy': 2, 'agn': 3}
    y = y.map(label_mapping).values
    X = X.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
                "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
                "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "label", "obsid"], axis=1).values
    
    # Read test data
    X_test = pd.read_pickle("Pickles/testv2.pkl")
    y_test = X_test["label"].map(label_mapping).values
    X_test = X_test.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
                "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
                "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "label", "obsid"], axis=1).values
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Clear memory
    del X, y
    gc.collect()

    # Convert to torch tensors and create datasets
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = BalancedDataset(X_train, y_train)
    val_dataset = BalancedValidationDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(BalancedValidationDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
                                                    torch.tensor(y_test, dtype=torch.long)), batch_size=batch_size, shuffle=False)
    

# Generate random hyperparameters for the VisionTransformer1D model
def random_hyperparams(patch_size_list, dim_list, depth_list, heads_list, mlp_dim_list, dropout_list, lr_list):
    patch_size = np.random.choice(patch_size_list, 2, replace=False)
    dim = np.random.choice(dim_list)
    depth = np.random.choice(depth_list)
    heads = np.random.choice(heads_list)
    mlp_dim = np.random.choice(mlp_dim_list)
    dropout = np.random.choice(dropout_list)
    lr = np.random.choice(lr_list)
    hyperparams = {"patch_size": patch_size, "dim": dim, "depth": depth, "heads": heads, "mlp_dim": mlp_dim, "dropout": dropout, "lr": lr}
    return hyperparams

# Hyperparameter tuning loop
hyperparams_list = [random_hyperparams(patch_size_list=[1, 250, 3748], dim_list=[32, 128, 512], 
                                       depth_list=[4, 12, 50], heads_list=[2, 8, 32], mlp_dim_list=[64, 128, 256, 512], 
                                       dropout_list=[0.1, 0.4], lr_list=[1e-4, 1e-3]) for _ in range(30)]


for i, hparams in enumerate(hyperparams_list):
    config = {"patch_size": hparams["patch_size"], "dim": hparams["dim"], "depth": hparams["depth"],
              "heads": hparams["heads"], "mlp_dim": hparams["mlp_dim"], "dropout": hparams["dropout"],
              "batch_size": batch_size, "lr": hparams["lr"], "num_epochs": num_epochs, "patience": patience}
    
    wandb.init(project="LAMOST-crossvit", entity="joaoc-university-of-southampton", config=config, mode="offline")
    
    model_vit = VisionTransformer1D(num_classes=num_classes, patch_size=hparams["patch_size"], dim=hparams["dim"],
                                    depth=hparams["depth"], heads=hparams["heads"], mlp_dim=hparams["mlp_dim"], dropout=hparams["dropout"])
    trained_model = train_model_vit(model_vit, train_loader, val_loader, test_loader, num_epochs=num_epochs, lr=hparams["lr"], max_patience=patience)
    
    wandb.finish()