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
import torch
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())

if torch.cuda.is_available():
    device = torch.device("cuda")
else:   
    device = torch.device("cpu")


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
# Training function (similar to your ConvNet setup but using WandB)
def train_model_vit(model, train_loader, val_loader, test_loader, num_epochs=10, lr=1e-4, patience=5, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Re-sample training data at the start of each epoch
        train_loader.dataset.re_sample()
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_accuracy = (outputs.argmax(dim=1) == y_batch).float().mean()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
                val_accuracy = (outputs.argmax(dim=1) == y_val).float().mean()
        
        # Test phase
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                outputs = model(X_test)
                loss = criterion(outputs, y_test)
                test_loss += loss.item() * X_test.size(0)
                test_accuracy += (outputs.argmax(dim=1) == y_test).float().mean()


        # Log metrics to WandB
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        test_loss /= len(test_loader.dataset)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch, 
                   "train_accuracy": train_accuracy.item(), "val_accuracy": val_accuracy.item(), 
                   "test_accuracy": test_accuracy.item(), "test_loss": test_loss})
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    return model
class VisionTransformer1D(nn.Module):
    def __init__(self, input_size=3748, num_classes=4, patch_size=17, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1):
        super(VisionTransformer1D, self).__init__()

        # Store patch size and dimensionality for embedding
        self.patch_size = patch_size
        self.dim = dim

        # Patch Embedding layer
        self.patch_embed = nn.Linear(patch_size, dim)

        # Positional Encoding (initialize to a reasonable size, but we will adjust it dynamically)
        max_patches = (input_size + patch_size - 1) // patch_size  # Approximate max patches
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, dim))

        # Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout),
            depth
        )

        # MLP Head
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # Handle input dimensions and ensure padding for patch divisibility
        batch_size, channels, seq_len = x.shape  # Assuming x has 3 dimensions
        x = x.squeeze(1) if channels == 1 else x  # Remove channel dimension if it's 1

        # Calculate required padding for divisibility by patch_size and pad input
        pad_length = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        x = nn.functional.pad(x, (0, pad_length))
        
        # Dynamically calculate number of patches after padding
        num_patches = x.size(1) // self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size)  # Reshape to patches
        
        # Embed patches and add positional encoding (resize pos_embedding if needed)
        if self.pos_embedding.size(1) != num_patches:
            self.pos_embedding = nn.Parameter(self.pos_embedding[:, :num_patches, :])
        x = self.patch_embed(x) + self.pos_embedding

        # Transformer forward pass
        x = self.transformer(x)

        # Classify based on the first token representation
        x = self.fc(x[:, 0])

        return x
batch_size = 128



# Example usage
if __name__ == "__main__":
    # Load and preprocess your data (example from original script)
    # Load and preprocess data
    X = pd.read_pickle("Pickles/train.pkl")
    y = X["label"]
    label_mapping = {'star': 0, 'binary_star': 1, 'galaxy': 2, 'agn': 3}
    y = y.map(label_mapping).values
    
    X = X.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
                "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
                "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "label"], axis=1).values
    
    # Read test data
    X_test = pd.read_pickle("Pickles/test.pkl")
    y_test = X_test["label"].map(label_mapping).values
    X_test = X_test.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
                "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
                "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "label"], axis=1).values
    
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

torch.cuda.empty_cache()
# Initialize WandB project
wandb.init(project="spectra-classification-vit", entity="joaoc-university-of-southampton")
# Initialize and train the model
model_vit = VisionTransformer1D(patch_size=10)
trained_model = train_model_vit(model_vit, train_loader, val_loader, test_loader, num_epochs=50, lr=1e-3, patience=10)

# Save the model and finish WandB session
wandb.finish()