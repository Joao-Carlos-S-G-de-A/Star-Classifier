import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader




# Import the memory-efficient MoE model
from MoE_Model_Functions import (
    StarClassifierWithMemoryEfficientMoE, 
    MemoryEfficientMoEProcessor, 
    CrossAttentionBlock,
    MultiModalBalancedMultiLabelDataset
)

# Import the memory-efficient training function
from MoE_Model_Functions import train_model_fusion_with_moe, calculate_metrics, calculate_class_weights


# Import the data to train
batch_size = 128
batch_limit = int(batch_size / 2.5)

# Load datasets
#X_train_full = pd.read_pickle("Pickles/train_data_transformed2.pkl")
#X_test_full = pd.read_pickle("Pickles/test_data_transformed.pkl")
# classes = pd.read_pickle("Pickles/Updated_list_of_Classes.pkl")
import pickle
# Open them in a cross-platform way
with open("Pickles/Updated_List_of_Classes_ubuntu.pkl", "rb") as f:
    classes = pickle.load(f)  # This reads the actual data
with open("Pickles/train_data_transformed_ubuntu.pkl", "rb") as f:
    X_train_full = pickle.load(f)
with open("Pickles/test_data_transformed_ubuntu.pkl", "rb") as f:
    X_test_full = pickle.load(f)




# Extract labels
y_train_full = X_train_full[classes]
y_test = X_test_full[classes]

# Drop labels from both datasets
X_train_full.drop(classes, axis=1, inplace=True)
X_test_full.drop(classes, axis=1, inplace=True)


# Columns for spectral data (assuming all remaining columns after removing Gaia are spectra)
gaia_columns = ["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", 
                "pmra_error", "pmdec_error", "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", 
                "phot_bp_mean_flux", "phot_rp_mean_flux", "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", 
                "flagnoflux"]

# Spectra data (everything that is not Gaia-related) and the column 'otype'
X_train_spectra = X_train_full.drop(columns={"otype", "obsid", *gaia_columns})
X_test_spectra = X_test_full.drop(columns={"otype", "obsid", *gaia_columns})

# Gaia data (only the selected columns)
X_train_gaia = X_train_full[gaia_columns]
X_test_gaia = X_test_full[gaia_columns]

# Count nans and infs in x_train_gaia
print(X_train_gaia.isnull().sum())
print(X_train_gaia.isin([np.inf, -np.inf]).sum())


# Free up memory
del X_train_full, X_test_full
gc.collect()



# Split training set into training and validation
X_train_spectra, X_val_spectra, X_train_gaia, X_val_gaia, y_train, y_val = train_test_split(
    X_train_spectra, X_train_gaia, y_train_full, test_size=0.2, random_state=42
)

# Free memory
del y_train_full
gc.collect()



# Convert spectra and Gaia data into PyTorch tensors
X_train_spectra = torch.tensor(X_train_spectra.values, dtype=torch.float32)
X_val_spectra = torch.tensor(X_val_spectra.values, dtype=torch.float32)
X_test_spectra = torch.tensor(X_test_spectra.values, dtype=torch.float32)



X_train_gaia = torch.tensor(X_train_gaia.values, dtype=torch.float32)
X_val_gaia = torch.tensor(X_val_gaia.values, dtype=torch.float32)
X_test_gaia = torch.tensor(X_test_gaia.values, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Print dataset shapes
print(f"X_train_spectra shape: {X_train_spectra.shape}")
print(f"X_val_spectra shape: {X_val_spectra.shape}")
print(f"X_test_spectra shape: {X_test_spectra.shape}")

print(f"X_train_gaia shape: {X_train_gaia.shape}")
print(f"X_val_gaia shape: {X_val_gaia.shape}")
print(f"X_test_gaia shape: {X_test_gaia.shape}")
train_test_split
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"y_test shape: {y_test.shape}")


train_dataset = MultiModalBalancedMultiLabelDataset(X_train_spectra, X_train_gaia, y_train, limit_per_label=batch_limit)
val_dataset = MultiModalBalancedMultiLabelDataset(X_val_spectra, X_val_gaia, y_val, limit_per_label=batch_limit)
test_dataset = MultiModalBalancedMultiLabelDataset(X_test_spectra, X_test_gaia, y_test, limit_per_label=batch_limit)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print the number of samples in each dataset
print(f"Train dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")


# Start the training process
if __name__ == "__main__":
    # Example config
    d_model_spectra = 1024  # Reduced from 2048 to save memory
    d_model_gaia = 1024     # Reduced from 2048 to save memory
    num_classes = 55
    input_dim_spectra = 3647
    input_dim_gaia = 18
    n_layers = 12           # Reduced from 20 to save memory while maintaining depth
    num_experts = 8        # Number of experts in MoE layers
    top_k = 2               # Number of experts used per token
    lr = 1e-4               # Adjusted learning rate for the MoE architecture
    patience = 100          # Reduced due to faster convergence with MoE
    num_epochs = 500        # Can be reduced as MoE often converges faster
    batch_size = 8
    gradient_accumulation_steps = 8  # Effective batch size = batch_size * gradient_accumulation_steps
    
    # Memory optimization settings
    offload_experts = False          # Enable expert offloading to CPU
    quantize_offloaded = False       # Enable quantization during offloading
    gradient_checkpointing = True   # Enable gradient checkpointing
    mixed_precision = True          # Enable mixed precision training
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)
    
    # Initialize WandB
    wandb.init(project="ALLSTARS_multimodal_MoE_fusion_memory_efficient")
    config = {
        "num_classes": num_classes,
        "d_model_spectra": d_model_spectra,
        "d_model_gaia": d_model_gaia,
        "input_dim_spectra": input_dim_spectra,
        "input_dim_gaia": input_dim_gaia,
        "n_layers": n_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "lr": lr,
        "patience": patience,
        "num_epochs": num_epochs,
        "offload_experts": offload_experts,
        "quantize_offloaded": quantize_offloaded,
        "gradient_checkpointing": gradient_checkpointing,
        "mixed_precision": mixed_precision,
        "gradient_accumulation_steps": gradient_accumulation_steps
    }
    wandb.config.update(config)
    
    # Instantiate the memory-efficient MoE model
    model = StarClassifierWithMemoryEfficientMoE(
        d_model_spectra=d_model_spectra,
        d_model_gaia=d_model_gaia,
        num_classes=num_classes,
        input_dim_spectra=input_dim_spectra,
        input_dim_gaia=input_dim_gaia,
        n_layers=n_layers,
        num_experts=num_experts,
        k=top_k,
        use_cross_attention=True,
        n_cross_attn_heads=8,
        expansion_factor=2,  # Reduced from 4 to save memory
        dropout=0.1,
        aux_loss_weight=1e-2,
        offload_experts=offload_experts,
        quantize_offloaded=quantize_offloaded,
        gradient_checkpointing=gradient_checkpointing
    )
    
    # Print model information
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} GB parameters")
    print(f"Note: Due to expert offloading, the memory usage will be much lower than the parameter count suggests")
    
    # Calculate actual memory requirements (approximately)
    active_experts_ratio = top_k / num_experts
    effective_params = sum(p.numel() for p in model.parameters()) * active_experts_ratio
    print(f"Effective parameters during forward pass: ~{effective_params/1e6:.1f}M")
    
    # Compute memory footprint
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'Total model size: {size_all_mb:.3f} MB')
    print(f'Approximate VRAM usage during training: {size_all_mb * active_experts_ratio * 3:.3f} MB')
    
    # Print model architecture summary
    print(model)
    
    # Train using the memory-efficient training function
    trained_model = train_model_fusion_with_moe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        lr=lr,
        max_patience=patience,
        device=device,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        expert_offload_interval=5,
        memory_profiling=True,
        use_checkpoint_ensembling=True,
        checkpoint_interval=20
    )
    
    wandb.finish()
    
    # Save the trained model
    # First ensure all experts are on the same device
    if hasattr(trained_model, 'prepare_for_checkpoint'):
        trained_model.prepare_for_checkpoint(device)
    
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config
    }, "Models/model_fusion_memory_efficient_moe.pth")
    
    # Optional: Run inference test on a small batch to verify performance
    print("\nRunning inference test...")
    trained_model.eval()
    with torch.no_grad():
        # Get a small batch from test_loader
        X_spc, X_ga, y_true = next(iter(test_loader))
        X_spc, X_ga = X_spc.to(device), X_ga.to(device)
        
        # Benchmark inference time
        import time
        start_time = time.time()
        
        # During inference, we can disable offloading for speed
        outputs = trained_model(X_spc, X_ga, disable_offloading=True)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        inference_time = time.time() - start_time
        
        # Calculate predictions
        predicted = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
        y_true = y_true.numpy()
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, predicted)
        
        print(f"Inference time for batch of {len(X_spc)}: {inference_time:.4f}s")
        print(f"Test metrics on sample batch:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

# Function to load the model later for inference
def load_model_for_inference(model_path, device='cuda'):
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model with the same configuration
    model = StarClassifierWithMemoryEfficientMoE(
        d_model_spectra=config['d_model_spectra'],
        d_model_gaia=config['d_model_gaia'],
        num_classes=config['num_classes'],
        input_dim_spectra=config['input_dim_spectra'],
        input_dim_gaia=config['input_dim_gaia'],
        n_layers=config['n_layers'],
        num_experts=config['num_experts'],
        k=config['top_k'],
        use_cross_attention=True,
        n_cross_attn_heads=8,
        expansion_factor=2,
        dropout=0.1,
        aux_loss_weight=1e-2,
        # For inference, we can disable offloading for speed
        offload_experts=False,
        quantize_offloaded=False,
        gradient_checkpointing=False
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model