import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
import time
from contextlib import nullcontext
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from torch.utils.data import Dataset, DataLoader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model_fusion_with_moe(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs=100,
    lr=2.5e-3,
    max_patience=20,
    device='cuda',
    mixed_precision=True,
    gradient_accumulation_steps=1,
    expert_offload_interval=5,  # How often to offload experts to CPU
    memory_profiling=False,     # Whether to print memory usage statistics
    use_checkpoint_ensembling=True,  # Save multiple checkpoints and ensemble
    checkpoint_interval=10      # Save checkpoints every N epochs
):
    """
    Training procedure optimized for memory-efficient MoE models
    
    Args:
        model: The MoE model (should be a StarClassifierWithMemoryEfficientMoE instance)
        train_loader, val_loader, test_loader: DataLoaders for training, validation, test
        num_epochs: Maximum number of epochs
        lr: Learning rate
        max_patience: Early stopping patience
        device: Device to train on
        mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of batches to accumulate gradients for
        expert_offload_interval: How often to offload experts to CPU
        memory_profiling: Whether to print memory usage statistics
        use_checkpoint_ensembling: Whether to save multiple checkpoints for ensembling
        checkpoint_interval: Save checkpoints every N epochs
    """
    model = model.to(device)
    
    # Set up optimizer with weight decay for non-bias and non-LayerNorm parameters
    # This helps reduce overfitting while keeping memory usage lower
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_loader)//gradient_accumulation_steps
    )
    
    # Set up mixed precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None
    
    # Compute class weights for weighted loss
    all_labels = []
    for _, _, y_batch in train_loader:
        all_labels.extend(y_batch.cpu().numpy())
    
    class_weights = calculate_class_weights(np.array(all_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    best_val_loss = float('inf')
    patience = max_patience
    best_models = []  # For ensembling if use_checkpoint_ensembling is True
    
    # Function to log memory usage if memory_profiling is enabled
    def log_memory():
        if memory_profiling and torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, "
                  f"{torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Resample training data for balanced training
        train_loader.dataset.re_sample()
        
        # Recompute class weights if needed
        all_labels = []
        for _, _, y_batch in train_loader:
            all_labels.extend(y_batch.cpu().numpy())
        class_weights = calculate_class_weights(np.array(all_labels))
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        
        # --- Training ---
        model.train()
        train_loss, train_acc = 0.0, 0.0
        batch_count = 0
        
        log_memory()
        
        for batch_idx, (X_spc, X_ga, y_batch) in enumerate(train_loader):
            X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
            
            # Determine whether to offload experts based on the interval
            # This reduces memory pressure by periodically forcing offloading
            disable_offloading = (batch_idx % expert_offload_interval != 0)
            
            # Mixed precision context
            with torch.cuda.amp.autocast() if mixed_precision and torch.cuda.is_available() else nullcontext():
                # Forward pass with MoE model (returns logits, aux_loss)
                outputs, aux_loss = model(X_spc, X_ga, disable_offloading=disable_offloading)
                
                # Calculate primary loss
                main_loss = criterion(outputs, y_batch)
                
                # Combine losses (main loss + auxiliary MoE loss)
                loss = main_loss + aux_loss
                
                # Normalize loss for gradient accumulation
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
            
            # Backward pass with mixed precision if enabled
            if mixed_precision and torch.cuda.is_available():
                scaler.scale(loss).backward()
                
                # Update weights if accumulation steps reached
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Optional gradient clipping to prevent instability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss.backward()
                
                # Update weights if accumulation steps reached
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            train_loss += (main_loss.item() * gradient_accumulation_steps) * X_spc.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predicted == y_batch).float()
            train_acc += correct.mean(dim=1).sum().item()
            batch_count += X_spc.size(0)
            
            # Force CUDA synchronization and garbage collection periodically
            # This helps prevent memory leaks and fragmentation
            if (batch_idx + 1) % expert_offload_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log_memory()
            
            # Free up memory immediately
            del X_spc, X_ga, y_batch, outputs, loss
        
        # --- Validation ---
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        val_count = 0
        with torch.no_grad():
            for X_spc, X_ga, y_batch in val_loader:
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                
                # During validation, we can disable expert offloading for speed
                outputs = model(X_spc, X_ga, disable_offloading=True)
                
                # If training and using aux_loss, but during eval the model returns just logits
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_spc.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct = (predicted == y_batch).float()
                val_acc += correct.mean(dim=1).sum().item()
                val_count += X_spc.size(0)
                
                # Free memory
                del X_spc, X_ga, y_batch, outputs
        
        # --- Test metrics ---
        test_loss, test_acc = 0.0, 0.0
        y_true, y_pred = [], []
        test_count = 0
        with torch.no_grad():
            for X_spc, X_ga, y_batch in test_loader:
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                
                # During testing, we can disable expert offloading for speed
                outputs = model(X_spc, X_ga, disable_offloading=True)
                
                # Handle the case where model returns (outputs, aux_loss) during training
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_spc.size(0)
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct = (predicted == y_batch).float()
                test_acc += correct.mean(dim=1).sum().item()
                test_count += X_spc.size(0)

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
                # Free memory
                del X_spc, X_ga, y_batch, outputs
        
        # Compute multi-label metrics
        all_metrics = calculate_metrics(np.array(y_true), np.array(y_pred))
        
        # Calculate average loss and accuracy
        train_loss = train_loss / batch_count
        train_acc = train_acc / batch_count
        val_loss = val_loss / val_count
        val_acc = val_acc / val_count
        test_loss = test_loss / test_count
        test_acc = test_acc / test_count
        
        epoch_time = time.time() - epoch_start_time
        
        # Logging
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
              f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
              f"Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, "
              f"Test f1: {all_metrics['micro_f1']:.4f}, "
              f"LR: {get_lr(optimizer):.6f}")
        
        # Log to wandb if available
        try:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr": get_lr(optimizer),
                "epoch_time": epoch_time,
                **all_metrics
            })
        except ImportError:
            pass
        
        # Save checkpoint for potential ensembling
        if use_checkpoint_ensembling and (epoch + 1) % checkpoint_interval == 0:
            # Prepare model for saving by ensuring all experts are on same device
            if hasattr(model, 'prepare_for_checkpoint'):
                model.prepare_for_checkpoint(device)
            
            best_models.append({
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch
            })
            
            # Keep only the top 5 models
            best_models = sorted(best_models, key=lambda x: x['val_loss'])[:5]
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            
            # Save best model
            if hasattr(model, 'prepare_for_checkpoint'):
                model.prepare_for_checkpoint(device)
            
            best_model = model.state_dict()
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_metrics': all_metrics
            }, f'best_model_moe.pth')
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break
    
    # Load best model
    model.load_state_dict(best_model)
    
    # If using ensembling, create and evaluate an ensemble model
    if use_checkpoint_ensembling and len(best_models) > 1:
        print(f"Creating ensemble from {len(best_models)} checkpoints...")
        ensemble_predictions = []
        
        # For each checkpoint, predict on test set
        for i, checkpoint in enumerate(best_models):
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            
            test_predictions = []
            with torch.no_grad():
                for X_spc, X_ga, _ in test_loader:
                    X_spc, X_ga = X_spc.to(device), X_ga.to(device)
                    outputs = model(X_spc, X_ga, disable_offloading=True)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    probs = torch.sigmoid(outputs)
                    test_predictions.extend(probs.cpu().numpy())
            
            ensemble_predictions.append(np.array(test_predictions))
        
        # Average predictions across all models
        ensemble_probs = np.mean(ensemble_predictions, axis=0)
        ensemble_preds = (ensemble_probs > 0.5).astype(float)
        
        # Get true labels
        y_true = []
        for _, _, y_batch in test_loader:
            y_true.extend(y_batch.numpy())
        
        y_true = np.array(y_true)
        
        # Calculate ensemble metrics
        ensemble_metrics = calculate_metrics(y_true, ensemble_preds)
        print(f"Ensemble metrics:")
        for k, v in ensemble_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Load best single model back
        model.load_state_dict(best_model)
    
    return model

def calculate_class_weights(y):
    """Calculate class weights for imbalanced datasets"""
    if y.ndim > 1:
        class_counts = np.sum(y, axis=0)
    else:
        class_counts = np.bincount(y)
    
    total_samples = y.shape[0] if y.ndim > 1 else len(y)
    
    # Prevent division by zero
    class_counts = np.where(class_counts == 0, 1, class_counts)
    
    # Calculate weights inversely proportional to class frequency
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    # Clip weights to prevent extremely large values
    class_weights = np.clip(class_weights, 0.1, 10.0)
    
    return class_weights

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics for multi-label classification"""
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
    return metrics

# Example usage
def create_and_train_model(X_train_spectra, X_train_gaia, y_train,
                         X_val_spectra, X_val_gaia, y_val,
                         X_test_spectra, X_test_gaia, y_test,
                         batch_size=128, batch_limit=51,
                         num_experts=8, top_k=2, n_layers=6,
                         device='cuda'):
    """Example function showing how to create and train the model"""
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = MultiModalBalancedMultiLabelDataset(
        X_train_spectra, X_train_gaia, y_train, limit_per_label=batch_limit
    )
    val_dataset = MultiModalBalancedMultiLabelDataset(
        X_val_spectra, X_val_gaia, y_val, limit_per_label=batch_limit
    )
    test_dataset = MultiModalBalancedMultiLabelDataset(
        X_test_spectra, X_test_gaia, y_test, limit_per_label=batch_limit
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    
    model = StarClassifierWithMemoryEfficientMoE(
        d_model_spectra=1024,  # Reduced from 2048 to save memory
        d_model_gaia=512,      # Reduced from 2048 to save memory
        num_classes=y_train.shape[1],
        input_dim_spectra=X_train_spectra.shape[1],
        input_dim_gaia=X_train_gaia.shape[1],
        n_layers=n_layers,
        num_experts=num_experts,
        top_k=top_k,
        use_cross_attention=True,
        n_cross_attn_heads=8,
        expansion_factor=2,     # Reduced from 4 to save memory
        dropout=0.1,
        aux_loss_weight=1e-2,
        offload_experts=True,
        quantize_offloaded=True,
        gradient_checkpointing=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Active parameters during forward pass (since MoE only activates top_k experts)
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active_params_adjusted = active_params / (num_experts / top_k)
    print(f"Effective parameters (during forward pass): ~{active_params_adjusted:,.0f}")
    
    # Train model
    model = train_model_fusion_with_moe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=100,
        lr=1e-3,             # Reduced learning rate for stability
        max_patience=20,
        device=device,
        mixed_precision=True,
        gradient_accumulation_steps=2,  # Accumulate gradients to increase effective batch size
        expert_offload_interval=5,      # Offload experts to CPU every 5 batches
        memory_profiling=True,          # Print memory usage statistics
        use_checkpoint_ensembling=True, # Save multiple checkpoints for ensembling
        checkpoint_interval=10          # Save checkpoints every 10 epochs
    )
    
    return model

# Memory profiling function to help debug memory issues
def profile_memory_usage(model, X_spectra, X_gaia, y, device='cuda'):
    """Profile memory usage of a model during training and inference"""
    import torch
    import gc
    
    # Move data to device
    X_spectra = X_spectra.to(device)
    X_gaia = X_gaia.to(device)
    y = y.to(device)
    
    # Clear memory before starting
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print initial memory usage
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"Initial GPU memory: {initial_memory / 1e9:.2f} GB")
    
    # Profile forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(X_spectra, X_gaia)
        forward_time = time.time() - start_time
    
    forward_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"Forward pass memory: {forward_memory / 1e9:.2f} GB (+{(forward_memory - initial_memory) / 1e9:.2f} GB)")
    print(f"Forward pass time: {forward_time:.4f} s")
    
    # Profile training step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Clear gradients
    optimizer.zero_grad()
    
    # Forward pass
    start_time = time.time()
    outputs, aux_loss = model(X_spectra, X_gaia)
    main_loss = criterion(outputs, y)
    loss = main_loss + aux_loss
    
    # Backward pass
    loss.backward()
    backward_time = time.time() - start_time
    
    backward_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"Training step memory: {backward_memory / 1e9:.2f} GB (+{(backward_memory - forward_memory) / 1e9:.2f} GB)")
    print(f"Training step time: {backward_time:.4f} s")
    
    # Apply optimizer step
    optimizer.step()
    
    step_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"After optimizer step: {step_memory / 1e9:.2f} GB")
    
    # Clear memory
    del outputs, loss, optimizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"Final memory: {final_memory / 1e9:.2f} GB")
    
    # Print memory usage for each layer
    print("\nMemory usage by layer type:")
    layer_types = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0]
        if layer_type not in layer_types:
            layer_types[layer_type] = 0
        layer_types[layer_type] += param.numel() * 4  # 4 bytes per float32
    
    for layer_type, memory in layer_types.items():
        print(f"{layer_type}: {memory / 1e6:.1f} MB")


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import gc
import time

class ExpertManager:
    """
    Manages experts by offloading inactive experts to CPU
    and prefetching experts that will be needed soon
    """
    def __init__(self, experts, device_map=None):
        """
        Args:
            experts: List of expert modules
            device_map: Optional mapping of expert_idx -> device
                        If None, experts will be offloaded to CPU when inactive
        """
        self.experts = experts
        self.num_experts = len(experts)
        self.device_map = device_map or {i: 'cpu' for i in range(self.num_experts)}
        
        # Track which device each expert is currently on
        self.current_device = {i: 'cpu' for i in range(self.num_experts)}
        
        # Move all experts to CPU initially
        for i, expert in enumerate(self.experts):
            self.experts[i] = expert.to('cpu')
            
        # Keep track of expert usage frequency for potential prefetching
        self.usage_count = {i: 0 for i in range(self.num_experts)}
        
        # Cache for storing expert outputs to avoid recomputation
        self.output_cache = {}
            
    def get_expert(self, expert_idx, target_device):
        """
        Get the expert at expert_idx, moving it to target_device if needed
        Returns the expert and whether it was moved
        """
        if self.current_device[expert_idx] != target_device:
            # Move expert to target device
            self.experts[expert_idx] = self.experts[expert_idx].to(target_device)
            self.current_device[expert_idx] = target_device
            return self.experts[expert_idx], True
        return self.experts[expert_idx], False
    
    def release_expert(self, expert_idx, target_device='cpu'):
        """
        Release an expert, moving it back to CPU to save VRAM
        """
        if self.current_device[expert_idx] != target_device:
            self.experts[expert_idx] = self.experts[expert_idx].to(target_device)
            self.current_device[expert_idx] = target_device
            # Force garbage collection to immediately free GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def prefetch_experts(self, expert_indices, target_device):
        """
        Prefetch experts to target_device in a background thread
        """
        # In a real implementation, this would be done in a separate thread
        # For simplicity, we'll just move them directly
        for idx in expert_indices:
            if self.current_device[idx] != target_device:
                self.experts[idx] = self.experts[idx].to(target_device)
                self.current_device[idx] = target_device
                
    def update_usage_stats(self, expert_indices):
        """
        Update usage statistics for experts
        """
        for idx in expert_indices:
            self.usage_count[idx] += 1
            
    def clear_cache(self):
        """
        Clear the output cache
        """
        self.output_cache = {}
        
    def all_to_device(self, device):
        """
        Move all experts to a specific device (used before saving checkpoint)
        """
        for i in range(self.num_experts):
            if self.current_device[i] != device:
                self.experts[i] = self.experts[i].to(device)
                self.current_device[i] = device

class OffloadableExpertMLP(nn.Module):
    """
    Expert module designed for efficient offloading
    Uses parameter sharding to reduce peak memory usage
    """
    def __init__(self, dim, expansion_factor=4, dropout=0.):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        
        # Split the expert into separately offloadable components
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Flag to track if parameters are quantized
        self.is_quantized = False
        self.original_params = None
    
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
    def quantize(self, dtype=torch.float16):
        """
        Quantize parameters to reduce memory footprint during offloading
        """
        if not self.is_quantized:
            # Save original parameters
            self.original_params = {
                'fc1': self.fc1.weight.data.clone(),
                'fc2': self.fc2.weight.data.clone(),
                'fc1_bias': self.fc1.bias.data.clone() if self.fc1.bias is not None else None,
                'fc2_bias': self.fc2.bias.data.clone() if self.fc2.bias is not None else None
            }
            
            # Quantize weights
            self.fc1.weight.data = self.fc1.weight.data.to(dtype)
            self.fc2.weight.data = self.fc2.weight.data.to(dtype)
            
            if self.fc1.bias is not None:
                self.fc1.bias.data = self.fc1.bias.data.to(dtype)
            if self.fc2.bias is not None:
                self.fc2.bias.data = self.fc2.bias.data.to(dtype)
                
            self.is_quantized = True
    
    def dequantize(self):
        """
        Restore original parameters after offloading
        """
        if self.is_quantized and self.original_params is not None:
            self.fc1.weight.data = self.original_params['fc1']
            self.fc2.weight.data = self.original_params['fc2']
            
            if self.fc1.bias is not None and self.original_params['fc1_bias'] is not None:
                self.fc1.bias.data = self.original_params['fc1_bias']
            if self.fc2.bias is not None and self.original_params['fc2_bias'] is not None:
                self.fc2.bias.data = self.original_params['fc2_bias']
                
            self.original_params = None
            self.is_quantized = False

class MemoryEfficientMoE(nn.Module):
    """
    Memory-efficient Mixture of Experts with expert offloading
    """
    def __init__(self, dim, num_experts=8, k=2, capacity_factor=1.25, dropout=0., 
                 offload_experts=True, quantize_offloaded=True, use_expert_cache=True):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.offload_experts = offload_experts
        self.quantize_offloaded = quantize_offloaded
        self.use_expert_cache = use_expert_cache
        
        # Create router network
        self.router = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, num_experts),
        )
        
        # Create experts
        self.experts = nn.ModuleList([
            OffloadableExpertMLP(dim, dropout=dropout) for _ in range(num_experts)
        ])
        
        # Create expert manager if offloading is enabled
        if offload_experts:
            self.expert_manager = ExpertManager(self.experts)
        
        self.router_dropout = nn.Dropout(dropout)
        
        # Expert cache mapping (token_hash, expert_idx) -> output
        self.expert_output_cache = {}
        
    def forward(self, x, disable_offloading=False):
        """
        x: [batch_size, seq_len, dim] or [batch_size, dim] if seq_len=1 and squeezed
        """
        original_shape = x.shape
        batch_size = x.shape[0]
        seq_len = 1 if len(original_shape) == 2 else original_shape[1]
        dim = original_shape[-1]
        
        device = x.device
        
        # Reshape to [batch_size * seq_len, dim] for routing
        x_flat = x.reshape(-1, dim)
        
        # Compute routing probabilities
        routing_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        
        if self.training:
            routing_logits = self.router_dropout(routing_logits)
        
        # Calculate routing probabilities
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Calculate the capacity of each expert
        capacity = int(self.capacity_factor * batch_size * seq_len / self.num_experts)
        
        # Get top-k experts per token
        routing_weights, routing_indices = torch.topk(routing_probs, self.k, dim=-1)
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Create output tensor
        combined_output = torch.zeros_like(x_flat)
        
        # Track load balancing
        load = torch.zeros(self.num_experts, device=device)
        
        # Cache for token hashes if using expert caching
        if self.use_expert_cache:
            # Create a hash for each token to use as cache key
            # In a real implementation, you'd use a more sophisticated hashing method
            token_hashes = torch.sum(x_flat * torch.arange(1, dim + 1, device=device), dim=-1)
        
        # Process tokens through their assigned experts
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (routing_indices == expert_idx).any(dim=-1)
            
            if not expert_mask.any():
                continue
                
            # Get indices of tokens assigned to this expert
            token_indices = torch.where(expert_mask)[0]
            
            # Get tokens assigned to this expert
            expert_inputs = x_flat[token_indices]
            
            # Get the position and weight for each token
            token_positions = torch.where(routing_indices == expert_idx)[0]
            k_positions = torch.where(routing_indices == expert_idx)[1]
            token_weights = routing_weights[token_positions, k_positions].unsqueeze(-1)
            
            # Update load balancing stats
            load[expert_idx] = len(token_indices) / (batch_size * seq_len)
            
            # Process through expert
            if self.offload_experts and not disable_offloading:
                # Check cache first if enabled
                if self.use_expert_cache:
                    cache_hits = 0
                    expert_outputs = torch.zeros_like(expert_inputs)
                    
                    for i, idx in enumerate(token_indices):
                        token_hash = token_hashes[idx].item()
                        cache_key = (token_hash, expert_idx)
                        
                        if cache_key in self.expert_output_cache:
                            expert_outputs[i] = self.expert_output_cache[cache_key]
                            cache_hits += 1
                        else:
                            # Process token through expert
                            expert, was_moved = self.expert_manager.get_expert(expert_idx, device)
                            
                            # Quantize when moving to GPU to save memory during transfer
                            if was_moved and self.quantize_offloaded:
                                expert.quantize()
                                
                            # Process single token
                            token_output = expert(expert_inputs[i:i+1])
                            expert_outputs[i:i+1] = token_output
                            
                            # Cache the result
                            self.expert_output_cache[cache_key] = token_output.squeeze(0).detach()
                            
                            # Dequantize if necessary
                            if was_moved and self.quantize_offloaded:
                                expert.dequantize()
                                
                            # Release expert back to CPU to save VRAM
                            self.expert_manager.release_expert(expert_idx)
                else:
                    # Get the expert, moving it to the current device if needed
                    expert, was_moved = self.expert_manager.get_expert(expert_idx, device)
                    
                    # Quantize when moving to GPU to save memory during transfer
                    if was_moved and self.quantize_offloaded:
                        expert.quantize()
                        
                    # Process tokens through expert
                    expert_outputs = expert(expert_inputs)
                    
                    # Dequantize if necessary
                    if was_moved and self.quantize_offloaded:
                        expert.dequantize()
                        
                    # Release expert back to CPU to save VRAM
                    self.expert_manager.release_expert(expert_idx)
            else:
                # Standard processing without offloading
                expert_outputs = self.experts[expert_idx](expert_inputs)
            
            # Combine outputs weighted by routing weights
            combined_output[token_indices] += token_weights * expert_outputs
        
        # Reshape output back to original shape
        output = combined_output.reshape(original_shape)
        
        # Calculate load balancing loss
        # Ideal load is 1/num_experts for each expert
        ideal_load = torch.ones_like(load) / self.num_experts
        load_balancing_loss = torch.sum((load - ideal_load) ** 2) * self.num_experts
        
        # Clear cache if it's getting too large
        if self.use_expert_cache and len(self.expert_output_cache) > 10000:
            self.expert_output_cache = {}
            
        return output, load_balancing_loss
    
    def prepare_for_checkpoint(self, device='cuda'):
        """
        Prepare model for checkpointing by moving all experts to the same device
        """
        if self.offload_experts:
            self.expert_manager.all_to_device(device)

class MemoryEfficientMoEBlock(nn.Module):
    """
    Memory-efficient MoE block for high-dimensional processing
    """
    def __init__(
        self, 
        dim, 
        num_experts=8, 
        k=2, 
        expansion_factor=4, 
        dropout=0., 
        drop_path=0.,
        offload_experts=True,
        quantize_offloaded=True
    ):
        super().__init__()
        
        # Normalization and gating
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, dim * 2)  # For gating
        
        # Memory-efficient MoE
        self.moe = MemoryEfficientMoE(
            dim=dim, 
            num_experts=num_experts, 
            k=k,
            dropout=dropout,
            offload_experts=offload_experts,
            quantize_offloaded=quantize_offloaded
        )
        
        # Output projection
        self.fc2 = nn.Linear(dim, dim)
        
        # Regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, disable_offloading=False):
        # Input shape: [B, seq_len, dim] or [B, dim] if seq_len=1 and squeezed
        shortcut = x
        
        # Handle sequence dimension consistently
        orig_shape = x.shape
        squeezed = False
        
        if len(orig_shape) == 3 and orig_shape[1] == 1:
            # Squeeze out sequence dimension for efficiency with seq_len=1
            x = x.squeeze(1)  # [B, dim]
            squeezed = True
        
        # Apply first normalization and split for gating
        x = self.norm1(x)
        x = self.fc1(x)
        gate, content = torch.chunk(x, 2, dim=-1)
        gate = F.gelu(gate)
        
        # Process through Memory-efficient MoE
        moe_output, load_balancing_loss = self.moe(content, disable_offloading=disable_offloading)
        
        # Apply gating and output projection
        x = self.fc2(gate * moe_output)
        
        # Apply drop path
        x = self.drop_path(x)
        
        # Restore original shape if needed
        if squeezed:
            x = x.unsqueeze(1)  # [B, 1, dim]
        
        # Add residual connection
        output = x + shortcut
        
        return output, load_balancing_loss

class MemoryEfficientMoEProcessor(nn.Module):
    """
    High-dimensional sequence processor with Memory-efficient MoE
    """
    def __init__(
        self, 
        dim, 
        depth=1, 
        num_experts=8, 
        k=2, 
        expansion_factor=4, 
        dropout=0., 
        drop_path=0.,
        offload_experts=True,
        quantize_offloaded=True,
        gradient_checkpointing=False
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            MemoryEfficientMoEBlock(
                dim=dim, 
                num_experts=num_experts, 
                k=k, 
                expansion_factor=expansion_factor, 
                dropout=dropout, 
                drop_path=drop_path if i > 0 else 0.,
                offload_experts=offload_experts,
                quantize_offloaded=quantize_offloaded
            ) for i in range(depth)
        ])
        
        self.gradient_checkpointing = gradient_checkpointing
    
    def forward(self, x, disable_offloading=False):
        total_load_balancing_loss = 0.0
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, disable_offloading=disable_offloading)
                    return custom_forward
                
                x_out, load_balancing_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), x
                )
            else:
                x_out, load_balancing_loss = block(x, disable_offloading=disable_offloading)
                
            x = x_out
            total_load_balancing_loss += load_balancing_loss
            
        return x, total_load_balancing_loss

class StarClassifierWithMemoryEfficientMoE(nn.Module):
    def __init__(
        self,
        d_model_spectra,
        d_model_gaia,
        num_classes,
        input_dim_spectra,
        input_dim_gaia,
        n_layers=6,
        num_experts=8,
        k=2,
        use_cross_attention=True,
        n_cross_attn_heads=8,
        expansion_factor=4,
        dropout=0.1,
        aux_loss_weight=1e-2,
        offload_experts=True,
        quantize_offloaded=True,
        gradient_checkpointing=False
    ):
        super().__init__()
        
        self.aux_loss_weight = aux_loss_weight
        self.gradient_checkpointing = gradient_checkpointing
        
        # --- Memory-efficient MoE processor for spectra ---
        self.processor_spectra = MemoryEfficientMoEProcessor(
            dim=d_model_spectra,
            depth=n_layers,
            num_experts=num_experts,
            k=k,
            expansion_factor=expansion_factor,
            dropout=dropout,
            drop_path=0.1,
            offload_experts=offload_experts,
            quantize_offloaded=quantize_offloaded,
            gradient_checkpointing=gradient_checkpointing
        )
        self.input_proj_spectra = nn.Linear(input_dim_spectra, d_model_spectra)
        
        # --- Memory-efficient MoE processor for gaia ---
        self.processor_gaia = MemoryEfficientMoEProcessor(
            dim=d_model_gaia,
            depth=n_layers,
            num_experts=num_experts,
            k=k,
            expansion_factor=expansion_factor,
            dropout=dropout,
            drop_path=0.1,
            offload_experts=offload_experts,
            quantize_offloaded=quantize_offloaded,
            gradient_checkpointing=gradient_checkpointing
        )
        self.input_proj_gaia = nn.Linear(input_dim_gaia, d_model_gaia)
        
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
        
    def forward(self, x_spectra, x_gaia, disable_offloading=False):
        """
        x_spectra : (batch_size, input_dim_spectra) or (batch_size, seq_len_spectra, input_dim_spectra)
        x_gaia    : (batch_size, input_dim_gaia) or (batch_size, seq_len_gaia, input_dim_gaia)
        """
        # Initialize auxiliary loss
        total_aux_loss = 0.0
        
        # Project inputs to high-dimensional space
        if len(x_spectra.shape) == 2:
            x_spectra = self.input_proj_spectra(x_spectra)  # (B, d_model_spectra)
            x_spectra = x_spectra.unsqueeze(1)              # (B, 1, d_model_spectra)
        else:
            x_spectra = self.input_proj_spectra(x_spectra)  # (B, seq_len, d_model_spectra)
        
        if len(x_gaia.shape) == 2:
            x_gaia = self.input_proj_gaia(x_gaia)           # (B, d_model_gaia)
            x_gaia = x_gaia.unsqueeze(1)                    # (B, 1, d_model_gaia)
        else:
            x_gaia = self.input_proj_gaia(x_gaia)           # (B, seq_len, d_model_gaia)
            
        # Process through Memory-efficient MoE layers
        x_spectra, spectra_aux_loss = self.processor_spectra(x_spectra, disable_offloading=disable_offloading)
        x_gaia, gaia_aux_loss = self.processor_gaia(x_gaia, disable_offloading=disable_offloading)
        
        # Accumulate auxiliary losses
        total_aux_loss += spectra_aux_loss + gaia_aux_loss
        
        # Optionally, use cross-attention for modality fusion
        if self.use_cross_attention:
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing for cross-attention as well
                def create_custom_forward_spectra(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                def create_custom_forward_gaia(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                x_spectra = torch.utils.checkpoint.checkpoint(
                    create_custom_forward_spectra(self.cross_attn_block_spectra),
                    x_spectra, x_gaia
                )
                
                x_gaia = torch.utils.checkpoint.checkpoint(
                    create_custom_forward_gaia(self.cross_attn_block_gaia),
                    x_gaia, x_spectra
                )
            else:
                # Standard forward pass
                x_spectra = self.cross_attn_block_spectra(x_spectra, x_gaia)
                x_gaia = self.cross_attn_block_gaia(x_gaia, x_spectra)
            
        # Pool across sequence dimension
        x_spectra = x_spectra.mean(dim=1)  # (B, d_model_spectra)
        x_gaia = x_gaia.mean(dim=1)        # (B, d_model_gaia)
        
        # Concatenate for late fusion
        x_fused = torch.cat([x_spectra, x_gaia], dim=-1)  # (B, d_model_spectra + d_model_gaia)
        
        # Final classification
        logits = self.classifier(x_fused)  # (B, num_classes)
        
        # During training, return logits and auxiliary loss
        # During inference, just return logits
        if self.training:
            return logits, self.aux_loss_weight * total_aux_loss
        else:
            return logits
    
    def prepare_for_checkpoint(self, device='cuda'):
        """
        Prepare the model for checkpointing by moving all experts to the same device
        Fixed to avoid recursion errors
        """
        # Only process direct children that have prepare_for_checkpoint method
        for name, module in self.named_children():
            if hasattr(module, 'prepare_for_checkpoint'):
                module.prepare_for_checkpoint(device=device)

# Example training function with memory optimization
def train_with_memory_optimizations(
    model, 
    dataloader, 
    optimizer, 
    criterion, 
    device, 
    epochs=1, 
    gradient_accumulation_steps=1,
    mixed_precision=True
):
    model.train()
    
    # Setup mixed precision training if enabled
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (spectra, gaia, labels) in enumerate(dataloader):
            # Move data to device
            spectra = spectra.to(device)
            gaia = gaia.to(device)
            labels = labels.to(device)
            
            # Mixed precision context
            with torch.cuda.amp.autocast() if mixed_precision else nullcontext():
                # Forward pass
                logits, aux_loss = model(spectra, gaia)
                
                # Calculate loss
                classification_loss = criterion(logits, labels)
                loss = classification_loss + aux_loss
                
                # Normalize loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with mixed precision if enabled
            if mixed_precision:
                scaler.scale(loss).backward()
                
                # Update weights if accumulation steps reached
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                # Update weights if accumulation steps reached
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            batch_count += 1
            
            # Free up memory
            del spectra, gaia, labels, logits, loss
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Prepare model for saving (moving all experts to the same device)
    model.prepare_for_checkpoint()
    
    return model

# Utility context manager for managing temporary device placement
class nullcontext:
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

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
def calculate_class_weights(y):
    if y.ndim > 1:  
        class_counts = np.sum(y, axis=0)  
    else:
        class_counts = np.bincount(y)

    total_samples = y.shape[0] if y.ndim > 1 else len(y)
    class_counts = np.where(class_counts == 0, 1, class_counts)  # Prevent division by zero
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    return class_weights

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
            self.X_gaia[index],     # gaia features
            self.y[index],          # multi-hot labels
        )
def calculate_class_weights(y):
    if y.ndim > 1:  
        class_counts = np.sum(y, axis=0)  
    else:
        class_counts = np.bincount(y)

    total_samples = y.shape[0] if y.ndim > 1 else len(y)
    class_counts = np.where(class_counts == 0, 1, class_counts)  # Prevent division by zero
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    return class_weights
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
    
    # Check if there are at least two classes present in y_true
    #if len(np.unique(y_true)) > 1:
        #metrics["roc_auc"] = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
    #else:
       # metrics["roc_auc"] = None  # or you can set it to a default value or message
    
    return metrics
