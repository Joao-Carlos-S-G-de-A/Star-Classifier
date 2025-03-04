import os
import gc
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2



def train_model_fusion(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs=100,
    lr=2.5e-3,
    max_patience=20,
    device='cuda',
    batch_accumulation=1,  # Gradient accumulation parameter
    use_reentrant=False    # Checkpoint parameter to address warning
):
    """
    Train a fusion model with gradient accumulation support.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_epochs: Maximum number of epochs to train
        lr: Learning rate
        max_patience: Early stopping patience
        device: Device to train on
        batch_accumulation: Number of batches to accumulate before updating weights
        use_reentrant: Whether to use reentrant checkpointing
        
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Calculate effective steps for OneCycleLR to prevent "stepped too many times" error
    effective_steps = len(train_loader) // batch_accumulation
    if len(train_loader) % batch_accumulation != 0:
        effective_steps += 1
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=effective_steps  # This is the key fix
    )
    
    # Calculate class weights
    all_labels = []
    for _, _, y_batch in train_loader:
        all_labels.extend(y_batch.cpu().numpy())
    
    class_weights = calculate_class_weights(np.array(all_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    best_val_loss = float('inf')
    patience = max_patience
    best_model = None
    
    for epoch in range(num_epochs):
        # Resample training data
        train_loader.dataset.re_sample()
        
        # Training with gradient accumulation
        model.train()
        train_loss, train_acc = 0.0, 0.0
        batch_count = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for i, (X_spc, X_ga, y_batch) in enumerate(train_loader):
            X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_spc, X_ga)
            loss = criterion(outputs, y_batch) / batch_accumulation  # Scale loss
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            train_loss += loss.item() * batch_accumulation * X_spc.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predicted == y_batch).float()
            train_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
            batch_count += X_spc.size(0)
            
            # Step optimizer and scheduler only after accumulating gradients
            if (i + 1) % batch_accumulation == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()  # Safe because steps_per_epoch is fixed
                optimizer.zero_grad()
        
        # Calculate average metrics
        train_loss /= batch_count
        train_acc /= batch_count
        
        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        batch_count = 0
        
        with torch.no_grad():
            for X_spc, X_ga, y_batch in val_loader:
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                
                outputs = model(X_spc, X_ga)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_spc.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct = (predicted == y_batch).float()
                val_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
                batch_count += X_spc.size(0)
        
        val_loss /= batch_count
        val_acc /= batch_count
        
        # Print progress
        #print(f'Epoch {epoch+1}/{num_epochs} - ' f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, 'f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            best_model = model.state_dict().copy()
            #print(f"New best model with validation loss: {val_loss:.4f}")
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    model.eval()
    
    # Test evaluation
    test_loss, test_acc = 0.0, 0.0
    all_preds, all_labels = [], []
    batch_count = 0
    
    with torch.no_grad():
        for X_spc, X_ga, y_batch in test_loader:
            X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
            
            outputs = model(X_spc, X_ga)
            loss = criterion(outputs, y_batch)
            
            test_loss += loss.item() * X_spc.size(0)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct = (predicted == y_batch).float()
            test_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
            
            batch_count += X_spc.size(0)
            
            # Collect predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    test_loss /= batch_count
    test_acc /= batch_count
    
    # Calculate detailed metrics
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print("Test Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model

class MemoryEfficientMamba(nn.Module):
    """
    Memory-efficient wrapper for Mamba2 with gradient checkpointing and stride alignment fixes.
    """
    def __init__(self, mamba, use_checkpoint=True):
        super().__init__()
        self.mamba = mamba
        self.use_checkpoint = use_checkpoint
    
    def forward(self, x):
        # Need to fix strides to be multiples of 8
        # Most reliable way: permute to channels_first, make contiguous, then permute back
        batch_size, seq_len, d_model = x.shape
        
        # Pad sequence dimension to be a multiple of 8 if needed
        # This helps with stride alignment on some devices
        pad_seq = 0
        if seq_len % 8 != 0:
            pad_seq = 8 - (seq_len % 8)
            # Pad sequence dimension with zeros
            x = nn.functional.pad(x, (0, 0, 0, pad_seq, 0, 0))
        
        # Ensure strides are properly aligned by reshaping and making contiguous
        # Note: x.transpose(1, 2) makes the former feature dimension the seq dimension which 
        # can align memory better
        x = x.transpose(1, 2).contiguous().transpose(1, 2).contiguous()
        
        # Verify strides are now divisible by 8
        if x.stride(0) % 8 != 0 or x.stride(2) % 8 != 0:
            # If still not aligned, need a more aggressive approach: clone the tensor
            x = x.clone()
        
        # Pass through the model with optional checkpointing
        if self.use_checkpoint and self.training:
            result = checkpoint(self.mamba, x, use_reentrant=False)
        else:
            result = self.mamba(x)
        
        # Remove padding if added
        if pad_seq > 0:
            result = result[:, :seq_len, :]
            
        return result
class EnsembleModelWithUncertainty:
    """
    Ensemble of models for uncertainty quantification.
    Takes a model class and creates multiple instances with different seeds.
    """
    def __init__(
        self, 
        model_class, 
        model_params, 
        num_models=5,
        device='cuda',
        uncertainty_method='entropy'
    ):
        """
        Initialize ensemble of models.
        
        Args:
            model_class: Class of the model to ensemble
            model_params: Parameters to pass to model initialization
            num_models: Number of models in the ensemble
            device: Device to use for computation
            uncertainty_method: Method to calculate uncertainty ('variance', 'entropy', etc.)
        """
        self.model_class = model_class
        self.model_params = model_params
        self.num_models = num_models
        self.device = device
        self.uncertainty_method = uncertainty_method
        self.models = []
        
        # Initialize models with different seeds
        for i in range(num_models):
            # Set seed for reproducibility with variation
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)
            
            # Create model instance
            model = self.model_class(**self.model_params).to(device)
            self.models.append(model)
        
        print(f"Created ensemble with {num_models} models")

    def train(
        self,
        train_loader,
        val_loader,
        test_loader,
        num_epochs=100,
        lr=2.5e-3,
        max_patience=20,
        batch_accumulation=1,
        use_wandb=True,
        wandb_project="star-classifier",
        wandb_config=None
    ):
        """
        Train all models in the ensemble.
        
        Args:
            Same as train_model_fusion, but trains all models in sequence
            Use wandb: Whether to use Weights & Biases for logging
            wandb_project: Project name for Weights & Biases
            wandb_config: Configuration to log in Weights & Biases
        """
        if use_wandb and wandb_config is None:
            wandb_config = {
                "num_models": self.num_models,
                "model_class": self.model_class.__name__,
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "batch_accumulation": batch_accumulation,
                "uncertainty_method": self.uncertainty_method,
                **self.model_params
            }
        
        # Train each model
        all_best_models = []
        
        for model_idx, model in enumerate(self.models):
            print(f"\n===== Training Model {model_idx + 1}/{self.num_models} =====")
            
            # Initialize wandb for this model
            if use_wandb:
                run_name = f"model_{model_idx + 1}_of_{self.num_models}"
                wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=wandb_config,
                    reinit=True,
                    group="ensemble"
                )
                
                # Log model architecture
                wandb.watch(model)
            
            # Train the model with the ensemble-specific training function
            trained_model, best_metrics = self._train_single_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                lr=lr,
                max_patience=max_patience,
                batch_accumulation=batch_accumulation,
                model_idx=model_idx
            )
            
            # Store trained model
            self.models[model_idx] = trained_model
            all_best_models.append(best_metrics)
            
            # Finish wandb run
            if use_wandb:
                wandb.finish()
            
            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()
        
        # Initialize final wandb run for ensemble metrics
        if use_wandb:
            run_name = f"ensemble_of_{self.num_models}_models"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=wandb_config,
                reinit=True,
                group="ensemble"
            )
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_ensemble_metrics(all_best_models)
            
            # Log ensemble metrics
            wandb.log(ensemble_metrics)
            wandb.finish()
        
        return self.models
    
    def _train_single_model(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        num_epochs=100,
        lr=2.5e-3,
        max_patience=20,
        batch_accumulation=1,
        model_idx=0
    ):
        """Train a single model in the ensemble."""
        device = self.device
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        # Calculate effective steps for OneCycleLR
        effective_steps = len(train_loader) // batch_accumulation
        if len(train_loader) % batch_accumulation != 0:
            effective_steps += 1
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=effective_steps
        )
        
        # Calculate class weights
        all_labels = []
        for _, _, y_batch in train_loader:
            all_labels.extend(y_batch.cpu().numpy())
        
        class_weights = calculate_class_weights(np.array(all_labels))
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        
        best_val_loss = float('inf')
        patience = max_patience
        best_model = None
        best_metrics = {}
        
        for epoch in range(num_epochs):
            # Resample training data
            train_loader.dataset.re_sample()
            
            # Training with gradient accumulation
            model.train()
            train_loss, train_acc = 0.0, 0.0
            batch_count = 0
            optimizer.zero_grad()  # Zero gradients at start of epoch
            
            for i, (X_spc, X_ga, y_batch) in enumerate(train_loader):
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_spc, X_ga)
                loss = criterion(outputs, y_batch) / batch_accumulation  # Scale loss
                
                # Backward pass
                loss.backward()
                
                # Update metrics
                train_loss += loss.item() * batch_accumulation * X_spc.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct = (predicted == y_batch).float()
                train_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
                batch_count += X_spc.size(0)
                
                # Step optimizer and scheduler only after accumulating gradients
                if (i + 1) % batch_accumulation == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    scheduler.step()  # Safe because steps_per_epoch is fixed
                    optimizer.zero_grad()
            
            # Calculate average metrics
            train_loss /= batch_count
            train_acc /= batch_count
            
            # Validation
            model.eval()
            val_loss, val_acc = 0.0, 0.0
            batch_count = 0
            
            with torch.no_grad():
                for X_spc, X_ga, y_batch in val_loader:
                    X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                    
                    outputs = model(X_spc, X_ga)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item() * X_spc.size(0)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    correct = (predicted == y_batch).float()
                    val_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
                    batch_count += X_spc.size(0)
            
            val_loss /= batch_count
            val_acc /= batch_count
            
            # Print progress
            #print(f'Model {model_idx+1} - Epoch {epoch+1}/{num_epochs} - 'f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, 'f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "hamming_loss": hamming_loss(np.array(all_labels), np.array(all_preds)),
                    "precision": precision_score(np.array(all_labels), np.array(all_preds), average='samples'),
                    "recall": recall_score(np.array(all_labels), np.array(all_preds), average='samples'),
                    "f1": f1_score(np.array(all_labels), np.array(all_preds), average='samples'),
                    "macro_f1": f1_score(np.array(all_labels), np.array(all_preds), average='macro'),
                    "micro_f1": f1_score(np.array(all_labels), np.array(all_preds), average='micro'),
                    "macro_precision": precision_score(np.array(all_labels), np.array(all_preds), average='macro'),
                    "micro_precision": precision_score(np.array(all_labels), np.array(all_preds), average='micro'),
                    "macro_recall": recall_score(np.array(all_labels), np.array(all_preds), average='macro'),
                    "micro_recall": recall_score(np.array(all_labels), np.array(all_preds), average='micro')
                })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = max_patience
                best_model = model.state_dict().copy()
                #print(f"New best model with validation loss: {val_loss:.4f}")
            else:
                patience -= 1
                if patience == 0:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and evaluate on test set
        model.load_state_dict(best_model)
        model.eval()
        
        # Test evaluation
        test_loss, test_acc = 0.0, 0.0
        all_preds, all_labels = [], []
        batch_count = 0
        
        with torch.no_grad():
            for X_spc, X_ga, y_batch in test_loader:
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                
                outputs = model(X_spc, X_ga)
                loss = criterion(outputs, y_batch)
                
                test_loss += loss.item() * X_spc.size(0)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                correct = (predicted == y_batch).float()
                test_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
                
                batch_count += X_spc.size(0)
                
                # Collect predictions and labels for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        test_loss /= batch_count
        test_acc /= batch_count
        
        # Calculate detailed metrics
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        print(f"\nModel {model_idx+1} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print("Test Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Store metrics for ensemble calculation
        best_metrics = {
            "model_idx": model_idx,
            "test_loss": test_loss,
            "test_acc": test_acc,
            **metrics
        }
        
        # Log final test metrics to wandb
        if wandb.run is not None:
            wandb.log({
                "test_loss": test_loss,
                "test_acc": test_acc,
                **metrics
            })
        
        return model, best_metrics
    
    def _train_single_model(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        num_epochs=100,
        lr=2.5e-3,
        max_patience=20,
        batch_accumulation=1,
        model_idx=0
    ):
        """Train a single model in the ensemble."""
        device = self.device
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
        # Calculate effective steps for OneCycleLR
        effective_steps = len(train_loader) // batch_accumulation
        if len(train_loader) % batch_accumulation != 0:
            effective_steps += 1
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=effective_steps
        )
        
        # Calculate class weights
        all_labels = []
        for _, _, y_batch in train_loader:
            all_labels.extend(y_batch.cpu().numpy())
        
        class_weights = calculate_class_weights(np.array(all_labels))
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        
        best_val_loss = float('inf')
        patience = max_patience
        best_model = None
        best_metrics = {}
        
        for epoch in range(num_epochs):
            # Resample training data
            train_loader.dataset.re_sample()
            
            # Training with gradient accumulation
            model.train()
            train_loss, train_acc = 0.0, 0.0
            batch_count = 0
            optimizer.zero_grad()  # Zero gradients at start of epoch
            
            for i, (X_spc, X_ga, y_batch) in enumerate(train_loader):
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_spc, X_ga)
                loss = criterion(outputs, y_batch) / batch_accumulation  # Scale loss
                
                # Backward pass
                loss.backward()
                
                # Update metrics
                train_loss += loss.item() * batch_accumulation * X_spc.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct = (predicted == y_batch).float()
                train_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
                batch_count += X_spc.size(0)
                
                # Step optimizer and scheduler only after accumulating gradients
                if (i + 1) % batch_accumulation == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    scheduler.step()  # Safe because steps_per_epoch is fixed
                    optimizer.zero_grad()
            
            # Calculate average metrics
            train_loss /= batch_count
            train_acc /= batch_count
            
            # Validation
            model.eval()
            val_loss, val_acc = 0.0, 0.0
            batch_count = 0
            val_preds, val_labels = [], []  # Added to collect validation predictions
            
            with torch.no_grad():
                for X_spc, X_ga, y_batch in val_loader:
                    X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                    
                    outputs = model(X_spc, X_ga)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item() * X_spc.size(0)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    correct = (predicted == y_batch).float()
                    val_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
                    batch_count += X_spc.size(0)
                    
                    # Collect predictions and labels for metrics
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(y_batch.cpu().numpy())
            
            val_loss /= batch_count
            val_acc /= batch_count
            
            # Print progress
            #print(f'Model {model_idx+1} - Epoch {epoch+1}/{num_epochs} - 'f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, 'f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Log to wandb
            if wandb.run is not None:
                # Calculate metrics using validation predictions
                val_metrics = calculate_metrics(np.array(val_labels), np.array(val_preds))
                
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    **val_metrics  # Include all validation metrics
                })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = max_patience
                best_model = model.state_dict().copy()
                #print(f"New best model with validation loss: {val_loss:.4f}")
            else:
                patience -= 1
                if patience == 0:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and evaluate on test set
        model.load_state_dict(best_model)
        model.eval()
        
        # Test evaluation
        test_loss, test_acc = 0.0, 0.0
        all_preds, all_labels = [], []
        batch_count = 0
        
        with torch.no_grad():
            for X_spc, X_ga, y_batch in test_loader:
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                
                outputs = model(X_spc, X_ga)
                loss = criterion(outputs, y_batch)
                
                test_loss += loss.item() * X_spc.size(0)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                correct = (predicted == y_batch).float()
                test_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
                
                batch_count += X_spc.size(0)
                
                # Collect predictions and labels for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        test_loss /= batch_count
        test_acc /= batch_count
        
        # Calculate detailed metrics
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        print(f"\nModel {model_idx+1} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print("Test Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Store metrics for ensemble calculation
        best_metrics = {
            "model_idx": model_idx,
            "test_loss": test_loss,
            "test_acc": test_acc,
            **metrics
        }
        
        # Log final test metrics to wandb
        if wandb.run is not None:
            wandb.log({
                "test_loss": test_loss,
                "test_acc": test_acc,
                **metrics
            })
        
        return model, best_metrics
    
    def _calculate_ensemble_metrics(self, all_metrics):
        """Calculate ensemble metrics based on individual model metrics."""
        ensemble_metrics = {}
        
        # Calculate average and standard deviation for each metric
        for key in all_metrics[0].keys():
            if key == "model_idx":
                continue
                
            values = [m[key] for m in all_metrics]
            ensemble_metrics[f"ensemble_mean_{key}"] = np.mean(values)
            ensemble_metrics[f"ensemble_std_{key}"] = np.std(values)
            ensemble_metrics[f"ensemble_min_{key}"] = np.min(values)
            ensemble_metrics[f"ensemble_max_{key}"] = np.max(values)
        
        return ensemble_metrics
    
    def predict(self, X_spectra, X_gaia, return_uncertainty=True):
        """
        Make predictions with the ensemble and optionally return uncertainty.
        
        Args:
            X_spectra: Spectral features
            X_gaia: Gaia features
            return_uncertainty: Whether to return uncertainty metrics
            
        Returns:
            predictions: Mean predictions across all models
            uncertainty: Uncertainty metrics if requested
        """
        # Move inputs to device
        X_spectra = X_spectra.to(self.device)
        X_gaia = X_gaia.to(self.device)
        
        # Get predictions from all models
        all_probs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                # Get raw logits
                logits = model(X_spectra, X_gaia)
                # Convert to probabilities
                probs = torch.sigmoid(logits)
                all_probs.append(probs)
        
        # Stack predictions [num_models, batch_size, num_classes]
        all_probs = torch.stack(all_probs)
        
        # Calculate mean predictions
        mean_probs = all_probs.mean(dim=0)
        
        # Convert mean probabilities to binary predictions
        predictions = (mean_probs > 0.5).float()
        
        if not return_uncertainty:
            return predictions
        
        # Calculate uncertainty metrics
        uncertainty = {}
        
        if self.uncertainty_method == 'variance':
            # Variance of probabilities across models
            uncertainty['variance'] = all_probs.var(dim=0)
            uncertainty['std'] = all_probs.std(dim=0)
            
            # Average uncertainty per sample
            uncertainty['mean_variance'] = uncertainty['variance'].mean(dim=1)
            uncertainty['mean_std'] = uncertainty['std'].mean(dim=1)
            
            # Model disagreement (how many models disagree with the majority)
            binary_preds = (all_probs > 0.5).float()
            majority_vote = (binary_preds.sum(dim=0) > (self.num_models / 2)).float()
            disagreement = torch.abs(binary_preds - majority_vote.unsqueeze(0)).sum(dim=0) / self.num_models
            uncertainty['disagreement'] = disagreement
            uncertainty['mean_disagreement'] = disagreement.mean(dim=1)
            
        elif self.uncertainty_method == 'entropy':
            # Mean probabilities across models
            mean_probs = all_probs.mean(dim=0)
            
            # Calculate entropy (with small epsilon to avoid log(0))
            epsilon = 1e-10
            mean_probs = torch.clamp(mean_probs, epsilon, 1-epsilon)
            entropy = -mean_probs * torch.log(mean_probs) - (1-mean_probs) * torch.log(1-mean_probs)
            
            # Store entropy metrics
            uncertainty['entropy'] = entropy
            uncertainty['mean_entropy'] = entropy.mean(dim=1)
        # Return predictions and uncertainty
        return predictions, uncertainty
    
    def save_ensemble(self, directory):
        """Save all models in the ensemble."""
        if not os.path.exists(directory): # Accumulate the models in the ensemble and save them on the same dir
            os.makedirs(directory)
        
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(directory, f"model_{i}.pth"))
            
        print(f"Saved ensemble to {directory}")
    
    def load_ensemble(self, directory):
        """Load all models in the ensemble."""
        for i, model in enumerate(self.models):
            model_path = os.path.join(directory, f"model_{i}.pth")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        print(f"Loaded ensemble from {directory}")

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
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x_q, x_kv):
        """
        Args:
            x_q: (batch_size, seq_len_q, d_model)
            x_kv: (batch_size, seq_len_kv, d_model)
        """
        # Cross-attention
        attn_output, _ = self.cross_attn(query=x_q, key=x_kv, value=x_kv)
        x = self.norm1(x_q + attn_output)

        # Feed forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class AsymmetricStarClassifier(nn.Module):
    """
    StarClassifierFusion with asymmetric dimensions for spectral and Gaia data.
    """
    def __init__(
        self,
        d_model_spectra,
        d_model_gaia,
        num_classes,
        input_dim_spectra,
        input_dim_gaia,
        n_layers=6,
        use_cross_attention=True,
        n_cross_attn_heads=8,
        d_state=16,
        d_conv=4,
        expand=2,
        use_checkpoint=True
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Input projection layers
        self.input_proj_spectra = nn.Linear(input_dim_spectra, d_model_spectra)
        self.input_proj_gaia = nn.Linear(input_dim_gaia, d_model_gaia)
        
        # Mamba layers for spectra
        self.mamba_spectra = nn.Sequential(
            *[Mamba2(
                d_model=d_model_spectra,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(n_layers)]
        )
        
        # Mamba layers for gaia
        self.mamba_gaia = nn.Sequential(
            *[Mamba2(
                d_model=d_model_gaia,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(n_layers)]
        )

        # Cross Attention (Optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            # Projection layers for cross-attention between different dimensions
            self.spectra_to_gaia_proj = nn.Linear(d_model_spectra, d_model_gaia)
            self.gaia_to_spectra_proj = nn.Linear(d_model_gaia, d_model_spectra)
            
            # Cross-attention blocks
            self.cross_attn_block_spectra = CrossAttentionBlock(d_model_spectra, n_heads=n_cross_attn_heads)
            self.cross_attn_block_gaia = CrossAttentionBlock(d_model_gaia, n_heads=n_cross_attn_heads)

        # Final Classifier - takes concatenated features
        fusion_dim = d_model_spectra + d_model_gaia
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, num_classes)
        )
    
    def forward(self, x_spectra, x_gaia):
        """
        Forward pass with asymmetric embeddings.
        
        Args:
            x_spectra: (batch_size, input_dim_spectra)
            x_gaia: (batch_size, input_dim_gaia)
        """
        # Project to embedding spaces of different dimensions
        x_spectra = self.input_proj_spectra(x_spectra)  # (B, d_model_spectra)
        x_gaia = self.input_proj_gaia(x_gaia)           # (B, d_model_gaia)
        
        # Ensure dimensions are multiples of 8
        padded_spectra_dim = ((x_spectra.shape[1] + 7) // 8) * 8
        padded_gaia_dim = ((x_gaia.shape[1] + 7) // 8) * 8
        
        if padded_spectra_dim != x_spectra.shape[1]:
            pad_spectra = nn.functional.pad(x_spectra, (0, padded_spectra_dim - x_spectra.shape[1]))
        else:
            pad_spectra = x_spectra
            
        if padded_gaia_dim != x_gaia.shape[1]:
            pad_gaia = nn.functional.pad(x_gaia, (0, padded_gaia_dim - x_gaia.shape[1]))
        else:
            pad_gaia = x_gaia
        
        # Add sequence dimension (batch_size, 1, features)
        x_spectra = pad_spectra.unsqueeze(1).contiguous()
        x_gaia = pad_gaia.unsqueeze(1).contiguous()

        # Process through Mamba encoders with gradient checkpointing
        try:
            if self.use_checkpoint and self.training:
                x_spectra = checkpoint(self.mamba_spectra, x_spectra, use_reentrant=False)
                x_gaia = checkpoint(self.mamba_gaia, x_gaia, use_reentrant=False)
            else:
                x_spectra = self.mamba_spectra(x_spectra)
                x_gaia = self.mamba_gaia(x_gaia)
        except Exception as e:
            # If Mamba fails, return zeros with appropriate dimensions
            print(f"Warning: Mamba forward pass failed with error: {str(e)}")
            print("Using input projections as features instead")
            # Skip Mamba and use the projected features directly
            x_spectra = pad_spectra.unsqueeze(1)
            x_gaia = pad_gaia.unsqueeze(1)

        # Cross-attention with dimension adaptation
        if self.use_cross_attention:
            try:
                # Project Gaia to match Spectra dimension for Spectra's cross-attention
                x_gaia_for_spectra = self.gaia_to_spectra_proj(x_gaia)
                
                # Project Spectra to match Gaia dimension for Gaia's cross-attention
                x_spectra_for_gaia = self.spectra_to_gaia_proj(x_spectra)
                
                # Cross-attention
                x_spectra_fused = self.cross_attn_block_spectra(x_spectra, x_gaia_for_spectra)
                x_gaia_fused = self.cross_attn_block_gaia(x_gaia, x_spectra_for_gaia)
                
                x_spectra = x_spectra_fused
                x_gaia = x_gaia_fused
            except Exception as e:
                print(f"Warning: Cross-attention failed with error: {str(e)}")
                print("Skipping cross-attention")
                # Keep original projected features
        
        # Pool across sequence dimension
        x_spectra = x_spectra.mean(dim=1)  # (B, d_model_spectra)
        x_gaia = x_gaia.mean(dim=1)        # (B, d_model_gaia)

        # Concatenate features of different dimensions
        x_fused = torch.cat([x_spectra, x_gaia], dim=-1)  # (B, d_model_spectra + d_model_gaia)

        # Final classification
        logits = self.classifier(x_fused)  # (B, num_classes)
        return logits


class MambaClassifierFallback(nn.Module):
    """
    Fallback model that uses simple MLP architecture when Mamba has compatibility issues
    """
    def __init__(
        self,
        d_model_spectra,
        d_model_gaia,
        num_classes,
        input_dim_spectra,
        input_dim_gaia,
        hidden_dim=1024,
        dropout=0.2
    ):
        super().__init__()
        
        # Input projection layers
        self.input_proj_spectra = nn.Linear(input_dim_spectra, d_model_spectra)
        self.input_proj_gaia = nn.Linear(input_dim_gaia, d_model_gaia)
        
        # Fusion MLP
        fusion_dim = d_model_spectra + d_model_gaia
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x_spectra, x_gaia):
        # Project inputs
        x_spectra = self.input_proj_spectra(x_spectra)
        x_gaia = self.input_proj_gaia(x_gaia)
        
        # Concatenate and feed to fusion MLP
        x_fused = torch.cat([x_spectra, x_gaia], dim=-1)
        logits = self.fusion_mlp(x_fused)
        
        return logits


class AsymmetricMemoryEfficientStarClassifier(nn.Module):
    """
    Memory-efficient version of StarClassifierFusion with asymmetric dimensions
    for spectral and Gaia data, allowing for much smaller Gaia embeddings.
    """
    def __init__(
        self,
        d_model_spectra,
        d_model_gaia,
        num_classes,
        input_dim_spectra,
        input_dim_gaia,
        n_layers=6,
        use_cross_attention=True,
        n_cross_attn_heads=8,
        d_state_spectra=16,
        d_state_gaia=8,  # Can be smaller for Gaia
        d_conv=4,
        expand=2,
        use_checkpoint=True,
        activation_checkpointing=True,
        use_half_precision=True,
        sequential_processing=True
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.activation_checkpointing = activation_checkpointing
        self.sequential_processing = sequential_processing
        
        # Store dimensions for later use
        self.d_model_spectra = d_model_spectra
        self.d_model_gaia = d_model_gaia
        
        # Use lower precision
        self.dtype = torch.float16 if use_half_precision else torch.float32

        # Input projection layers - project inputs to their respective embedding spaces
        self.input_proj_spectra = nn.Linear(input_dim_spectra, d_model_spectra)
        self.input_proj_gaia = nn.Linear(input_dim_gaia, d_model_gaia)
        
        # Memory-efficient Mamba layers for Spectra (higher dimension)
        self.mamba_spectra_layers = nn.ModuleList([
            self._create_mamba_layer(
                d_model=d_model_spectra,
                d_state=d_state_spectra,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        
        # Memory-efficient Mamba layers for Gaia (lower dimension)
        self.mamba_gaia_layers = nn.ModuleList([
            self._create_mamba_layer(
                d_model=d_model_gaia,
                d_state=d_state_gaia,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])

        # Cross-attention (optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            # Adaptation layers for cross-attention with different dimensions
            # For Gaia→Spectra attention, we need to project Gaia to match Spectra dimension
            self.gaia_to_spectra_proj = nn.Linear(d_model_gaia, d_model_spectra)
            
            # For Spectra→Gaia attention, we need to project Spectra to match Gaia dimension
            self.spectra_to_gaia_proj = nn.Linear(d_model_spectra, d_model_gaia)
            
            # Create cross-attention blocks
            self.cross_attn_block_spectra = self._create_cross_attn_block(
                d_model=d_model_spectra, n_heads=n_cross_attn_heads
            )
            self.cross_attn_block_gaia = self._create_cross_attn_block(
                d_model=d_model_gaia, n_heads=n_cross_attn_heads
            )

        # Final classifier
        # Add a projection layer to transform concatenated features to a common fusion dimension
        fusion_dim = d_model_spectra + d_model_gaia
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)
    
    def _create_mamba_layer(self, d_model, d_state, d_conv, expand):
        """Create a memory-efficient Mamba layer."""
        # Ensure d_model is a multiple of 8 to avoid stride alignment issues
        d_model = ((d_model + 7) // 8) * 8
        
        mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=True  # Add bias to potentially help with alignment issues
        )
        
        # Wrap with gradient checkpointing if requested
        if self.activation_checkpointing:
            return MemoryEfficientMamba(mamba, use_checkpoint=True)
        else:
            return mamba
        
        # Wrap with gradient checkpointing if requested
        if self.activation_checkpointing:
            return MemoryEfficientMamba(mamba, use_checkpoint=True)
        else:
            return mamba
    
    def _create_cross_attn_block(self, d_model, n_heads):
        """Creates a cross-attention block with optional gradient checkpointing."""
        class CrossAttentionBlock(nn.Module):
            def __init__(self, d_model, n_heads):
                super().__init__()
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=d_model, 
                    num_heads=n_heads, 
                    batch_first=True
                )
                self.norm1 = nn.LayerNorm(d_model)
                
                # Smaller FFN to save memory
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, 2 * d_model),  # Reduced from 4x
                    nn.ReLU(),
                    nn.Linear(2 * d_model, d_model)   # Reduced from 4x
                )
                self.norm2 = nn.LayerNorm(d_model)
                
            def forward(self, x_q, x_kv):
                # Cross-attention
                attn_output, _ = self.cross_attn(query=x_q, key=x_kv, value=x_kv)
                x = self.norm1(x_q + attn_output)
                
                # Feed forward
                ffn_out = self.ffn(x)
                x = self.norm2(x + ffn_out)
                
                return x
        
        block = CrossAttentionBlock(d_model, n_heads)
        
        # Wrap with gradient checkpointing if requested
        if self.activation_checkpointing:
            def forward_with_checkpoint(module, x_q, x_kv):
                def custom_forward(x_q, x_kv):
                    return module(x_q, x_kv)
                return checkpoint(custom_forward, x_q, x_kv, use_reentrant=False)
            
            class CheckpointedCrossAttention(nn.Module):
                def __init__(self, block):
                    super().__init__()
                    self.block = block
                
                def forward(self, x_q, x_kv):
                    return forward_with_checkpoint(self.block, x_q, x_kv)
            
            return CheckpointedCrossAttention(block)
        else:
            return block
    
    def _process_mamba_layers(self, x, layers):
        """Process input through Mamba layers, optionally sequentially to save memory."""
        if self.sequential_processing:
            for layer in layers:
                x = layer(x)
                # Optional: explicitly delete intermediate activations
                torch.cuda.empty_cache()
        else:
            # Process all layers at once (uses more memory but faster)
            for layer in layers:
                x = layer(x)
        return x
    
    def forward(self, x_spectra, x_gaia):
        # Convert to half precision if requested
        if hasattr(self, 'dtype') and self.dtype == torch.float16:
            x_spectra = x_spectra.half()
            x_gaia = x_gaia.half()
        
        # Project inputs to their respective embedding spaces
        x_spectra = self.input_proj_spectra(x_spectra)
        x_gaia = self.input_proj_gaia(x_gaia)
        
        # Add sequence dimension if needed and ensure contiguous memory layout
        if len(x_spectra.shape) == 2:
            x_spectra = x_spectra.unsqueeze(1).contiguous()
        else:
            x_spectra = x_spectra.contiguous()
            
        if len(x_gaia.shape) == 2:
            x_gaia = x_gaia.unsqueeze(1).contiguous()
        else:
            x_gaia = x_gaia.contiguous()
        
        # Process through Mamba layers
        x_spectra = self._process_mamba_layers(x_spectra, self.mamba_spectra_layers)
        x_gaia = self._process_mamba_layers(x_gaia, self.mamba_gaia_layers)
        
        # Optional cross-attention (with dimension adaptation)
        if self.use_cross_attention:
            # Project Gaia features to match spectra dimension for spectra's cross-attention
            x_gaia_projected = self.gaia_to_spectra_proj(x_gaia)
            
            # Project Spectra features to match Gaia dimension for Gaia's cross-attention
            x_spectra_projected = self.spectra_to_gaia_proj(x_spectra)
            
            # Cross-attention from spectra -> gaia (using projected Gaia)
            x_spectra_fused = self.cross_attn_block_spectra(x_spectra, x_gaia_projected)
            
            # Cross-attention from gaia -> spectra (using projected Spectra)
            x_gaia_fused = self.cross_attn_block_gaia(x_gaia, x_spectra_projected)
            
            # Update embeddings
            x_spectra = x_spectra_fused
            x_gaia = x_gaia_fused
            
            # Free memory
            del x_gaia_projected, x_spectra_projected, x_spectra_fused, x_gaia_fused
            torch.cuda.empty_cache()
        
        # Pool across sequence dimension
        x_spectra = x_spectra.mean(dim=1)
        x_gaia = x_gaia.mean(dim=1)
        
        # Concatenate (different dimensions are fine for concatenation)
        x_fused = torch.cat([x_spectra, x_gaia], dim=-1)
        
        # Final classification
        x_fused = self.layer_norm(x_fused)
        logits = self.classifier(x_fused)
        
        return logits


class MultiModalBalancedMultiLabelDataset(Dataset):
    """
    A balanced multi-label dataset that returns (X_spectra, X_gaia, y).
    It uses balancing strategy to ensure approximately equal representation of each class.
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
    
class MultiModalBalancedMultiLabelDataset(Dataset):
    """
    A balanced multi-label dataset that returns (X_spectra, X_gaia, y).
    It uses balancing strategy to ensure approximately equal representation of each class.
    """
    def __init__(self, X_spectra, X_gaia, y, limit_per_label=201, min_samples_per_label=1):
        """
        Args:
            X_spectra (torch.Tensor): [num_samples, num_spectra_features]
            X_gaia (torch.Tensor): [num_samples, num_gaia_features]
            y (torch.Tensor): [num_samples, num_classes], multi-hot labels
            limit_per_label (int): limit or target number of samples per label
            min_samples_per_label (int): minimum number of samples to keep for each label
        """
        self.X_spectra = X_spectra
        self.X_gaia = X_gaia
        self.y = y
        self.limit_per_label = limit_per_label
        self.min_samples_per_label = min_samples_per_label
        self.num_classes = y.shape[1]
        
        # Print class distribution before balancing
        class_counts = torch.sum(self.y, axis=0).cpu().numpy()
        print(f"Class distribution before balancing: min={class_counts.min()}, max={class_counts.max()}, "
              f"mean={class_counts.mean():.1f}, median={np.median(class_counts)}")
        print(f"Classes with fewer than {min_samples_per_label} samples: "
              f"{sum(1 for c in class_counts if c < min_samples_per_label)}")
        
        self.indices = self.balance_classes()
        
        # Calculate and print the class distribution after balancing
        balanced_counts = np.zeros(self.num_classes)
        for idx in self.indices:
            balanced_counts += self.y[idx].cpu().numpy()
        
        print(f"Class distribution after balancing: min={balanced_counts.min()}, max={balanced_counts.max()}, "
              f"mean={balanced_counts.mean():.1f}, median={np.median(balanced_counts)}")
        print(f"Classes with fewer than {min_samples_per_label} samples after balancing: "
              f"{sum(1 for c in balanced_counts if c < min_samples_per_label)}")

    def balance_classes(self):
        """
        Balance classes by including a target number of samples for each class.
        For rare classes, samples are duplicated to reach the target.
        For common classes, samples are randomly selected up to the limit.
        
        Returns:
            List of indices for the balanced dataset
        """
        indices = []
        class_counts = torch.sum(self.y, axis=0).cpu().numpy()
        
        # First, handle rare classes to ensure they are properly represented
        for cls in range(self.num_classes):
            cls_indices = np.where(self.y[:, cls].cpu().numpy() == 1)[0]
            
            if len(cls_indices) == 0:
                print(f"Warning: Class {cls} has no samples!")
                continue
                
            # For very rare classes, duplicate samples to reach at least min_samples_per_label
            if len(cls_indices) < self.min_samples_per_label:
                # Calculate how many times we need to duplicate each sample on average
                duplication_factor = max(1, int(np.ceil(self.min_samples_per_label / len(cls_indices))))
                
                # Duplicate the indices
                duplicated_indices = np.repeat(cls_indices, duplication_factor)
                
                # If we have too many, trim down
                if len(duplicated_indices) > self.limit_per_label:
                    duplicated_indices = np.random.choice(duplicated_indices, self.limit_per_label, replace=False)
                
                indices.extend(duplicated_indices)
                
            # For classes with more than minimum but less than target, duplicate to reach target
            elif len(cls_indices) < self.limit_per_label:
                # Add all original indices
                indices.extend(cls_indices)
                
                # Calculate how many more we need
                additional_needed = self.limit_per_label - len(cls_indices)
                
                # Sample with replacement to get additional_needed more samples
                extra_indices = np.random.choice(cls_indices, additional_needed, replace=True)
                indices.extend(extra_indices)
                
            # For common classes, just randomly select up to the limit
            else:
                selected_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
                indices.extend(selected_indices)
        
        # Convert to unique indices (a sample may be selected for multiple classes)
        indices = np.unique(indices)
        np.random.shuffle(indices)
        
        return indices

    def re_sample(self):
        """Resample the dataset to get a different balance of examples"""
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


def fix_onecycle_scheduler(scheduler, train_loader, batch_accumulation):
    """
    Fix the OneCycleLR scheduler to account for gradient accumulation.
    
    Args:
        scheduler: The scheduler to fix
        train_loader: The training data loader
        batch_accumulation: Number of batches to accumulate before updating
        
    Returns:
        Fixed scheduler
    """
    # Calculate correct number of steps
    effective_steps_per_epoch = (len(train_loader) + batch_accumulation - 1) // batch_accumulation
    total_steps = effective_steps_per_epoch * scheduler.total_steps // len(train_loader)
    
    # Create new scheduler with correct step count
    new_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        scheduler.optimizer,
        max_lr=scheduler.max_lrs,
        total_steps=total_steps,
        pct_start=scheduler.pct_start,
        anneal_strategy=scheduler.anneal_strategy,
        div_factor=scheduler.div_factor,
        final_div_factor=scheduler.final_div_factor
    )
    
    return new_scheduler


def calculate_class_weights(y):
    """
    Calculate class weights to handle class imbalance.
    
    Args:
        y: Ground truth labels (numpy array)
        
    Returns:
        Class weights (numpy array)
    """
    if y.ndim > 1:
        class_counts = np.sum(y, axis=0)
    else:
        class_counts = np.bincount(y)
    total_samples = y.shape[0] if y.ndim > 1 else len(y)
    class_counts = np.where(class_counts == 0, 1, class_counts)  # Prevent division by zero
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights


def calculate_metrics(y_true, y_pred):
    """
    Calculate various metrics for multi-label classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
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
    # ROC AUC could be added here if needed
    return metrics

def calculate_metrics(y_true, y_pred):
    """
    Calculate various metrics for multi-label classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "micro_f1": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred)
    }
    return metrics


def train_model_fusion_with_wandb(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs=100,
    lr=2.5e-3,
    max_patience=20,
    device='cuda',
    batch_accumulation=1,
    use_reentrant=False,
    project_name="star-classifier",
    run_name=None,
    config=None
):
    """
    Train a fusion model with gradient accumulation support and Weights & Biases logging.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_epochs: Maximum number of epochs to train
        lr: Learning rate
        max_patience: Early stopping patience
        device: Device to train on
        batch_accumulation: Number of batches to accumulate before updating weights
        use_reentrant: Whether to use reentrant checkpointing
        project_name: Name of the wandb project
        run_name: Name of the wandb run
        config: Configuration to log in wandb
        
    Returns:
        Trained model
    """
    # Initialize wandb
    if config is None:
        config = {
            "learning_rate": lr,
            "batch_accumulation": batch_accumulation,
            "architecture": model.__class__.__name__,
            "epochs": num_epochs,
            "early_stopping_patience": max_patience,
            "dataset_size": len(train_loader.dataset)
        }
    
    wandb.init(project=project_name, name=run_name, config=config)
    wandb.watch(model)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Calculate effective steps for OneCycleLR to prevent "stepped too many times" error
    effective_steps = len(train_loader) // batch_accumulation
    if len(train_loader) % batch_accumulation != 0:
        effective_steps += 1
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=effective_steps  # This is the key fix
    )
    
    # Calculate class weights
    all_labels = []
    for _, _, y_batch in train_loader:
        all_labels.extend(y_batch.cpu().numpy())
    
    class_weights = calculate_class_weights(np.array(all_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    best_val_loss = float('inf')
    patience = max_patience
    best_model = None
    
    for epoch in range(num_epochs):
        # Resample training data
        train_loader.dataset.re_sample()
        
        # Training with gradient accumulation
        model.train()
        train_loss, train_acc = 0.0, 0.0
        batch_count = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for i, (X_spc, X_ga, y_batch) in enumerate(train_loader):
            X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_spc, X_ga)
            loss = criterion(outputs, y_batch) / batch_accumulation  # Scale loss
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            train_loss += loss.item() * batch_accumulation * X_spc.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predicted == y_batch).float()
            train_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
            batch_count += X_spc.size(0)
            
            # Step optimizer and scheduler only after accumulating gradients
            if (i + 1) % batch_accumulation == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()  # Safe because steps_per_epoch is fixed
                optimizer.zero_grad()
        
        # Calculate average metrics
        train_loss /= batch_count
        train_acc /= batch_count
        
        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for X_spc, X_ga, y_batch in val_loader:
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                
                outputs = model(X_spc, X_ga)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_spc.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct = (predicted == y_batch).float()
                val_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
                val_batch_count += X_spc.size(0)
        
        val_loss /= val_batch_count
        val_acc /= val_batch_count
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            best_model = model.state_dict().copy()
            print(f"New best model with validation loss: {val_loss:.4f}")
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Test evaluation (after each epoch)
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        all_preds, all_labels = [], []
        test_batch_count = 0
        
        with torch.no_grad():
            for X_spc, X_ga, y_batch in test_loader:
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                
                outputs = model(X_spc, X_ga)
                loss = criterion(outputs, y_batch)
                
                test_loss += loss.item() * X_spc.size(0)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                correct = (predicted == y_batch).float()
                test_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
                
                test_batch_count += X_spc.size(0)
                
                # Collect predictions and labels for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        test_loss /= test_batch_count
        test_acc /= test_batch_count
        
        # Calculate detailed metrics
        all_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            **all_metrics
        })
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    model.eval()
    
    # Final test evaluation
    test_loss, test_acc = 0.0, 0.0
    all_preds, all_labels = [], []
    test_batch_count = 0
    
    with torch.no_grad():
        for X_spc, X_ga, y_batch in test_loader:
            X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
            
            outputs = model(X_spc, X_ga)
            loss = criterion(outputs, y_batch)
            
            test_loss += loss.item() * X_spc.size(0)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct = (predicted == y_batch).float()
            test_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
            
            test_batch_count += X_spc.size(0)
            
            # Collect predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    test_loss /= test_batch_count
    test_acc /= test_batch_count
    
    # Calculate detailed metrics
    all_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print("Test Metrics:")
    for metric, value in all_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Log final metrics to wandb
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        **{f"final_{k}": v for k, v in all_metrics.items()}
    })
    
    # Close wandb
    wandb.finish()
    
    return model
    """
    Train a fusion model with gradient accumulation support.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_epochs: Maximum number of epochs to train
        lr: Learning rate
        max_patience: Early stopping patience
        device: Device to train on
        batch_accumulation: Number of batches to accumulate before updating weights
        
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Calculate effective steps for OneCycleLR to prevent "stepped too many times" error
    effective_steps = len(train_loader) // batch_accumulation
    if len(train_loader) % batch_accumulation != 0:
        effective_steps += 1
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=effective_steps  # This is the key fix
    )
    
    # Calculate class weights
    all_labels = []
    for _, _, y_batch in train_loader:
        all_labels.extend(y_batch.cpu().numpy())
    
    class_weights = calculate_class_weights(np.array(all_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    best_val_loss = float('inf')
    patience = max_patience
    best_model = None
    
    for epoch in range(num_epochs):
        # Resample training data
        train_loader.dataset.re_sample()
        
        # Training with gradient accumulation
        model.train()
        train_loss, train_acc = 0.0, 0.0
        batch_count = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for i, (X_spc, X_ga, y_batch) in enumerate(train_loader):
            X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_spc, X_ga)
            loss = criterion(outputs, y_batch) / batch_accumulation  # Scale loss
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            train_loss += loss.item() * batch_accumulation * X_spc.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predicted == y_batch).float()
            train_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
            batch_count += X_spc.size(0)
            
            # Step optimizer and scheduler only after accumulating gradients
            if (i + 1) % batch_accumulation == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()  # Safe because steps_per_epoch is fixed
                optimizer.zero_grad()
        
        # Calculate average metrics
        train_loss /= batch_count
        train_acc /= batch_count
        
        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        batch_count = 0
        
        with torch.no_grad():
            for X_spc, X_ga, y_batch in val_loader:
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                
                outputs = model(X_spc, X_ga)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_spc.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct = (predicted == y_batch).float()
                val_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
                batch_count += X_spc.size(0)
        
        val_loss /= batch_count
        val_acc /= batch_count
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            best_model = model.state_dict().copy()
            print(f"New best model with validation loss: {val_loss:.4f}")
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    model.eval()
    
    # Test evaluation
    test_loss, test_acc = 0.0, 0.0
    all_preds, all_labels = [], []
    batch_count = 0
    
    with torch.no_grad():
        for X_spc, X_ga, y_batch in test_loader:
            X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
            
            outputs = model(X_spc, X_ga)
            loss = criterion(outputs, y_batch)
            
            test_loss += loss.item() * X_spc.size(0)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct = (predicted == y_batch).float()
            test_acc += correct.mean(dim=1).mean().item() * X_spc.size(0)
            
            batch_count += X_spc.size(0)
            
            # Collect predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    test_loss /= batch_count
    test_acc /= batch_count
    
    # Calculate detailed metrics
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print("Test Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model


# Sample configurations optimized for 24GB GPU with dimensions aligned to 8
MEMORY_EFFICIENT_CONFIGS = {
    # Maximum spectral dimension with small Gaia dimension
    "max_spectral": {
        "d_model_spectra": 3072,  # Already divisible by 8
        "d_model_gaia": 512,      # Already divisible by 8
        "num_classes": 55,
        "input_dim_spectra": 3647,
        "input_dim_gaia": 18,
        "n_layers": 12,
        "d_state": 32,
        "d_conv": 4,
        "expand": 2,
        "use_cross_attention": True,
        "n_cross_attn_heads": 8
    },
    
    # Balanced dimensions for better cross-modal learning
    "balanced": {
        "d_model_spectra": 3072,  # Already divisible by 8
        "d_model_gaia": 512,      # Already divisible by 8
        "num_classes": 55,
        "input_dim_spectra": 3647,
        "input_dim_gaia": 18,
        "n_layers": 10,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "use_cross_attention": True,
        "n_cross_attn_heads": 6
    },
    
    # Extremely low memory usage
    "low_memory": {
        "d_model_spectra": 2048,  # Already divisible by 8
        "d_model_gaia": 256,      # Already divisible by 8
        "num_classes": 55,
        "input_dim_spectra": 3647,
        "input_dim_gaia": 18,
        "n_layers": 8,
        "d_state": 16,            
        "d_conv": 4,              # Increased from 2 to 4
        "expand": 2,              # Increased from 1 to 2
        "use_cross_attention": True,
        "n_cross_attn_heads": 4
    },
    
    # Ultra small model for compatibility
    "minimal": {
        "d_model_spectra": 512,   # Small but divisible by 8
        "d_model_gaia": 128,      # Small but divisible by 8
        "num_classes": 55,
        "input_dim_spectra": 3647,
        "input_dim_gaia": 18,
        "n_layers": 4,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "use_cross_attention": True,
        "n_cross_attn_heads": 2
    }
}


# Main execution code
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set batch size and batch accumulation
    batch_size = 128
    batch_limit = int(batch_size / 2.5)
    batch_accumulation = 4  # Accumulate gradients over 4 batches
    
    # Load datasets
    print("Loading datasets...")
    try:
        with open("Pickles/Updated_List_of_Classes_ubuntu.pkl", "rb") as f:
            classes = pickle.load(f)
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
        
        # Define Gaia columns
        gaia_columns = [
            "parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", 
            "pmra", "pmdec", "pmra_error", "pmdec_error", "phot_g_mean_flux", 
            "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", 
            "phot_rp_mean_flux", "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", 
            "flagnoflux"
        ]
        
        # Spectra data (everything that is not Gaia-related)
        X_train_spectra = X_train_full.drop(columns={"otype", "obsid", *gaia_columns})
        X_test_spectra = X_test_full.drop(columns={"otype", "obsid", *gaia_columns})
        
        # Gaia data (only the selected columns)
        X_train_gaia = X_train_full[gaia_columns]
        X_test_gaia = X_test_full[gaia_columns]
        
        # Check for NaNs and infs
        print("NaN counts in Gaia training data:")
        print(X_train_gaia.isnull().sum())
        print("Inf counts in Gaia training data:")
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
        
        # Convert to PyTorch tensors
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
        
        print(f"y_train shape: {y_train.shape}")
        print(f"y_val shape: {y_val.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # Create datasets and dataloaders
        train_dataset = MultiModalBalancedMultiLabelDataset(X_train_spectra, X_train_gaia, y_train, limit_per_label=batch_limit)
        val_dataset = MultiModalBalancedMultiLabelDataset(X_val_spectra, X_val_gaia, y_val, limit_per_label=batch_limit)
        test_dataset = MultiModalBalancedMultiLabelDataset(X_test_spectra, X_test_gaia, y_test, limit_per_label=batch_limit)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Print dataset sizes
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Validation dataset: {len(val_dataset)} samples")
        print(f"Test dataset: {len(test_dataset)} samples")
        
        # Create the model
        # Choose configuration based on available VRAM
        print("Creating model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = None
        
        # Select configuration based on available VRAM
        if torch.cuda.is_available():
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            print(f"Available VRAM: {vram_mb:.2f} MB")
            
            if vram_mb > 22000:  # More than 22GB
                config = MEMORY_EFFICIENT_CONFIGS["max_spectral"]
                print("Using 'max_spectral' configuration")
            elif vram_mb > 12000:  # More than 12GB
                config = MEMORY_EFFICIENT_CONFIGS["balanced"]
                print("Using 'balanced' configuration")
            else:
                config = MEMORY_EFFICIENT_CONFIGS["low_memory"]
                print("Using 'low_memory' configuration")
        else:
            config = MEMORY_EFFICIENT_CONFIGS["minimal"]
            print("CUDA not available. Using 'minimal' configuration")

        config = MEMORY_EFFICIENT_CONFIGS["low_memory"]
        print("Using 'low_memory' configuration for testing")        
        # Create model parameters dictionary
        model_params = {
            "d_model_spectra": config["d_model_spectra"],
            "d_model_gaia": config["d_model_gaia"],
            "num_classes": config["num_classes"],
            "input_dim_spectra": config["input_dim_spectra"],
            "input_dim_gaia": config["input_dim_gaia"],
            "n_layers": config["n_layers"],
            "d_state_spectra": config["d_state"], 
            "d_state_gaia": 16,  # Must be 16 (not 8) for alignment
            "d_conv": config["d_conv"],
            "expand": config["expand"],
            "use_cross_attention": config["use_cross_attention"],
            "n_cross_attn_heads": config["n_cross_attn_heads"],
            "use_checkpoint": True,
            "activation_checkpointing": True,
            "use_half_precision": False,  # Disable half precision to fix dtype issues
            "sequential_processing": True
        }
        
        print(f"Creating model with d_model_spectra={model_params['d_model_spectra']}, d_model_gaia={model_params['d_model_gaia']}")
        
        # Initialize wandb
        wandb.init(project="star-classifier", name="ensemble_training")
        
        # Create and train ensemble model
        ensemble_size = 10  # Can be adjusted based on available resources
        ensemble = EnsembleModelWithUncertainty(
            model_class=AsymmetricMemoryEfficientStarClassifier,
            model_params=model_params,
            num_models=ensemble_size,
            device=device,
            uncertainty_method='entropy'
        )
        
        # Train the ensemble
        print(f"Training ensemble with {ensemble_size} models on {device}...")
        ensemble.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=600,
            lr=3e-5,
            max_patience=50,
            batch_accumulation=4,
            use_wandb=True,
            wandb_project="star-classifier",
            wandb_config={
                "ensemble_size": ensemble_size,
                "model_type": "AsymmetricMemoryEfficientStarClassifier",
                **model_params
            }
        )
        
        # Save the ensemble
        ensemble.save_ensemble("ensemble_models")

        # Initialize a new wandb run specifically for uncertainty analysis
        wandb.init(project="star-classifier", name="uncertainty_analysis", group="ensemble")
        
        # Fix the uncertainty testing code to handle both uncertainty methods
        print("\nTesting uncertainty quantification on validation data...")
        val_uncertainties = []
        val_predictions = []
        val_labels = []

        # Process validation set in batches
        for X_spc, X_ga, y_batch in val_loader:
            # Get predictions and uncertainty metrics
            preds, uncertainty = ensemble.predict(X_spc, X_ga, return_uncertainty=True)
            
            # Store predictions and labels
            val_predictions.extend(preds.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())
            
            # Store mean uncertainty for each sample - select appropriate key based on method
            if ensemble.uncertainty_method == 'entropy':
                # Using the entropy-based uncertainty
                val_uncertainties.extend(uncertainty['mean_entropy'].cpu().numpy())
            elif ensemble.uncertainty_method == 'variance':
                # Using the variance-based uncertainty
                val_uncertainties.extend(uncertainty['mean_std'].cpu().numpy())
            else:
                # Fallback to any available uncertainty metric
                key = next(iter(k for k in uncertainty.keys() if k.startswith('mean_')))
                val_uncertainties.extend(uncertainty[key].cpu().numpy())

        # Convert to numpy arrays
        val_uncertainties = np.array(val_uncertainties)
        val_predictions = np.array(val_predictions)
        val_labels = np.array(val_labels)

        # Analyze relationship between uncertainty and prediction error
        prediction_errors = np.abs(val_predictions - val_labels).mean(axis=1)

        # Calculate correlation between uncertainty and error
        correlation = np.corrcoef(val_uncertainties, prediction_errors)[0, 1]
        print(f"Correlation between uncertainty and prediction error: {correlation:.4f}")

        # Log to wandb
        uncertainty_metric_name = 'mean_entropy' if ensemble.uncertainty_method == 'entropy' else 'mean_std'
        wandb.log({
            "uncertainty_error_correlation": correlation,
            "mean_uncertainty": np.mean(val_uncertainties),
            "max_uncertainty": np.max(val_uncertainties),
            "min_uncertainty": np.min(val_uncertainties)
        })

        # Create a scatter plot of uncertainty vs error
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(val_uncertainties, prediction_errors, alpha=0.5)
        ax.set_xlabel(f'Prediction Uncertainty ({uncertainty_metric_name})')
        ax.set_ylabel('Prediction Error')
        ax.set_title(f'Uncertainty vs Error (Correlation: {correlation:.4f})')
        plt.tight_layout()

        # Log figure to wandb
        wandb.log({"uncertainty_vs_error": wandb.Image(fig)})

        # Close wandb
        wandb.finish()
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure the Pickles directory contains the required files.")
    except Exception as e:
        print(f"An error occurred: {e}")