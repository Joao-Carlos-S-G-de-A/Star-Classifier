import os
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Load models and make predictions (adjust paths as needed)
def load_ensemble_models(ensemble_dir="ensemble_models", model_class=None, model_params=None, device='cuda'):
    """
    Load ensemble models from the specified directory.
    
    Args:
        ensemble_dir: Directory containing model files
        model_class: The model class to instantiate
        model_params: Parameters for model initialization
        device: Device to load models on
        
    Returns:
        List of loaded models
    """
    models = []
    
    if not os.path.exists(ensemble_dir):
        print(f"Error: Directory {ensemble_dir} does not exist.")
        return models
    
    if model_class is None or model_params is None:
        print("Error: model_class and model_params must be provided.")
        return models
    
    model_files = [f for f in os.listdir(ensemble_dir) if f.startswith("model_") and f.endswith(".pth")]
    
    if not model_files:
        print(f"No model files found in {ensemble_dir}")
        return models
    
    for model_file in sorted(model_files):
        try:
            # Initialize model
            model = model_class(**model_params).to(device)
            
            # Load state
            model.load_state_dict(torch.load(os.path.join(ensemble_dir, model_file), map_location=device))
            model.eval()
            
            models.append(model)
            print(f"Loaded {model_file}")
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
    
    print(f"Successfully loaded {len(models)} models")
    return models

# Calculate uncertainties (entropy or variance-based)
def calculate_uncertainty(all_probs, method='entropy'):
    """
    Calculate uncertainty metrics from probability predictions.
    
    Args:
        all_probs: Tensor of shape [num_models, num_samples, num_classes]
        method: 'entropy' or 'variance'
        
    Returns:
        Dictionary with uncertainty metrics
    """
    # Mean probabilities across models
    mean_probs = all_probs.mean(dim=0)  # [num_samples, num_classes]
    
    if method == 'entropy':
        # Calculate entropy from mean probabilities
        # Clip probabilities to avoid log(0)
        epsilon = 1e-10
        mean_probs_clipped = torch.clamp(mean_probs, epsilon, 1-epsilon)
        
        # Entropy for binary classification: -p*log2(p) - (1-p)*log2(1-p)
        entropy = -mean_probs_clipped * torch.log2(mean_probs_clipped) - \
                 (1-mean_probs_clipped) * torch.log2(1-mean_probs_clipped)
        
        # Calculate mean entropy per sample
        mean_entropy = entropy.mean(dim=1)
        
        return {
            'entropy': entropy,
            'mean_entropy': mean_entropy
        }
    
    elif method == 'variance':
        # Variance of probabilities across models
        variance = all_probs.var(dim=0)  # [num_samples, num_classes]
        
        # Calculate mean variance per sample
        mean_variance = variance.mean(dim=1)
        
        # Model disagreement (how many models disagree with the majority)
        binary_preds = (all_probs > 0.5).float()
        majority_vote = (binary_preds.sum(dim=0) > (all_probs.shape[0] / 2)).float()
        disagreement = torch.abs(binary_preds - majority_vote.unsqueeze(0)).sum(dim=0) / all_probs.shape[0]
        
        return {
            'variance': variance,
            'mean_variance': mean_variance,
            'disagreement': disagreement,
            'mean_disagreement': disagreement.mean(dim=1)
        }
    
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

# Run ensemble prediction and uncertainty analysis on test data
def analyze_ensemble_predictions(models, test_loader, class_names, device='cuda', uncertainty_method='entropy'):
    """
    Run ensemble prediction and calculate uncertainties.
    
    Args:
        models: List of trained models
        test_loader: DataLoader with test data
        class_names: List of class names
        device: Device to run models on
        uncertainty_method: 'entropy' or 'variance'
        
    Returns:
        DataFrame with predictions, ground truth, and uncertainty metrics
    """
    if not models:
        print("No models available for prediction.")
        return None
    
    results = []
    
    with torch.no_grad():
        for batch_idx, (X_spc, X_gaia, y_batch) in enumerate(test_loader):
            X_spc, X_gaia, y_batch = X_spc.to(device), X_gaia.to(device), y_batch.to(device)
            
            # Get predictions from all models
            all_probs = []
            for model in models:
                logits = model(X_spc, X_gaia)
                probs = torch.sigmoid(logits)
                all_probs.append(probs)
            
            # Stack predictions [num_models, batch_size, num_classes]
            all_probs = torch.stack(all_probs)
            
            # Calculate mean probabilities across models
            mean_probs = all_probs.mean(dim=0)
            
            # Calculate uncertainties
            uncertainty = calculate_uncertainty(all_probs, method=uncertainty_method)
            
            # Convert to binary predictions
            predictions = (mean_probs > 0.5).float()
            
            # Calculate errors
            errors = torch.abs(predictions - y_batch)
            mean_errors = errors.mean(dim=1)
            
            # Store results
            for i in range(X_spc.size(0)):
                sample_result = {
                    'sample_id': batch_idx * test_loader.batch_size + i,
                    'mean_error': mean_errors[i].item(),
                }
                
                # Add uncertainty metrics
                for key, value in uncertainty.items():
                    if key.startswith('mean_'):
                        sample_result[key] = value[i].item()
                
                # Add per-class ground truth, predictions, and uncertainties
                for j, class_name in enumerate(class_names):
                    sample_result[f'true_{class_name}'] = y_batch[i, j].item()
                    sample_result[f'pred_{class_name}'] = predictions[i, j].item()
                    sample_result[f'prob_{class_name}'] = mean_probs[i, j].item()
                    
                    if 'entropy' in uncertainty:
                        sample_result[f'entropy_{class_name}'] = uncertainty['entropy'][i, j].item()
                    if 'variance' in uncertainty:
                        sample_result[f'variance_{class_name}'] = uncertainty['variance'][i, j].item()
                
                results.append(sample_result)
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate correlation between uncertainty and error
    if uncertainty_method == 'entropy':
        correlation = results_df['mean_entropy'].corr(results_df['mean_error'])
    else:
        correlation = results_df['mean_variance'].corr(results_df['mean_error'])
    
    print(f"Correlation between uncertainty and error: {correlation:.4f}")
    
    return results_df

# Calculate per-class metrics
def calculate_class_metrics(results_df, class_names):
    """
    Calculate performance metrics for each class.
    
    Args:
        results_df: DataFrame with predictions and ground truth
        class_names: List of class names
        
    Returns:
        DataFrame with per-class metrics
    """
    metrics = []
    
    for class_name in class_names:
        y_true = results_df[f'true_{class_name}'].values
        y_pred = results_df[f'pred_{class_name}'].values
        
        # Check if we have any positive examples
        if y_true.sum() == 0:
            print(f"Warning: No positive examples for class {class_name}")
            continue
            
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate average entropy for correct and incorrect predictions
        if f'entropy_{class_name}' in results_df.columns:
            entropy_values = results_df[f'entropy_{class_name}'].values
            correct_mask = y_true == y_pred
            incorrect_mask = ~correct_mask
            
            avg_entropy = entropy_values.mean()
            avg_entropy_correct = entropy_values[correct_mask].mean() if correct_mask.any() else 0
            avg_entropy_incorrect = entropy_values[incorrect_mask].mean() if incorrect_mask.any() else 0
        else:
            avg_entropy = avg_entropy_correct = avg_entropy_incorrect = 0
        
        # Get confusion matrix counts
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        metrics.append({
            'class_name': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'avg_entropy': avg_entropy,
            'avg_entropy_correct': avg_entropy_correct,
            'avg_entropy_incorrect': avg_entropy_incorrect,
            'support': tp + fn,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        })
    
    # Convert to DataFrame and sort by F1 score
    metrics_df = pd.DataFrame(metrics).sort_values('f1', ascending=False)
    return metrics_df

# Analyze uncertainty distribution
def analyze_uncertainty_distribution(results_df, uncertainty_col='mean_entropy', error_col='mean_error', num_bins=20):
    """
    Analyze the distribution of uncertainty values.
    
    Args:
        results_df: DataFrame with predictions and uncertainties
        uncertainty_col: Column name for uncertainty values
        error_col: Column name for error values
        num_bins: Number of bins for histogram
        
    Returns:
        DataFrame with binned uncertainty distribution and average errors
    """
    if uncertainty_col not in results_df.columns or error_col not in results_df.columns:
        print(f"Error: Columns {uncertainty_col} or {error_col} not found in results DataFrame")
        return None
    
    # Calculate min and max uncertainty values
    min_uncertainty = results_df[uncertainty_col].min()
    max_uncertainty = results_df[uncertainty_col].max()
    
    # Create bins
    bin_edges = np.linspace(min_uncertainty, max_uncertainty, num_bins + 1)
    bin_width = (max_uncertainty - min_uncertainty) / num_bins
    
    # Assign samples to bins
    results_df['bin_index'] = pd.cut(
        results_df[uncertainty_col], 
        bins=bin_edges, 
        labels=False, 
        include_lowest=True
    )
    
    # Calculate statistics for each bin
    bin_stats = []
    
    for i in range(num_bins):
        bin_data = results_df[results_df['bin_index'] == i]
        
        if len(bin_data) > 0:
            bin_stats.append({
                'bin_start': min_uncertainty + i * bin_width,
                'bin_end': min_uncertainty + (i + 1) * bin_width,
                'count': len(bin_data),
                'avg_error': bin_data[error_col].mean(),
                'std_error': bin_data[error_col].std(),
                'min_error': bin_data[error_col].min(),
                'max_error': bin_data[error_col].max(),
            })
    
    # Convert to DataFrame
    bin_stats_df = pd.DataFrame(bin_stats)
    
    # Add bin label for plotting
    bin_stats_df['label'] = bin_stats_df.apply(
        lambda x: f"{x['bin_start']:.3f}-{x['bin_end']:.3f}", 
        axis=1
    )
    
    return bin_stats_df

# Create visualizations
def create_visualizations(results_df, class_metrics_df, uncertainty_bins_df, uncertainty_col='mean_entropy'):
    """
    Create visualizations for ensemble model performance and uncertainty.
    
    Args:
        results_df: DataFrame with predictions and uncertainties
        class_metrics_df: DataFrame with per-class metrics
        uncertainty_bins_df: DataFrame with binned uncertainty distribution
        uncertainty_col: Column name for uncertainty values
    """
    # Set up figure size
    plt.figure(figsize=(20, 25))
    
    # 1. Class Performance Plot
    plt.subplot(3, 1, 1)
    class_plot_data = class_metrics_df.sort_values('f1', ascending=True).tail(15)  # Show top 15 classes
    class_plot_data.plot(
        x='class_name', 
        y=['precision', 'recall', 'f1'], 
        kind='barh', 
        ax=plt.gca(),
        title='Performance by Stellar Class (Top 15 Classes)',
        xlabel='Score',
        ylabel='Stellar Class'
    )
    plt.tight_layout()
    
    # 2. Uncertainty Distribution
    plt.subplot(3, 1, 2)
    ax1 = uncertainty_bins_df.plot(
        x='label', 
        y='count', 
        kind='bar',
        color='blue',
        alpha=0.7,
        title='Uncertainty Distribution and Avg. Error',
        ax=plt.gca()
    )
    ax1.set_xlabel('Uncertainty Bin')
    ax1.set_ylabel('Count')
    
    # Add average error as a line plot
    ax2 = ax1.twinx()
    uncertainty_bins_df.plot(
        x='label', 
        y='avg_error', 
        kind='line',
        color='red',
        marker='o',
        ax=ax2
    )
    ax2.set_ylabel('Average Error', color='red')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # 3. Uncertainty vs Error Scatter Plot
    plt.subplot(3, 1, 3)
    plt.scatter(
        results_df[uncertainty_col], 
        results_df['mean_error'],
        alpha=0.5
    )
    plt.title('Uncertainty vs. Prediction Error')
    plt.xlabel(f'Prediction Uncertainty ({uncertainty_col})')
    plt.ylabel('Prediction Error')
    
    # Add correlation coefficient
    correlation = results_df[uncertainty_col].corr(results_df['mean_error'])
    plt.annotate(
        f'Correlation: {correlation:.4f}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
    )
    
    # Add linear regression line
    z = np.polyfit(results_df[uncertainty_col], results_df['mean_error'], 1)
    p = np.poly1d(z)
    plt.plot(
        np.sort(results_df[uncertainty_col]), 
        p(np.sort(results_df[uncertainty_col])), 
        "r--", 
        linewidth=2
    )
    
    plt.tight_layout()
    plt.savefig('ensemble_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to 'ensemble_uncertainty_analysis.png'")
def analyze_ensemble_predictions(models, test_loader, class_names, device='cuda', uncertainty_method='entropy'):
    """
    Run ensemble prediction and calculate uncertainties with consistent precision.
    """
    if not models:
        print("No models available for prediction.")
        return None
    
    # First, check for any mixed precision in models
    print("Checking for mixed precision in models...")
    for i, model in enumerate(models):
        mixed_found = False
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                if not mixed_found:
                    print(f"Model {i} has mixed precision parameters:")
                    mixed_found = True
                print(f"  Parameter {name} has dtype {param.dtype}")
        
        if not mixed_found:
            print(f"Model {i} has consistent float32 parameters")
    
    # Convert all models to consistent precision (float32)
    print("Converting all models to consistent float32 precision...")
    for model in models:
        for param in model.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.float()
    
    results = []
    
    with torch.no_grad():
        for batch_idx, (X_spc, X_ga, y_batch) in enumerate(test_loader):
            # Move inputs to device with consistent precision
            X_spc = X_spc.to(device).float()  # Ensure float32
            X_ga = X_ga.to(device).float()    # Ensure float32
            y_batch = y_batch.to(device)
            
            # Get predictions from all models
            all_probs = []
            for model in models:
                logits = model(X_spc, X_ga)
                probs = torch.sigmoid(logits)
                all_probs.append(probs)
            
            # Stack predictions [num_models, batch_size, num_classes]
            all_probs = torch.stack(all_probs)
            
            # Calculate mean probabilities across models
            mean_probs = all_probs.mean(dim=0)
            
            # Calculate uncertainties
            uncertainty = calculate_uncertainty(all_probs, method=uncertainty_method)
            
            # Convert to binary predictions
            predictions = (mean_probs > 0.5).float()
            
            # Calculate errors
            errors = torch.abs(predictions - y_batch)
            mean_errors = errors.mean(dim=1)
            
            # Store results
            for i in range(X_spc.size(0)):
                sample_result = {
                    'sample_id': batch_idx * test_loader.batch_size + i,
                    'mean_error': mean_errors[i].item(),
                }
                
                # Add uncertainty metrics
                for key, value in uncertainty.items():
                    if key.startswith('mean_'):
                        sample_result[key] = value[i].item()
                
                # Add per-class ground truth, predictions, and uncertainties
                for j, class_name in enumerate(class_names):
                    sample_result[f'true_{class_name}'] = y_batch[i, j].item()
                    sample_result[f'pred_{class_name}'] = predictions[i, j].item()
                    sample_result[f'prob_{class_name}'] = mean_probs[i, j].item()
                    
                    if 'entropy' in uncertainty:
                        sample_result[f'entropy_{class_name}'] = uncertainty['entropy'][i, j].item()
                    if 'variance' in uncertainty:
                        sample_result[f'variance_{class_name}'] = uncertainty['variance'][i, j].item()
                
                results.append(sample_result)
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"Processed {batch_idx + 1} batches")

    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate correlation between uncertainty and error
    if uncertainty_method == 'entropy' and 'mean_entropy' in results_df.columns:
        correlation = results_df['mean_entropy'].corr(results_df['mean_error'])
    elif uncertainty_method == 'variance' and 'mean_variance' in results_df.columns:
        correlation = results_df['mean_variance'].corr(results_df['mean_error'])
    else:
        correlation = None
    
    if correlation is not None:
        print(f"Correlation between uncertainty and error: {correlation:.4f}")
    
    return results_df

def analyze_ensemble_predictions(models, test_loader, class_names, device='cuda', uncertainty_method='entropy'):
    """
    Run ensemble prediction and calculate uncertainties with careful precision handling.
    """
    if not models:
        print("No models available for prediction.")
        return None
    
    print("Checking model parameters...")
    for i, model in enumerate(models):
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                print(f"  Model {i} parameter {name} has dtype {param.dtype}")
    
    print("Making sure all buffers are also in float32...")
    for i, model in enumerate(models):
        for name, buf in model.named_buffers():
            if buf.dtype != torch.float32 and torch.is_floating_point(buf):
                print(f"  Converting buffer {name} from {buf.dtype} to float32")
                buf.data = buf.data.float()
    
    # Set model to ensure float32 outputs
    for model in models:
        model.to(dtype=torch.float32)
    
    results = []
    
    with torch.no_grad():
        for batch_idx, (X_spc, X_ga, y_batch) in enumerate(test_loader):
            # Here's the critical part - explicitly convert inputs to float32
            # before sending to device to ensure consistency
            X_spc = X_spc.float().to(device)
            X_ga = X_ga.float().to(device)
            y_batch = y_batch.to(device)
            
            # Print debug info for first batch only
            if batch_idx == 0:
                print(f"Input tensor dtypes: X_spc={X_spc.dtype}, X_ga={X_ga.dtype}")
            
            # Get predictions from all models
            all_probs = []
            for model in models:
                try:
                    logits = model(X_spc, X_ga)
                    probs = torch.sigmoid(logits)
                    all_probs.append(probs)
                except RuntimeError as e:
                    print(f"Error in model forward pass: {e}")
                    # Try to debug what's happening
                    print(f"Model input dtypes: {X_spc.dtype}, {X_ga.dtype}")
                    for name, param in model.named_parameters():
                        if "input_proj" in name:
                            print(f"  {name} dtype: {param.dtype}")
                    raise
            
            # Rest of the function remains the same...
            # Stack predictions
            all_probs = torch.stack(all_probs)
            
            # Calculate mean probabilities
            mean_probs = all_probs.mean(dim=0)
            
            # Calculate uncertainties
            uncertainty = calculate_uncertainty(all_probs, method=uncertainty_method)
            
            # Convert to binary predictions
            predictions = (mean_probs > 0.5).float()
            
            # Calculate errors
            errors = torch.abs(predictions - y_batch)
            mean_errors = errors.mean(dim=1)
            
            # Rest of your code...
            # Print progress
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"Processed {batch_idx + 1} batches")
                
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate correlation between uncertainty and error
    if uncertainty_method == 'entropy' and 'mean_entropy' in results_df.columns:
        correlation = results_df['mean_entropy'].corr(results_df['mean_error'])
    elif uncertainty_method == 'variance' and 'mean_variance' in results_df.columns:
        correlation = results_df['mean_variance'].corr(results_df['mean_error'])
    else:
        correlation = None
    
    if correlation is not None:
        print(f"Correlation between uncertainty and error: {correlation:.4f}")
    
    return results_df
# Example usage
if __name__ == "__main__":
    # This is a template script - you need to fill in the specific model class and parameters
    # Replace these imports with your actual model imports
    from MAMBA2_Ensamble_Assymetry_v2 import AsymmetricMemoryEfficientStarClassifier, MultiModalBalancedMultiLabelDataset
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load class names
    with open("Pickles/Updated_List_of_Classes_ubuntu.pkl", "rb") as f:
        class_names = pickle.load(f)
    
    # Load datasets
    print("Loading datasets...")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set batch size and batch accumulation
    batch_size = 128
    batch_limit = int(batch_size / 2.5)
    batch_accumulation = 4  # Accumulate gradients over 4 batches


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

    # Create test dataset and dataloader
    test_dataset = MultiModalBalancedMultiLabelDataset(X_test_spectra, X_test_gaia, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Define model parameters - replace with your actual model parameters
    model_params = {
        "d_model_spectra": 2048,  # Already divisible by 8
        "d_model_gaia": 256,      # Already divisible by 8
        "num_classes": 55,
        "input_dim_spectra": 3647,
        "input_dim_gaia": 18,
        "n_layers": 8,
        "d_state_spectra": 16,
        "d_state_gaia": 16,
        "d_conv": 4,
        "expand": 2,
        "use_cross_attention": True,
        "n_cross_attn_heads": 4
    }
    
    # Load ensemble models
    models = load_ensemble_models(
        ensemble_dir="ensemble_models",
        model_class=AsymmetricMemoryEfficientStarClassifier,
        model_params=model_params,
        device=device
    )
    # After loading models
    if models:
        # Check model dtype
        for param in models[0].parameters():
            print(f"Model is using dtype: {param.dtype}")
            break
    
    # Run ensemble prediction and analysis
    results_df = analyze_ensemble_predictions(
        models=models,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        uncertainty_method='entropy'  # or 'variance'
    )
    
    if results_df is not None:
        # Save results
        results_df.to_csv("ensemble_results.csv", index=False)
        
        # Calculate per-class metrics
        class_metrics_df = calculate_class_metrics(results_df, class_names)
        class_metrics_df.to_csv("class_metrics.csv", index=False)
        
        # Analyze uncertainty distribution
        uncertainty_bins_df = analyze_uncertainty_distribution(
            results_df,
            uncertainty_col='mean_entropy',  # or 'mean_variance'
            error_col='mean_error',
            num_bins=20
        )
        uncertainty_bins_df.to_csv("uncertainty_bins.csv", index=False)
        
        # Create visualizations
        create_visualizations(
            results_df,
            class_metrics_df,
            uncertainty_bins_df,
            uncertainty_col='mean_entropy'  # or 'mean_variance'
        )
        
        print("Analysis complete. Results saved to CSV files.")