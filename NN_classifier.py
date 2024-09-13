import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def NN(df, gpu=True, nn_thichness=50, batch_size = 32, lr = 0.003, epochs=200, patience=20, min_delta=0.001, test_size=0.2):
    # Check for GPU
    if gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print("Running on ", device, " cores")
 

    # Split the data into features and labels
    X = df[:, 1:-1]  # All columns except the last one and first one
    y = df[:, -1]    # The last column
    ypreencode = y

    # Use LabelEncoder instead of OneHotEncoder
    encoder = LabelEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Convert y to a PyTorch tensor and move to the GPU (if available)
    y = torch.tensor(y, dtype=torch.long).to(device)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y.cpu().numpy(), test_size = test_size, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create PyTorch Datasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # Define the neural network model
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, (input_size+nn_thichness)/2)
            self.fc2 = nn.Linear((input_size+nn_thichness)/2, nn_thichness)
            self.fc3 = nn.Linear(nn_thichness, nn_thichness)
            self.fc4 = nn.Linear(nn_thichness, (nn_thichness+output_size)/2)
            self.fc5 = nn.Linear((nn_thichness+output_size)/2, output_size)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.relu(self.fc4(x))
            x = self.fc5(x)
            return x

    # Calculate class weights
    class_counts = np.bincount(y_train_tensor.cpu().numpy())
    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]
    class_weights = torch.tensor(np.clip(class_weights, 0, 100), dtype=torch.float32).to(device)

    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[1]
    output_size = len(np.unique(ypreencode))
    model = NeuralNetwork(input_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    # Early stopping
    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    # Function to calculate accuracy
    def calculate_accuracy(loader, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        return correct / total

    # Training the model with a loading bar, early stopping, and adaptive learning rate
    early_stopping = EarlyStopping(patience, min_delta)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({'Loss': f'{running_loss/len(train_loader):.4f}'})
                pbar.update(1)

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        scheduler.step(val_loss)  # Step the scheduler
        print(f'\nValidation Loss after Epoch {epoch+1}: {val_loss:.4f}')

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Calculate and print the test accuracy after each epoch
        test_accuracy = calculate_accuracy(test_loader, model)
        print(f'Test Accuracy after Epoch {epoch+1}: {test_accuracy:.4f}')

        model.to(device)

    # Make predictions on the test set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            # _, labels = torch.max(y_batch.data, 1)  # Remove this line
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())  # Just use y_batch directly

    # Get the unique labels from your predictions and true labels
    unique_labels = np.unique(all_labels)

    # Print the classification report for accuracy per category
    report = classification_report(all_labels, all_preds, labels=unique_labels, target_names=encoder.classes_[unique_labels])
    print("Classification Report:\n", report)


    # Generate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix
    plt.figure(figsize=(30, 30))
    sns.set(font_scale=1.5)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', vmin=0, vmax=5000, 
                xticklabels=encoder.classes_, yticklabels=encoder.classes_, 
                annot_kws={"size": 15})
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('Actual', fontsize=20)
    plt.title('Confusion Matrix', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    return model