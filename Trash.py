# Load the data
Withbinaries = np.load("FirstData.npy", allow_pickle=True)

# Split the data into features and labels
X = Withbinaries[:, 1:-1]  # All columns except the last one
y = Withbinaries[:, -1]    # The last column
ypreencode = y

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the neural network in PyTorch
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 40)
        self.fc2 = nn.Linear(40, 60)
        self.fc3 = nn.Linear(60, 60)
        self.fc4 = nn.Linear(60, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x

# Model parameters
input_size = X_train.shape[1]
num_classes = len(np.unique(ypreencode))
model = NeuralNet(input_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), "star_classifier_first.pth")


from astroquery.vizier import Vizier

v = Vizier(columns=['_RAJ2000', '_DEJ2000','B-V', 'Vmag', 'Plx'],
           column_filters={"Vmag":">10"}, keywords=["optical", "xry"])  

gaia_catalog_list = Vizier.find_catalogs('GAIA DR3') 
print({k:v.description for k,v in gaia_catalog_list.items()})
coordinates = SkyCoord(135.9, -65.3, unit=("deg", "deg"))

result = Vizier(catalog='I/355/spectra').query_region(coordinates, radius="1d0m")

#result = result.to_pandas()


simbad_data = pd.concat((filtered_result, filtered_result2))
gaia_data = pd.concat((r, r2))
print(simbad_data.shape)
print(gaia_data.shape)

# Convert Gaia source_id to string
gaia_data['source_id'] = gaia_data['source_id'].astype(str)

# Filter SIMBAD data to only include rows where 'ids' contains 'Gaia DR3'
simbad_data['gaia_id'] = simbad_data['ids'].apply(lambda x: next((id for id in x.split('|') if id.startswith('Gaia DR3')), None))

simbad_data['gaia_id'] = simbad_data['gaia_id'].str.lstrip('Gaia DR3')
simbad_data = simbad_data.dropna(subset=['gaia_id'])

# Convert Gaia ID to integer
simbad_data['gaia_id'] = simbad_data['gaia_id'].astype(str)


print(simbad_data['gaia_id'])
# Merge Gaia and SIMBAD data on matching IDs
merged_data = pd.merge(r, simbad_data, left_on='source_id', right_on='gaia_id', how='inner')

# Display the merged data
print(merged_data)

def split_ids_into_chunks(id_string, chunk_size=1000):
    # Split the string into a list of IDs
    id_list = id_string.split(', ')
    
    # Create chunks of the specified size
    chunks = [', '.join(id_list[i:i + chunk_size]) for i in range(0, len(id_list), chunk_size)]
    
    return chunks

# Example usage
GaiaDR3SourceIDs = ', '.join(simbad_data['gaia_id'].astype(str))
chunks = split_ids_into_chunks(GaiaDR3SourceIDs)

# Print the chunks to verify
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")


    
def GetGAIAData(GaiaDR3SourceIDs):
    try:
        dfGaia = pd.DataFrame()
        qry = f"SELECT * FROM gaiadr3.gaia_source gs WHERE gs.source_id in ({GaiaDR3SourceIDs});"
        job = Gaia.launch_job_async(qry)
        tblGaia = job.get_results()
        dfGaia = tblGaia.to_pandas()
        print(dfGaia)
    except Exception as e:
        print(f"An error occurred: {e}")

def split_ids_into_chunks(id_string, chunk_size=10000):
    id_list = id_string.split(', ')
    chunks = [', '.join(id_list[i:i + chunk_size]) for i in range(0, len(id_list), chunk_size)]
    return chunks

# Example usage
GaiaDR3SourceIDs = ', '.join(simbad_data['gaia_id'].astype(str))
chunks = split_ids_into_chunks(GaiaDR3SourceIDs)

# Process each chunk
for chunk in chunks:
    GetGAIAData(chunk)
def GetGAIAData(GaiaDR3SourceIDs):
    # gets the GAIA data for the provided GaiaDR3SourceIDs's
    # and writes to a local CSV
        
    dfGaia = pd.DataFrame()
    
    #job = Gaia.launch_job_async( "select top 100 * from gaiadr2.gaia_source where parallax>0 and parallax_over_error>3. ") # Select `good' parallaxes
    qry = "SELECT * FROM gaiadr3.gaia_source gs WHERE gs.source_id in (" + GaiaDR3SourceIDs + ");"
    
    job = Gaia.launch_job_async( qry )
    tblGaia = job.get_results()       #Astropy table
    dfGaia = tblGaia.to_pandas()      #convert to Pandas dataframe
    print(dfGaia)
    
    #npGAIARecords = dfGaia.to_numpy() #convert to numpy array    
    #lstGAIARecords = [list(x) for x in npGAIARecords]   #convert to List[]
    
    #FileForLocalStorage = FolderForLocalStorage + str(lstGAIARecords[0][2]) + '.csv'  # use SourceID from 1st record
    #dfGaia.to_csv (FileForLocalStorage, index = False, header=True)    
GetGAIAData(simbad_data['gaia_id'])
from astroquery.gaia import Gaia
import pandas as pd

def GetGAIAData(GaiaDR3SourceIDs):
    try:
        dfGaia = pd.DataFrame()
        qry = f"SELECT * FROM gaiadr3.gaia_source gs WHERE gs.source_id in ({GaiaDR3SourceIDs});"
        job = Gaia.launch_job_async(qry)
        tblGaia = job.get_results()
        dfGaia = tblGaia.to_pandas()
        print(dfGaia)
    except Exception as e:
        print(f"An error occurred: {e}")
simbadgaiaid = simbad_data['gaia_id'].str.cat(sep=', ')
GetGAIAData(simbad_data['gaia_id'].str.cat(sep=', '))

Gaia.ROW_LIMIT = -1  # Ensure the default row limit.
coord = SkyCoord(ra=0, dec=90, unit=(u.degree, u.degree), frame='icrs')
j = Gaia.cone_search_async(coord, radius=u.Quantity(7.0, u.deg), columns=("source_id", "ra", "dec", "phot_g_mean_flux", "phot_g_mean_flux_error", "pm", "parallax", "parallax_error", "phot_bp_mean_flux", "phot_bp_mean_flux_error", "phot_rp_mean_flux", "phot_rp_mean_flux_error", "teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper", "logg_gspphot", "logg_gspphot_lower", "logg_gspphot_upper", "mh_gspphot", "mh_gspphot_upper", "mh_gspphot_lower", "bp_rp", "bp_g", "g_rp",    ))  
r = j.get_results()
r.pprint()  
r = r.to_pandas()
coord = SkyCoord(ra=0, dec=-90, unit=(u.degree, u.degree), frame='icrs')
j = Gaia.cone_search_async(coord, radius=u.Quantity(7.0, u.deg), columns=("source_id", "ra", "dec", "phot_g_mean_flux", "phot_g_mean_flux_error", "pm", "parallax", "parallax_error", "phot_bp_mean_flux", "phot_bp_mean_flux_error", "phot_rp_mean_flux", "phot_rp_mean_flux_error", "teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper", "logg_gspphot", "logg_gspphot_lower", "logg_gspphot_upper", "mh_gspphot", "mh_gspphot_upper", "mh_gspphot_lower", "bp_rp", "bp_g", "g_rp",    ))  
r2 = j.get_results()
r2.pprint()  
r2= r2.to_pandas()


import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm  # Progress bar

def load_spectra(file_list):
    spectra_data = []
    min_rows = np.inf  # Initialize as infinite to find the minimum number of rows

    # First pass: determine the minimum number of rows across all spectra
    print("Determining the minimum number of rows across spectra...")
    for file_path in tqdm(file_list, desc="Calculating min rows", unit="file"):
        with fits.open(file_path) as hdul:
            # Access the primary HDU (index 0) and get the first row of data
            spectra = hdul[0].data[0]  # First row of the primary HDU
            min_rows = min(min_rows, len(spectra))

    # Second pass: load and truncate the spectra to the minimum number of rows
    print(f"Loading and truncating spectra to {min_rows} rows...")
    for file_path in tqdm(file_list, desc="Loading spectra", unit="file"):
        with fits.open(file_path) as hdul:
            # Access the first row of the primary HDU and truncate
            spectra = hdul[0].data[0][:min_rows]
            spectra_data.append(spectra)

    # Convert the list of spectra to a NumPy array for easier processing
    return np.array(spectra_data)

# Simulate the file list (replace with actual file paths)
#file_list = ["path_to_spectra1.fits", "path_to_spectra2.fits"]  # Add all your actual file paths here

# Load the truncated spectra with progress bars
spectra_data = load_spectra(file_list)
print(f"Spectra data shape: {spectra_data.shape}")  # (num_files, min_rows)



def load_spectra(file_list):
    spectra_data = []
    min_rows = np.inf  # Initialize as infinite to find the minimum number of rows

    # First pass: determine the minimum number of rows across all spectra
    for file_path in file_list:
        with fits.open(file_path) as hdul:
            spectra = hdul[1].data[0]  # Assuming 'flux' column contains the spectra
            min_rows = min(min_rows, len(spectra))

    # Second pass: load and truncate the spectra to the minimum number of rows
    for file_path in file_list:
        with fits.open(file_path) as hdul:
            spectra = hdul[1].data[0][:min_rows]  # Truncate to the min number of rows
            spectra_data.append(spectra)
    
    # Convert the list of spectra to a NumPy array for easier processing
    return np.array(spectra_data)

# Load the truncated spectra
spectra_data = load_spectra(file_list)
print(f"Spectra data shape: {spectra_data.shape}")  # (num_files, min_rows)

# Define the directories containing your spectra
spectra_dirs = {
    "gal_spectra": 0,  # Label 0 for galaxies
    "star_spectra": 1,  # Label 1 for stars
    "agn_spectra": 2,   # Label 2 for AGNs
    "bin_spectra": 3    # Label 3 for binary stars
}

file_list = []
labels = []

# Iterate over the directories and assign labels based on the directory name
for dir_name, label in spectra_dirs.items():
    dir_path = os.path.join(os.getcwd(), dir_name)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            #if file.endswith(".fits"):
            file_path = os.path.join(root, file)
            file_list.append(file_path)
            labels.append(label)

print(f"Total spectra files collected: {len(file_list)}")


quantity_support()  # for getting units on the axes below  

f = fits.open('gal_spectra/110033')  
# The spectrum is in the second HDU of this file.
specdata = f[0].data 
specdata = specdata[0]  # The spectrum is in the first row of the data array.
f.close() 

# Load the crossmatched data
gal_lamost_data = pd.read_pickle("gal_lamost_data.pkl")
obsid_list = gal_lamost_data['obsid'].values



# OLD  Custom Dataset for Spectra
class SpectraDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # Assuming each file is a FITS file containing the spectra
        with fits.open(file_path) as hdul:
            spectra_data = hdul[1].data['flux']  # Assuming 'flux' is the field containing the spectra
        
        label = self.labels[idx]
        
        # Convert spectra to torch tensor
        spectra_tensor = torch.tensor(spectra_data, dtype=torch.float32)
        
        if self.transform:
            spectra_tensor = self.transform(spectra_tensor)
        
        return spectra_tensor, label

# Assume file_list contains paths to your downloaded FITS files and labels contains the corresponding labels
file_list = ["path_to_spectrum1.fits", "path_to_spectrum2.fits", ...]
labels = [0, 1, 2, 3]  # Corresponding to stars, binary stars, non-active galaxies, AGNs


# OLDimport torch.nn as nn
import torch.nn.functional as F

class SpectraCNN(nn.Module):
    def __init__(self):
        super(SpectraCNN, self).__init__()
        
        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * 256, 128)  # Assuming input spectra length of 256
        self.fc2 = nn.Linear(128, 4)  # 4 output classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, kernel_size=2)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from astropy.io import fits

def load_all_spectra(file_list):
    """Load and normalize all spectra files."""
    spectra_data = []
    for file_path in tqdm(file_list, desc="Loading spectra", unit="file"):
        try:
            with fits.open(file_path) as hdul:
                spectra = hdul[0].data[0]
                normalized_spectra = normalize_spectra(spectra)
                spectra_data.append(normalized_spectra)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return np.array(spectra_data)

def normalize_spectra(spectra):
    """Normalize spectra by dividing by the mean and applying the natural logarithm."""
    mean_value = np.mean(spectra)
    if mean_value == 0:
        return spectra  # Avoid division by zero
    normalized_spectra = np.log1p(spectra / mean_value)  # Use log1p for numerical stability (log(1 + x))
    return normalized_spectra

# Example usage:
file_list, labels = generate_file_list()

# Load spectra and create datasets from the loaded data
spectra_data = load_all_spectra(file_list)  # Load all spectra into memory
labels = np.array(labels)

# Create TensorFlow datasets using the loaded data
def create_tf_dataset(spectra_data, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((spectra_data, labels))
    dataset = dataset.shuffle(buffer_size=len(spectra_data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_files, train_labels, val_files, val_labels = split_dataset(file_list, labels)

# Load the spectra for training and validation sets
train_spectra = load_all_spectra(train_files)
val_spectra = load_all_spectra(val_files)

# Create TensorFlow datasets
train_dataset = create_tf_dataset(train_spectra, np.array(train_labels))
val_dataset = create_tf_dataset(val_spectra, np.array(val_labels))

# Now use these datasets to train the model
convnet_model = create_convnet(input_shape=(train_spectra.shape[1], 1), num_classes=len(set(labels)))
history = train_convnet(convnet_model, train_dataset, val_dataset, epochs=20, batch_size=32)

import os
import numpy as np
import tensorflow as tf
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def load_single_spectrum(file_path):
    """Load and normalize a single spectrum from a FITS file."""
    try:
        with fits.open(file_path) as hdul:
            spectra = hdul[0].data[0]
            return normalize_spectra(spectra)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None  # Return None if there's an error

def load_all_spectra_parallel(file_list, max_workers=100):
    """Load and normalize spectra in parallel using ThreadPoolExecutor."""
    spectra_data = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use ThreadPoolExecutor to parallelize the loading of FITS files
        results = list(tqdm(executor.map(load_single_spectrum, file_list), 
                            total=len(file_list), desc="Loading spectra"))

    # Filter out None results (in case any files failed to load)
    spectra_data = [spectrum for spectrum in results if spectrum is not None]

    return np.array(spectra_data)

def normalize_spectra(spectra):
    """Normalize spectra by dividing by the mean and applying the natural logarithm."""
    mean_value = np.mean(spectra)
    if mean_value == 0:
        return spectra  # Avoid division by zero
    normalized_spectra = np.log1p(spectra / mean_value)  # Use log1p for numerical stability (log(1 + x))
    return normalized_spectra

# Example usage:
file_list, labels = generate_file_list()

# Load spectra data in parallel using multiple threads
spectra_data = load_all_spectra_parallel(file_list, max_workers=100)

# Convert labels to numpy array
labels = np.array(labels)

# Create TensorFlow datasets using the loaded data
def create_tf_dataset(spectra_data, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((spectra_data, labels))
    dataset = dataset.shuffle(buffer_size=len(spectra_data)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Split the dataset
train_files, train_labels, val_files, val_labels = split_dataset(file_list, labels)

# Load training and validation spectra in parallel
train_spectra = load_all_spectra_parallel(train_files, max_workers=100)
val_spectra = load_all_spectra_parallel(val_files, max_workers=100)

# Create TensorFlow datasets
train_dataset = create_tf_dataset(train_spectra, np.array(train_labels))
val_dataset = create_tf_dataset(val_spectra, np.array(val_labels))

# Now use these datasets to train the model
convnet_model = create_convnet(input_shape=(train_spectra.shape[1], 1), num_classes=len(set(labels)))
history = train_convnet(convnet_model, train_dataset, val_dataset, epochs=20, batch_size=32)

# Function to train the model with the training and validation datasets
def train_convnet(model, train_dataset, val_dataset, epochs=10, batch_size=32):
    # Fit the model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        batch_size=batch_size)
    
    return history

    import tensorflow as tf


# Function to train the model with the training and validation datasets
def train_convnet(model, train_dataset, val_dataset, epochs=10, batch_size=32):
    # Fit the model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        batch_size=batch_size)
    
    return history

# Determine input shape and number of classes from the dataset
sample_spectra = load_spectra([train_files[0]])  # Load one sample to get input shape
input_shape = (sample_spectra.shape[1], 1)  # Assuming 1D spectra, with shape (length, channels)
num_classes = len(set(labels))  # Assuming 'labels' are numerical categories

# Create a CNN model with custom parameters
convnet_model = create_convnet(input_shape=input_shape, num_classes=num_classes, 
                               num_filters=[32, 64], 
                               kernel_size=(3,), 
                               dense_units=128, 
                               dropout_rate=0.5)

# Train the model using the training and validation datasets
history = train_convnet(convnet_model, train_dataset, val_dataset, epochs=20, batch_size=32)

# You can now access metrics such as accuracy, loss, and validation loss from the `history` object.

# Example usage: Generate a file_list and load the spectra
def generate_file_list():
    # Define the directories containing your spectra
    spectra_dirs = {
        "gal_spectra": 0,  # Label 0 for galaxies
        "star_spectra": 1,  # Label 1 for stars
        "agn_spectra": 2,   # Label 2 for AGNs
        "bin_spectra": 3    # Label 3 for binary stars
    }

    file_list = []
    labels = []

    # Iterate over the directories and assign labels based on the directory name
    print("Gathering FITS files...")
    for dir_name, label in spectra_dirs.items():
        dir_path = os.path.join(os.getcwd(), dir_name)
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                #if file.endswith(".fits"):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
                labels.append(label)

    print(f"Total spectra files collected: {len(file_list)}")
    return file_list, labels

def load_spectra(file_list, known_rows=None):
    spectra_data = []

    # First pass: determine the minimum number of rows across all spectra
    print("Determining minimum number of rows across spectra...")
    if known_rows is None:
        known_rows = np.inf
        for file_path in tqdm(file_list, desc="Finding min rows", unit="file"):
            try:
                with fits.open(file_path) as hdul:
                    # Access the primary HDU (index 0) and the first row of data
                    spectra = hdul[0].data[0]
                    known_rows = min(known_rows, len(spectra))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Second pass: load and truncate spectra to the minimum number of rows
    print(f"\nLoading spectra (truncated to {known_rows} rows)...")
    for file_path in tqdm(file_list, desc="Loading spectra", unit="file"):
        try:
            with fits.open(file_path) as hdul:
                # Access the first row of the primary HDU and truncate
                spectra = hdul[0].data[0][:known_rows]
                spectra_data.append(spectra)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Convert the list of spectra to a NumPy array for easier processing
    spectra_data = np.array(spectra_data)
    return spectra_datac

    # Load the spectra and monitor progress
file_list, labels = generate_file_list()
spectra_data = load_spectra(file_list, known_rows=3748)





import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt

def generate_file_list():
    spectra_dirs = {
        "gal_spectra": 0,  # Label 0 for galaxies
        "star_spectra": 1,  # Label 1 for stars
        "agn_spectra": 2,   # Label 2 for AGNs
        "bin_spectra": 3    # Label 3 for binary stars
    }

    file_list = []
    labels = []

    print("Gathering FITS files...")
    for dir_name, label in spectra_dirs.items():
        dir_path = os.path.join(os.getcwd(), dir_name)
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
                labels.append(label)

    print(f"Total spectra files collected: {len(file_list)}")
    return file_list, labels

def load_spectra(file_list, known_rows=None):
    spectra_data = []
    if known_rows is None:
        known_rows = np.inf
        for file_path in tqdm(file_list, desc="Finding min rows", unit="file"):
            try:
                with fits.open(file_path) as hdul:
                    spectra = hdul[0].data[0]
                    known_rows = min(known_rows, len(spectra))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"\nLoading spectra (truncated to {known_rows} rows)...")
    for file_path in tqdm(file_list, desc="Loading spectra", unit="file"):
        try:
            with fits.open(file_path) as hdul:
                spectra = hdul[0].data[0][:known_rows]
                spectra_data.append(spectra)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    spectra_data = np.array(spectra_data)
    return spectra_data

def create_dataset(file_list, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))

    # Load and parse the FITS files
    def load_and_parse(file_path, label):
        # Load spectra from FITS file
        spectra = tf.py_function(load_spectra, [file_path], tf.float32)
        return spectra, label

    # Apply the loading function to the dataset
    dataset = dataset.map(load_and_parse, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch the data
    dataset = dataset.shuffle(buffer_size=len(file_list)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def split_dataset(file_list, labels, val_split=0.2):
    total_size = len(file_list)
    val_size = int(val_split * total_size)
    
    # Shuffle the data before splitting
    indices = np.random.permutation(total_size)
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    train_files = [file_list[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_files = [file_list[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_files, train_labels, val_files, val_labels

# New function to plot sample spectra
def plot_sample_spectra(file_list, num_samples=2):
    for i in range(min(num_samples, len(file_list))):
        try:
            with fits.open(file_list[i]) as hdul:
                spectra = hdul[0].data[0]
                plt.figure(figsize=(10, 6))
                plt.plot(spectra, label=f'Spectrum {i+1}')
                plt.xlabel('Wavelength')
                plt.ylabel('Intensity')
                plt.title('Sample Spectra')
                plt.legend()
                plt.show()
        except Exception as e:
            print(f"Error reading {file_list[i]}: {e}")
    

# Example usage
file_list, labels = generate_file_list()

# Split the dataset
train_files, train_labels, val_files, val_labels = split_dataset(file_list, labels)

# Create TensorFlow datasets
train_dataset = create_dataset(train_files, train_labels)
val_dataset = create_dataset(val_files, val_labels)


# New function to plot the spectra
def plot_sample_spectra(file_list, num_samples=5):
    plt.figure(figsize=(10, 6))
    for i in range(min(num_samples, len(file_list))):
        try:
            with fits.open(file_list[i]) as hdul:
                spectra = hdul[0].data[0]
                plt.plot(spectra, label=f'Spectrum {i+1}')
        except Exception as e:
            print(f"Error reading {file_list[i]}: {e}")
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title('Sample Spectra')
    plt.legend()
    plt.show()
plot_spectra(train_files)

import tensorflow as tf
import os
import numpy as np

from tqdm import tqdm
from astropy.io import fits

def generate_file_list():
    spectra_dirs = {
        "gal_spectra": 0,  # Label 0 for galaxies
        "star_spectra": 1,  # Label 1 for stars
        "agn_spectra": 2,   # Label 2 for AGNs
        "bin_spectra": 3    # Label 3 for binary stars
    }

    file_list = []
    labels = []

    print("Gathering FITS files...")
    for dir_name, label in spectra_dirs.items():
        dir_path = os.path.join(os.getcwd(), dir_name)
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
                labels.append(label)

    print(f"Total spectra files collected: {len(file_list)}")
    return file_list, labels

def load_spectra(file_list, known_rows=None):
    spectra_data = []
    if known_rows is None:
        known_rows = np.inf
        for file_path in tqdm(file_list, desc="Finding min rows", unit="file"):
            try:
                with fits.open(file_path) as hdul:
                    spectra = hdul[0].data[0]
                    known_rows = min(known_rows, len(spectra))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"\nLoading spectra (truncated to {known_rows} rows)...")
    for file_path in tqdm(file_list, desc="Loading spectra", unit="file"):
        try:
            with fits.open(file_path) as hdul:
                spectra = hdul[0].data[0][:known_rows]
                spectra_data.append(spectra)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    spectra_data = np.array(spectra_data)
    return spectra_data

def create_dataset(file_list, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))

    # Load and parse the FITS files
    def load_and_parse(file_path, label):
        # Load spectra from FITS file
        spectra = tf.py_function(load_spectra, [file_path], tf.float32)
        return spectra, label

    # Apply the loading function to the dataset
    dataset = dataset.map(load_and_parse, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch the data
    dataset = dataset.shuffle(buffer_size=len(file_list)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def split_dataset(file_list, labels, val_split=0.2):
    total_size = len(file_list)
    val_size = int(val_split * total_size)
    
    # Shuffle the data before splitting
    indices = np.random.permutation(total_size)
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    train_files = [file_list[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_files = [file_list[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_files, train_labels, val_files, val_labels

# Example usage
file_list, labels = generate_file_list()

# Split the dataset
train_files, train_labels, val_files, val_labels = split_dataset(file_list, labels)

# Create TensorFlow datasets
train_dataset = create_dataset(train_files, train_labels)
val_dataset = create_dataset(val_files, val_labels)


# Function to train the model with the training and validation datasets
def train_convnet(model, limit_per_label=2000, epochs=1, batch_size=32, patience=5):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    # Use the updated function that loads datasets from pre-separated directories
    train_dataset, val_dataset = generate_datasets_from_preseparated(limit_per_dir=limit_per_label)[0:2]
    
    # Fit the model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping])
    
    return history


def train_convnet_many_times(model, epochs_per_run=1, batch_size=32, num_runs=10, limit_per_label=2000):
    histories = []
    for i in range(num_runs):
        print(f"Training run {i+1}/{num_runs}...")
        # Use the updated train_convnet function
        history = train_convnet(model, limit_per_label=limit_per_label, epochs=epochs_per_run, batch_size=batch_size)
        histories.append(history)
    
    return histories


def load_single_spectrum(file_path, target_length=3748):
    """Load and normalize a single spectrum from a FITS file, truncating or padding to target_length."""
    try:
        with fits.open(file_path) as hdul:
            spectra = hdul[0].data[0]
            spectra = normalize_spectra(spectra)
            
            # Truncate or pad spectra to ensure uniform length
            if len(spectra) > target_length:
                spectra = spectra[:target_length]  # Truncate
            else:
                spectra = np.pad(spectra, (0, max(0, target_length - len(spectra))), mode='constant')  # Pad with zeros
            
            return spectra
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None  # Return None if there's an error


def load_all_spectra_parallel(file_list, target_length=3748, max_workers_=512):
    """Load and normalize spectra in parallel using ThreadPoolExecutor."""
    spectra_data = []
    
    with ThreadPoolExecutor(max_workers=max_workers_) as executor:
        # Use ThreadPoolExecutor to parallelize the loading of FITS files
        results = list(tqdm(executor.map(lambda f: load_single_spectrum(f, target_length), file_list), 
                            total=len(file_list), desc="Loading spectra"))

    # Filter out None results (in case any files failed to load)
    spectra_data = [spectrum for spectrum in results if spectrum is not None]

    return np.array(spectra_data)

len_ = 3748 # Length of the spectra data


def generate_file_list(limit_per_dir = 10000):
    spectra_dirs = {
        "gal_spectra": 0,  # Label 0 for galaxies
        "star_spectra": 1,  # Label 1 for stars
        "agn_spectra": 2,   # Label 2 for AGNs
        "bin_spectra": 3    # Label 3 for binary stars
    }

    file_list = []
    labels = []

    print("Gathering FITS files...")
    for dir_name, label in spectra_dirs.items():
        dir_path = os.path.join(os.getcwd(), dir_name)
        dir_files = []

        # Collect all files in the directory
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                dir_files.append(file_path)
        
        # Randomly select files up to the limit
        if len(dir_files) > limit_per_dir:
            selected_files = random.sample(dir_files, limit_per_dir)
        else:
            selected_files = dir_files
        
        # Append selected files and their labels
        file_list.extend(selected_files)
        labels.extend([label] * len(selected_files))

    print(f"Total spectra files collected: {len(file_list)}")
    return file_list, labels


def load_spectra(file_list, known_rows=None):
    spectra_data = []
    if known_rows is None:
        known_rows = np.inf
        for file_path in tqdm(file_list, desc="Finding min rows", unit="file"):
            try:
                with fits.open(file_path) as hdul:
                    spectra = hdul[0].data[0]
                    known_rows = min(known_rows, len(spectra))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"\nLoading spectra (truncated to {known_rows} rows)...")
    for file_path in tqdm(file_list, desc="Loading spectra", unit="file"):
        try:
            with fits.open(file_path) as hdul:
                spectra = hdul[0].data[0][:known_rows]
                normalized_spectra = normalize_spectra(spectra)
                spectra_data.append(normalized_spectra)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    spectra_data = np.array(spectra_data)
    return spectra_data


def create_dataset(file_list, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))

    def load_and_parse(file_path, label):
        spectra = tf.py_function(load_spectra, [file_path], tf.float32)
        return spectra, label

    dataset = dataset.map(load_and_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(file_list)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def split_dataset(file_list, labels, val_split=0.2):
    total_size = len(file_list)
    val_size = int(val_split * total_size)
    
    indices = np.random.permutation(total_size)
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    train_files = [file_list[i] for i in train_indices]
    print(train_files)
    train_labels = [labels[i] for i in train_indices]
    val_files = [file_list[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    return train_files, train_labels, val_files, val_labels


def normalize_spectra(spectra):
    """Normalize spectra by dividing by the mean and applying the natural logarithm."""
    mean_value = np.mean(spectra)
    std_value = np.std(spectra)
    min_value = np.min(spectra)
    if std_value == 0:
        print("Warning: Standard deviation is zero, cannot normalize spectra.")
        return spectra  # Avoid division by zero
    normalized_spectra = ((spectra - min_value  + 0.01) / (mean_value - min_value + 0.01)) - 1 # min_value is added to avoid negative values
    return normalized_spectra


def load_single_spectrum(file_path, target_lenth=3748):
    """Load and normalize a single spectrum from a FITS file, truncating or padding to target_length."""
    try:
        with fits.open(file_path) as hdul:
            spectra = hdul[0].data[0]
            spectra = normalize_spectra(spectra)
            
            # Truncate or pad spectra to ensure uniform length
            if len(spectra) > target_length:
                spectra = spectra[:target_length]  # Truncate
            else:
                spectra = np.pad(spectra, (0, max(0, target_length - len(spectra))), mode='constant')  # Pad with zeros
            
            return spectra
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None  # Return None if there's an error


def load_all_spectra_parallel(file_list, target_length=3748, max_workers_=512):
    """Load and normalize spectra in parallel using ThreadPoolExecutor."""
    spectra_data = []
    
    with ThreadPoolExecutor(max_workers=max_workers_) as executor:
        # Use ThreadPoolExecutor to parallelize the loading of FITS files
        results = list(tqdm(executor.map(lambda f: load_single_spectrum(f, target_length), file_list), 
                            total=len(file_list), desc="Loading spectra"))

    # Filter out None results (in case any files failed to load)
    spectra_data = [spectrum for spectrum in results if spectrum is not None]

    return np.array(spectra_data)


def generate_random_dataset(lim_per_label = 2000):
    # Example usage:
    file_list, labels = generate_file_list(limit_per_dir=lim_per_label)
    # Convert labels to numpy array
    labels = np.array(labels)
    # Continue with creating train/validation datasets
    train_files, train_labels, val_files, val_labels = split_dataset(file_list, labels)
    # Load training and validation spectra in parallel
    train_spectra = load_all_spectra_parallel(train_files, target_length=len_)
    val_spectra = load_all_spectra_parallel(val_files, target_length=len_)
    # Create TensorFlow datasets
    train_dataset, val_dataset = removenan(train_spectra, train_labels, val_spectra, val_labels)
    return train_dataset, val_dataset

# Specify the directory
directory = 'gaia_training_set/gal_data'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Loop through the DataFrame and save each row as a .npy file
for index, row in matched_data.iterrows():
    # Extract the filename
    filename = row['obsid']
    
    # Select the columns you want to save
    values_to_save = row[['ra','ra_error', 'dec', 'dec_error', 'phot_g_mean_flux', 'phot_g_mean_flux_error', 'phot_bp_mean_flux', 'phot_bp_mean_flux_error', 'phot_rp_mean_flux', 'phot_rp_mean_flux_error', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error']].values
    
    # Save the values as a .npy file in the specified directory
    np.save(os.path.join(directory, f"{filename}.npy"), values_to_save)

    
        # Step 3: Create a new frequency space (1000 parameters) for interpolation
        new_frequencies = np.linspace(frequencies.min(), frequencies.max(), 1000)

        # Step 4: Interpolate the FFT magnitudes into the new frequency space
        interpolator = interp1d(frequencies, fft_magnitude, kind='linear', fill_value="extrapolate")
        interpolated_fft_magnitude = interpolator(new_frequencies)

        # Step 5: Store the interpolated data along with labels and other metadata
        # Create a dictionary for the interpolated spectrum
        interpolated_data = {f'fft_mag_{i}': value for i, value in enumerate(interpolated_fft_magnitude)}

        # Add the original metadata back (e.g., file_name, label, row)
        interpolated_data['file_name'] = row['file_name']
        interpolated_data['label'] = row['label']
        interpolated_data['row'] = row['row']

        # Append the interpolated data to the results list
        results_list.append(interpolated_data)
    
    import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.interpolate import interp1d

# Load the data from the pickle file
df = pd.read_pickle('Pickles/lmst/flux_bin_df.pkl')  # Load the flux data
df_freq = pd.read_pickle('Pickles/lmst/freq_bin_df.pkl')  # Load the corresponding original frequency data

# Set directory to save the interpolated data
output_dir = 'Pickles/lmst/interpolated_data/train_bin.pkl'

# Initialize an empty list to store the results before concatenating into a DataFrame
results_list = []

# Loop through each row in the DataFrame (each row is a spectrum)
for index, row in df.iterrows():
    # Extract the fluxes (assuming they start at column 0 and continue to the last column)
    fluxes = row[:-3].values  # Exclude the last columns (file_name, label, row)
    print(index)
    # Extract the frequencies (if they are constant, you can define it once outside the loop)
    frequencies = df_freq.iloc[int(index/3), :-3].values  # Exclude the last columns (file_name, label, row)
    #frequencies = np.linspace(frequencies[0], frequencies[-1], len(fluxes))  # Assuming linearly spaced frequencies
    # Step 1: Perform DFT on the flux data
    flux_fft = fft(fluxes)

    # Step 2: Get the magnitude of the FFT (absolute values)
    fft_magnitude = np.abs(flux_fft)
    
    # Step 3: Create a new frequency space (1000 parameters) for interpolation
    new_frequencies = np.linspace(frequencies.min(), frequencies.max(), 1000)
    
    # Step 4: Interpolate the FFT magnitudes into the new frequency space
    interpolator = interp1d(frequencies, fft_magnitude, kind='linear', fill_value="extrapolate")
    interpolated_fft_magnitude = interpolator(new_frequencies)
    
    # Step 5: Store the interpolated data along with labels and other metadata
    # Create a dictionary for the interpolated spectrum
    interpolated_data = {f'fft_mag_{i}': value for i, value in enumerate(interpolated_fft_magnitude)}
    
    # Add the original metadata back (e.g., file_name, label, row)
    interpolated_data['file_name'] = row['file_name']
    interpolated_data['label'] = row['label']
    interpolated_data['row'] = row['row']
    
    # Append the interpolated data to the results list
    results_list.append(interpolated_data)

# Step 6: Convert the list of results into a DataFrame
interpolated_spectra_df = pd.DataFrame(results_list)

# Save the resulting DataFrame to a new pickle file
interpolated_spectra_df.to_pickle(output_dir)

# Optionally, save to CSV or other formats
# interpolated_spectra_df.to_csv('interpolated_spectra_data.csv', index=False)
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.interpolate import interp1d
from tqdm import tqdm

def interpolate_spectrum(fluxes_loc, frequencies_loc, output_dir):
    """Interpolates the flux values to fill in missing data points."""
    # Load the data from the pickle file
    df_flux = pd.read_pickle(fluxes_loc).reset_index(drop=True)  # Reset index for zero-based iteration
    df_freq = pd.read_pickle(frequencies_loc).reset_index(drop=True)  # Same for df_freq


    # Initialize an empty list to store the results before concatenating into a DataFrame
    results_list = []


    # Loop through each row in the DataFrame (each row is a spectrum) with tqdm for progress bar
    for index, row in tqdm(df_flux.iterrows(), total=len(df_flux), desc='Interpolating spectra'):
        # Extract the fluxes (assuming they start at column 0 and continue to the last column)
        fluxes = row[:-3].values  # Exclude the last columns (file_name, label, row)


        # Extract the frequencies (if they are constant, you can define it once outside the loop)
        frequencies = df_freq.iloc[int(index/3), :-3].values  # Exclude the last columns (file_name, label, row)

        # Count the number of NaN and 0 values in the fluxes and frequencies
        fluxes = pd.to_numeric(row[:-3], errors='coerce').values  # Exclude and convert to numeric
        frequencies = pd.to_numeric(df_freq.iloc[index, :-3], errors='coerce').values  # Same for frequencies
        num_nan = np.isnan(fluxes).sum() + np.isnan(frequencies).sum() # Count NaN values
        num_zero = (fluxes == 0).sum() + (frequencies == 0).sum()  # Count zero values

        # Initialize lists to store problematic file_names
        nan_files = []

        # Special handling for NaN values, counting nans in sequence
        if num_nan > 10:
            num_nan_seq = 0
            for i in range(len(fluxes)):
                if np.isnan(fluxes[i]) or np.isnan(frequencies[i]):
                    num_nan_seq += 1
                else:
                    num_nan_seq = 0
                if num_nan_seq > 10:
                    print(f"File {row['file_name']} has more than 10 NaN values in sequence.")
                    nan_files.append(row['file_name'])
                    break
            continue
        if num_zero > 10:
            num_zero_seq = 0
            for i in range(len(fluxes)):
                if fluxes[i] == 0 or frequencies[i] == 0:
                    num_zero_seq += 1
                else:
                    num_zero_seq = 0
                if num_zero_seq > 10:
                    print(f"File {row['file_name']} has more than 10 zero values in sequence.")
                    nan_files.append(row['file_name'])
                    break
            continue

        # Deal with NaN values
        fluxes = fluxes[~np.isnan(fluxes)]
        frequencies = frequencies[~np.isnan(fluxes)]

        # Interpolate to fill in missing values
        f = interp1d(frequencies, fluxes, kind='linear', fill_value="extrapolate")
        new_frequencies = np.linspace(frequencies.min(), frequencies.max(), len(row[:-3].values))

        # Interpolated flux values
        interpolated_fluxes = f(new_frequencies)

        # Store the interpolated data along with labels and other metadata
        # Create a dictionary for the interpolated spectrum
        interpolated_data = {f'flux_{i}': value for i, value in enumerate(interpolated_fluxes)}

        # Add the original metadata back (e.g., file_name, label, row)
        interpolated_data['file_name'] = row['file_name']
        interpolated_data['label'] = row['label']
        #interpolated_data['row'] = row['row']

        # Append the interpolated data to the results list
        results_list.append(interpolated_data)

        if int(index/3) % 100 == 0:  # Save every 100 rows
            pd.DataFrame(results_list).to_pickle(output_dir, mode='a')  # Save in append mode
            results_list = []  # Clear list to free memory


    # Convert the list of results into a DataFrame
    #interpolated_spectra_df = pd.DataFrame(results_list)

    # Save the resulting DataFrame to a new pickle file
    #interpolated_spectra_df.to_pickle(output_dir)

    # Return the list of files with NaN values
    return nan_files




import os

def interpolate_spectrum(fluxes_loc, frequencies_loc, output_dir, limit=10):
    """Interpolates the flux values to fill in missing data points."""
    # Load the data from the pickle file
    df_flux = pd.read_pickle(fluxes_loc).reset_index(drop=True)  # Reset index for zero-based iteration
    df_freq = pd.read_pickle(frequencies_loc).reset_index(drop=True)  # Same for df_freq

    # Initialize an empty list to store the results before concatenating into a DataFrame
    results_list = []

    # Initialize lists to store problematic file_names
    nan_files = []

    # Overwrite the output file at the beginning
    if os.path.exists(output_dir):
        os.remove(output_dir)

    # Loop through each row in the DataFrame (each row is a spectrum) with tqdm for progress bar
    for index, row in tqdm(df_flux.iterrows(), total=len(df_flux), desc='Interpolating spectra'):

    # Extract the fluxes (assuming they start at column 0 and continue to the last column)
        fluxes = row[:-3].values  # Exclude the last columns (file_name, label, row)


        # Extract the frequencies (if they are constant, you can define it once outside the loop)
        frequencies = df_freq.iloc[int(index/3), :-3].values  # Exclude the last columns (file_name, label, row)

        # Count the number of NaN and 0 values in the fluxes and frequencies
        fluxes = pd.to_numeric(row[:-3], errors='coerce').values  # Exclude and convert to numeric
        frequencies = pd.to_numeric(df_freq.iloc[index, :-3], errors='coerce').values  # Same for frequencies
        num_nan = np.isnan(fluxes).sum() + np.isnan(frequencies).sum() # Count NaN values
        num_zero = (fluxes == 0).sum() + (frequencies == 0).sum()  # Count zero values



        # Special handling for NaN values, counting nans in sequence, except for the first 10
        if num_nan > limit and index > limit*3 and index < len(df_flux*3)-limit*3:
            num_nan_seq = 0
            for i in range(len(fluxes)):
                if np.isnan(fluxes[i]) or np.isnan(frequencies[i]):
                    num_nan_seq += 1
                else:
                    num_nan_seq = 0
                if num_nan_seq > limit:
                    nan_files.append(row['file_name'])
                    break
            continue
        if num_zero > limit and index > limit*3 and index < len(df_flux*3)-limit*3:
            num_zero_seq = 0
            for i in range(len(fluxes)):
                if fluxes[i] == 0 or frequencies[i] == 0:
                    num_zero_seq += 1
                else:
                    num_zero_seq = 0
                if num_zero_seq > limit:
                    nan_files.append(row['file_name'])
                    break
            continue

        # Deal with NaN values
        fluxes = fluxes[~np.isnan(fluxes)]
        frequencies = frequencies[~np.isnan(fluxes)]

        # Interpolate to fill in missing values
        f = interp1d(frequencies, fluxes, kind='linear', fill_value="extrapolate")
        new_frequencies = np.linspace(frequencies.min(), frequencies.max(), len(row[:-3].values))

        # Interpolated flux values
        interpolated_fluxes = f(new_frequencies)

        # Store the interpolated data along with labels and other metadata
        # Create a dictionary for the interpolated spectrum
        interpolated_data = {f'flux_{i}': value for i, value in enumerate(interpolated_fluxes)}

        # Add the original metadata back (e.g., file_name, label, row)
        interpolated_data['file_name'] = row['file_name']
        interpolated_data['label'] = row['label']
        
        # Append the interpolated data to the results list
        results_list.append(interpolated_data)

        if int(index / 3) % 100 == 0:  # Save every 1000 rows
            # Check if the output file already exists
            if os.path.exists(output_dir):
                existing_df = pd.read_pickle(output_dir)  # Load existing data
                new_df = pd.DataFrame(results_list)
                # Concatenate existing and new data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_pickle(output_dir)  # Save combined DataFrame
            else:
                # If the file doesn't exist, create a new DataFrame and save
                pd.DataFrame(results_list).to_pickle(output_dir)

            #results_list = []  # Clear list to free memory
    print(f"Initial number of rows: {len(df_flux)}")
    print(f"Number of rows after removing NaN/zero values: {len(results_list)}")

    # After the loop, save any remaining results
    if results_list:
        if os.path.exists(output_dir):
            existing_df = pd.read_pickle(output_dir)
            new_df = pd.DataFrame(results_list)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_pickle(output_dir)
        else:
            pd.DataFrame(results_list).to_pickle(output_dir)
    print(f"Number of rows after removing NaN/zero values: {len(results_list)}")
    print(f"Interpolation complete. Number of NaN files: {len(nan_files)}")

    return nan_files
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import gc

def interpolate_spectrum(fluxes_loc, frequencies_loc, output_dir, limit=10, edge_limit=20, chunk_size=1000):
    """Interpolates the flux values to fill in missing data points."""
    # Load the entire data
    df_freq = pd.read_pickle(frequencies_loc).reset_index(drop=True).drop(columns=['row'])
    df_flux = pd.read_pickle(fluxes_loc).reset_index(drop=True).drop(columns=['row'])

    # Initialize lists to store problematic file_names
    nan_files = []  
    cnt_success = 0

    # Overwrite the output file at the beginning
    if os.path.exists(output_dir):
        os.remove(output_dir)

    # Split the DataFrame into chunks
    num_chunks = len(df_flux) // chunk_size + 1

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(df_flux))

        df_flux_chunk = df_flux.iloc[start_idx:end_idx]
        df_freq_chunk = df_freq.iloc[start_idx:end_idx]

        results_list = []

        for relative_index, row in tqdm(df_flux_chunk.iterrows(), total=len(df_flux_chunk), desc=f'Interpolating spectra (chunk {chunk_idx + 1}/{num_chunks})'):
            try:
                # Use absolute index for accessing corresponding frequency row
                abs_index = start_idx + relative_index

                # Fetch the corresponding frequency row based on the absolute index
                fluxes = row[:-2].values
                frequencies = df_freq_chunk.iloc[relative_index, :-2].values

                # Convert to numeric and handle NaNs
                fluxes = pd.to_numeric(row[:-2], errors='coerce').values
                frequencies = pd.to_numeric(df_freq_chunk.iloc[relative_index, :-2], errors='coerce').values

                num_nan = np.isnan(fluxes).sum() + np.isnan(frequencies).sum()
                num_zero = (fluxes == 0).sum() + (frequencies == 0).sum()

                if num_nan > limit or num_zero > limit:
                    nan_files.append(row['file_name'])
                    continue

                # Filter NaNs
                fluxes = fluxes[~np.isnan(fluxes)]
                frequencies = frequencies[~np.isnan(fluxes)]

                # Interpolation
                f = interp1d(frequencies, fluxes, kind='linear', fill_value="extrapolate")
                new_frequencies = np.linspace(frequencies.min(), frequencies.max(), len(row[:-2].values))
                interpolated_fluxes = f(new_frequencies)

                interpolated_data = {f'flux_{i}': value for i, value in enumerate(interpolated_fluxes)}
                interpolated_data['file_name'] = row['file_name']
                interpolated_data['label'] = row['label']
                results_list.append(interpolated_data)

            except Exception as e:
                print(f"Error processing row {abs_index}: {e}")
                continue

        if os.path.exists(output_dir):
            existing_df = pd.read_pickle(output_dir)
            new_df = pd.DataFrame(results_list)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_pickle(output_dir)
        else:
            pd.DataFrame(results_list).to_pickle(output_dir)

        cnt_success += len(results_list)
        results_list = []
        gc.collect()  # Explicitly call garbage collection

    print(f"Total successful interpolations: {cnt_success}")
    print(f"Total skipped due to NaNs or zeros: {len(nan_files)}")

    return nan_files

flux_loc = 'Pickles/lmst/flux_star_df.pkl'
freq_loc = 'Pickles/lmst/freq_star_df.pkl'
output_dir = 'Pickles/lmst/interpolated_data/train_star.pkl'

tstarnan = interpolate_spectrum(flux_loc, freq_loc, output_dir, limit=10)
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

def interpolate_spectrum_in_chunks(fluxes_loc, frequencies_loc, output_dir, chunk_size=5000, limit=10, edge_limit=20):
    """Interpolates the flux values to fill in missing data points, processing the CSV file in chunks."""
    # Initialize lists to store problematic file_names
    nan_files = []  

    # Overwrite the output file at the beginning
    if os.path.exists(output_dir):
        os.remove(output_dir)

    # Open the CSV files in chunks
    flux_chunks = pd.read_csv(fluxes_loc, chunksize=chunk_size)
    freq_chunks = pd.read_csv(frequencies_loc, chunksize=chunk_size)

    # Loop through the chunks simultaneously
    for flux_chunk, freq_chunk in zip(tqdm(flux_chunks, desc="Processing chunks"), freq_chunks):
        # Ensure both chunks have the same number of rows
        if len(flux_chunk) != len(freq_chunk):
            raise ValueError("Flux and frequency chunks have different sizes!")

        # Initialize an empty list to store the results for the current chunk
        results_list = []

        # Loop through each row in the chunk (each row is a spectrum)
        for idx, row in flux_chunk.iterrows():
            # Extract the fluxes (assuming they start at column 0 and continue to the last column)
            fluxes = row[:-2].values  # Exclude the last columns (file_name, label)

            # Extract the corresponding frequencies from the same row in the freq_chunk
            frequencies = freq_chunk.iloc[idx, :-2].values  # Exclude the last columns (file_name, label)

            # Count the number of NaN and 0 values in the fluxes and frequencies
            fluxes = pd.to_numeric(row[:-2], errors='coerce').values  # Exclude and convert to numeric
            frequencies = pd.to_numeric(freq_chunk.iloc[idx, :-2], errors='coerce').values  # Same for frequencies

            num_nan = np.isnan(fluxes).sum() + np.isnan(frequencies).sum()  # Count NaN values
            num_zero = (fluxes == 0).sum() + (frequencies == 0).sum()  # Count zero values

            if num_nan > limit and idx > edge_limit and idx < len(fluxes) - edge_limit:
                nan_files.append(row['file_name'])
                continue

            if num_zero > limit and idx > edge_limit and idx < len(fluxes) - edge_limit:
                nan_files.append(row['file_name'])
                continue

            # Remove NaN values
            fluxes = fluxes[~np.isnan(fluxes)]
            frequencies = frequencies[~np.isnan(frequencies)]

            # Interpolate to fill in missing values
            f = interp1d(frequencies, fluxes, kind='linear', fill_value="extrapolate")
            new_frequencies = np.linspace(frequencies.min(), frequencies.max(), len(row[:-2].values))

            # Interpolated flux values
            interpolated_fluxes = f(new_frequencies)

            # Store the interpolated data along with labels and other metadata
            interpolated_data = {f'flux_{i}': value for i, value in enumerate(interpolated_fluxes)}

            # Add the original metadata back (e.g., file_name, label, row)
            interpolated_data['file_name'] = row['file_name']
            interpolated_data['label'] = row['label']

            # Append the interpolated data to the results list
            results_list.append(interpolated_data)

        # Save the chunk results
        if os.path.exists(output_dir):
            existing_df = pd.read_pickle(output_dir)  # Load existing data
            new_df = pd.DataFrame(results_list)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_pickle(output_dir)  # Save combined DataFrame
        else:
            pd.DataFrame(results_list).to_pickle(output_dir)

    return nan_files
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

def interpolate_spectrum(fluxes_loc, frequencies_loc, output_dir, limit=10):
    """Interpolates the flux values to fill in missing data points."""
    # Load the data from the pickle file
    df_flux = pd.read_pickle(fluxes_loc).reset_index(drop=True)  # Reset index for zero-based iteration
    df_freq = pd.read_pickle(frequencies_loc).reset_index(drop=True)  # Same for df_freq

    # Initialize an empty list to store the results before concatenating into a DataFrame
    results_list = []

    # Initialize lists to store problematic file_names
    nan_files = []  

    # Count the number of successful interpolations
    cnt_sucess = 0

    # Overwrite the output file at the beginning
    if os.path.exists(output_dir):
        os.remove(output_dir)

    # Loop through each row in the DataFrame (each row is a spectrum) with tqdm for progress bar
    for index, row in tqdm(df_flux.iterrows(), total=len(df_flux), desc='Interpolating spectra'):

        # Extract the fluxes (assuming they start at column 0 and continue to the last column)
        fluxes = row[:-3].values  # Exclude the last columns (file_name, label, row)

        # Extract the frequencies
        frequencies = df_freq.iloc[int(index), :-3].values  # Exclude the last columns (file_name, label, row)

        # Count the number of NaN and 0 values in the fluxes and frequencies
        fluxes = pd.to_numeric(row[:-3], errors='coerce').values  # Exclude and convert to numeric
        frequencies = pd.to_numeric(df_freq.iloc[index, :-3], errors='coerce').values  # Same for frequencies
        num_nan = np.isnan(fluxes).sum() + np.isnan(frequencies).sum()  # Count NaN values
        num_zero = (fluxes == 0).sum() + (frequencies == 0).sum()  # Count zero values

        # Special handling for NaN values, counting nans in sequence, except for the first and last 10
        if num_nan > limit and index > limit and index < len(fluxes)-limit:
            num_nan_seq = 0
            for i in range(len(fluxes)):
                if np.isnan(fluxes[i]) or np.isnan(frequencies[i]):
                    num_nan_seq += 1
                else:
                    num_nan_seq = 0
                if num_nan_seq > limit:
                    nan_files.append(row['file_name'])
                    break
            continue
        if num_zero > limit and index > limit and index < len(fluxes)-limit:
            num_zero_seq = 0
            for i in range(len(fluxes)):
                if fluxes[i] == 0 or frequencies[i] == 0:
                    num_zero_seq += 1
                else:
                    num_zero_seq = 0
                if num_zero_seq > limit:
                    nan_files.append(row['file_name'])
                    break
            continue

        # Deal with NaN values
        fluxes = fluxes[~np.isnan(fluxes)]
        frequencies = frequencies[~np.isnan(fluxes)]

        # Interpolate to fill in missing values
        f = interp1d(frequencies, fluxes, kind='linear', fill_value="extrapolate")
        new_frequencies = np.linspace(frequencies.min(), frequencies.max(), len(row[:-3].values))

        # Interpolated flux values
        interpolated_fluxes = f(new_frequencies)

        # Store the interpolated data along with labels and other metadata
        # Create a dictionary for the interpolated spectrum
        interpolated_data = {f'flux_{i}': value for i, value in enumerate(interpolated_fluxes)}

        # Add the original metadata back (e.g., file_name, label, row)
        interpolated_data['file_name'] = row['file_name']
        interpolated_data['label'] = row['label']
        
        # Append the interpolated data to the results list
        results_list.append(interpolated_data)

        if index % 5000 == 0:  # Save every 100 rows
            # Check if the output file already exists
            if os.path.exists(output_dir):
                existing_df = pd.read_pickle(output_dir)  # Load existing data
                new_df = pd.DataFrame(results_list)
                # Concatenate existing and new data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_pickle(output_dir)  # Save combined DataFrame
            else:
                # If the file doesn't exist, create a new DataFrame and save
                pd.DataFrame(results_list).to_pickle(output_dir)
            cnt_sucess += len(results_list)  # Increment the count of successful interpolations
            results_list = []  # Clear list to free memory

    print(f"Initial number of rows: {len(df_flux)}")

    # After the loop, save any remaining results
    if results_list:
        if os.path.exists(output_dir):
            existing_df = pd.read_pickle(output_dir)
            new_df = pd.DataFrame(results_list)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_pickle(output_dir)
        else:
            pd.DataFrame(results_list).to_pickle(output_dir)
        cnt_sucess += len(results_list)
    print(f"Number of rows after removing NaN/zero values: {cnt_sucess}")
    print(f"Interpolation complete. Number of NaN files: {len(nan_files)}")

    return nan_files
import os
import h5py
from astropy.io import fits
import logging
from concurrent.futures import ThreadPoolExecutor
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_fits_to_h5(fits_file, h5_file, target_length=3748):
    """Converts a single FITS file to HDF5 format."""
    try:
        with fits.open(fits_file) as hdul:
            if len(hdul) > 0:  # Check if the Primary HDU exists
                spectra_data = hdul[0].data  # Assuming the spectra data is in the Primary HDU
                if spectra_data is not None:
                    spectra_data = spectra_data[:target_length]  # Trim to target length if necessary
                    
                    # Save to HDF5
                    with h5py.File(h5_file, 'w') as hf:
                        hf.create_dataset('spectra', data=spectra_data)
                else:
                    logging.error(f"{fits_file} does not contain data in the Primary HDU")
            else:
                logging.error(f"{fits_file} does not contain the expected HDU")
    except Exception as e:
        logging.error(f"Error converting {fits_file} to {h5_file}: {e}")

def batch_convert_fits_to_h5(file_list, target_dir, target_length=3748):
    """Convert a batch of FITS files to HDF5 format."""
    os.makedirs(target_dir, exist_ok=True)  # Create target directory if it doesn't exist
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(convert_fits_to_h5, fits_file, os.path.join(target_dir, os.path.splitext(os.path.basename(fits_file))[0] + ".h5"), target_length) for fits_file in file_list]
        for future in futures:
            future.result()  # Wait for all threads to complete

    logging.info(f"All FITS files converted to HDF5 and saved in {target_dir}")

def generate_file_list_from_directories(base_dirs, limit_per_dir=10000):
    """Generates a list of files and labels from the pre-separated directories."""
    spectra_dirs = {
        "gal_spectra": 0,  # Label 0 for galaxies
        "star_spectra": 1,  # Label 1 for stars
        "agn_spectra": 2,   # Label 2 for AGNs
        "bin_spectra": 3    # Label 3 for binary stars
    }

    file_list = []
    labels = []

    logging.info("Gathering FITS files from pre-separated directories...")
    for dir_name, label in spectra_dirs.items():
        for base_dir in base_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            if os.path.exists(dir_path):
                logging.info(f"Checking directory: {dir_path}")
                dir_files = os.listdir(dir_path)
                logging.info(f"Found {len(dir_files)} files in {dir_path}")

                # Collect all files in the directory
                for file in dir_files:
                    file_path = os.path.join(dir_path, file)
                    file_list.append(file_path)

                # Randomly select files up to the limit
                if len(file_list) > limit_per_dir:
                    selected_files = random.sample(file_list, limit_per_dir)
                else:
                    selected_files = file_list

                # Append selected files and their labels
                labels.extend([label] * len(selected_files))

    logging.info(f"Total spectra files collected: {len(file_list)}")
    return file_list, labels

# Convert all FITS files to HDF5
train_files, train_labels = generate_file_list_from_directories(["training_set"], limit_per_dir=10000)
val_files, val_labels = generate_file_list_from_directories(["validation_set"], limit_per_dir=10000)

batch_convert_fits_to_h5(train_files, "training_h5")
batch_convert_fits_to_h5(val_files, "validation_h5")


import os
import numpy as np
from astropy.io import fits
import logging
from concurrent.futures import ThreadPoolExecutor
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_fits_to_npy(fits_file, npy_file, target_length=3748):
    """Converts a single FITS file to NumPy format."""
    try:
        with fits.open(fits_file) as hdul:
            if len(hdul) > 0:  # Check if the Primary HDU exists
                spectra_data = hdul[0].data  # Assuming the spectra data is in the Primary HDU
                if spectra_data is not None:
                    spectra_data = spectra_data[:target_length]  # Trim to target length if necessary
                    
                    # Save to NumPy array
                    np.save(npy_file, spectra_data)
                else:
                    logging.error(f"{fits_file} does not contain data in the Primary HDU")
            else:
                logging.error(f"{fits_file} does not contain the expected HDU")
    except Exception as e:
        logging.error(f"Error converting {fits_file} to {npy_file}: {e}")

def batch_convert_fits_to_npy(file_list, target_dir, target_length=3748):
    """Convert a batch of FITS files to NumPy format."""
    os.makedirs(target_dir, exist_ok=True)  # Create target directory if it doesn't exist
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(convert_fits_to_npy, fits_file, os.path.join(target_dir, os.path.splitext(os.path.basename(fits_file))[0] + ".npy"), target_length) for fits_file in file_list]
        for future in futures:
            future.result()  # Wait for all threads to complete

    logging.info(f"All FITS files converted to NumPy arrays and saved in {target_dir}")

def generate_file_list_from_directories(base_dirs, limit_per_dir=10000):
    """Generates a list of files and labels from the pre-separated directories."""
    spectra_dirs = {
        "gal_spectra": 0,  # Label 0 for galaxies
        "star_spectra": 1,  # Label 1 for stars
        "agn_spectra": 2,   # Label 2 for AGNs
        "bin_spectra": 3    # Label 3 for binary stars
    }

    file_list = []
    labels = []

    logging.info("Gathering FITS files from pre-separated directories...")
    for dir_name, label in spectra_dirs.items():
        for base_dir in base_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            if os.path.exists(dir_path):
                logging.info(f"Checking directory: {dir_path}")
                dir_files = os.listdir(dir_path)
                logging.info(f"Found {len(dir_files)} files in {dir_path}")

                # Collect all files in the directory
                for file in dir_files:
                    file_path = os.path.join(dir_path, file)
                    file_list.append(file_path)

                # Randomly select files up to the limit
                if len(file_list) > limit_per_dir:
                    selected_files = random.sample(file_list, limit_per_dir)
                else:
                    selected_files = file_list

                # Append selected files and their labels
                labels.extend([label] * len(selected_files))

    logging.info(f"Total spectra files collected: {len(file_list)}")
    return file_list, labels

# Convert all FITS files to NumPy arrays
train_files, train_labels = generate_file_list_from_directories(["training_set/bin_spectra"], limit_per_dir=10000)
val_files, val_labels = generate_file_list_from_directories(["validation_set/bin_spectra"], limit_per_dir=10000)

batch_convert_fits_to_npy(train_files, "training_npy/bin_spectra")
batch_convert_fits_to_npy(val_files, "validation_npy/bin_spectra")
# open the pkl file with the list of files to drop
nan_gaia = pd.read_pickle("Pickles/drops/gaiaall.pkl")
num_to_remove = len(nan_gaia)
num_removed = 0


# Training dataset
# GALAXIES
df = pd.read_pickle('Pickles/lmst/interpolated_data/val_gal.pkl')
lenghtpre = len(df)
df = df[~df['file_name'].isin(nan_gaia)]    
print(f"GALAXIES: Number of rows before dropping: {lenghtpre} and after dropping: {len(df)}")
num_removed += lenghtpre - len(df)
df.to_pickle('Pickles/lmst/interpolated_data/val_gal2.pkl')

# BINARY STARS
df = pd.read_pickle('Pickles/lmst/interpolated_data/val_bin.pkl')
lenghtpre = len(df)
df = df[~df['file_name'].isin(nan_gaia)]
print(f"BINARY STARS: Number of rows before dropping: {lenghtpre} and after dropping: {len(df)}")
num_removed += lenghtpre - len(df)
df.to_pickle('Pickles/lmst/interpolated_data/val_bin2.pkl')

# AGNs
df = pd.read_pickle('Pickles/lmst/interpolated_data/val_agn.pkl')
lenghtpre = len(df)
df = df[~df['file_name'].isin(nan_gaia)]
print(f"AGNs: Number of rows before dropping: {lenghtpre} and after dropping: {len(df)}")
num_removed += lenghtpre - len(df)
df.to_pickle('Pickles/lmst/interpolated_data/val_agn2.pkl')

# STARS
df = pd.read_pickle('Pickles/lmst/interpolated_data/val_star.pkl')
lenghtpre = len(df)
df = df[~df['file_name'].isin(nan_gaia)]
print(f"STARS: Number of rows before dropping: {lenghtpre} and after dropping: {len(df)}")
num_removed += lenghtpre - len(df)
print(f"Total number of rows removed: {num_removed} out of {num_to_remove}")
df.to_pickle('Pickles/lmst/interpolated_data/val_star2.pkl')

# remove bad lamost spectra from gaia dataframes whose obsid is in bad_lamost
print("len(binary) before removing bad lamost spectra:", len(vbingaia))
vbingaia = vbingaia[~vbingaia["obsid"].isin(bad_lamost)]
print("len(binary) after removing bad lamost spectra:", len(vbingaia))

print("len(star) before removing bad lamost spectra:", len(vstargaia))
vstargaia = vstargaia[~vstargaia["obsid"].isin(bad_lamost)]
print("len(star) after removing bad lamost spectra:", len(vstargaia))

print("len(gal) before removing bad lamost spectra:", len(vgalgaia))
vgalgaia = vgalgaia[~vgalgaia["obsid"].isin(bad_lamost)]
print("len(gal) after removing bad lamost spectra:", len(vgalgaia))

print("len(agn) before removing bad lamost spectra:", len(vagngaia))
vagngaia = vagngaia[~vagngaia["obsid"].isin(bad_lamost)]
print("len(agn) after removing bad lamost spectra:", len(vagngaia))

# Make the folder for the cleaned dataframes
!mkdir Pickles/vcleaned4

# Save the cleaned dataframes
vbingaia.to_pickle("Pickles/vcleaned4/bin_gaia.pkl")
vstargaia.to_pickle("Pickles/vcleaned4/star_gaia.pkl")
vgalgaia.to_pickle("Pickles/vcleaned4/gal_gaia.pkl")
vagngaia.to_pickle("Pickles/vcleaned4/agn_gaia.pkl")

X = pd.read_pickle("Pickles/fusionv0/all.pkl")
y = X["label"]
X = X.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", "phot_g_mean_flux", "flagnopllx"
                        ,"phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "obsid", "label"], axis=1)

# show column names
print(X.columns)
# Create the data generators
train_generator = BalancedDataGenerator(X_train, y_train, batch_size=32, limit_per_label=1600)
val_generator = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)


input_shape = (X_train.shape)  # Adjust based on your data
num_classes = len(np.unique(y_val))

model = create_convnet(input_shape, num_classes)

# Train the model
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=10,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import gc
import pandas as pd

# Custom BalancedDataGenerator class
class BalancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, limit_per_label=1600):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.limit_per_label = limit_per_label
        self.classes = np.unique(y)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indices = []
        for cls in self.classes:
            cls_indices = np.where(self.y == cls)[0]
            if len(cls_indices) > self.limit_per_label:
                cls_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
            self.indices.extend(cls_indices)
        np.random.shuffle(self.indices)

# Create the Conv1D model
def create_convnet(input_shape, num_classes, 
                   num_filters=[128, 128, 128, 128, 128, 128, 128, 128], 
                   kernel_size=9,
                   dense_units1=256, 
                   dense_units2=128,
                   dense_units3=64,
                   dropout_rate=0.2,
                   padding='same'):
    model = tf.keras.models.Sequential()
    
    # First convolutional layer
    model.add(tf.keras.layers.Conv1D(filters=num_filters[0], kernel_size=kernel_size, 
                                     activation='relu', input_shape=input_shape, padding=padding))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    
    # Additional convolutional layers
    for filters in num_filters[1:]:
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                         activation='relu', padding=padding))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    
    # Flatten the output and add dense layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=dense_units1, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    # Adding another dense layer
    if dense_units2:
        model.add(tf.keras.layers.Dense(units=dense_units2, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    # Adding another dense layer
    if dense_units3:
        model.add(tf.keras.layers.Dense(units=dense_units3, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    # Optimizer and loss function
    optimizer_ = tf.keras.optimizers.AdamW(learning_rate=1e-4) 

    # Compile the model
    model.compile(optimizer=optimizer_, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Function to train the model with the training dataset and pre-loaded validation dataset
def train_convnet(model, train_dataset, val_dataset,  limit_per_label=2000, epochs=1, batch_size=32, patience=5):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Fit the model using the pre-loaded validation dataset
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping])
    
    return history

# Function to train the model multiple times
def train_convnet_many_times(model, train_dataset, val_dataset, epochs_per_run=1, batch_size=32, num_runs=10, limit_per_label=2000):
    histories = []
    for i in range(num_runs):
        print(f"Training run {i+1}/{num_runs}...")
        history = train_convnet(model, train_dataset, val_dataset, limit_per_label=limit_per_label, epochs=epochs_per_run, batch_size=batch_size)
        histories.append(history)
    
    return histories

def print_confusion_matrix(convnet_model, val_spectranan, val_labelsnan):   
    # Make predictions on the validation/test dataset
    val_predictions = convnet_model.predict(val_spectranan)

    # Convert the predictions to class labels (assuming one-hot encoding)
    predicted_labels = np.argmax(val_predictions, axis=1)

    # Convert true labels if they are in one-hot encoded format
    true_labels = np.array(val_labelsnan)  # Assuming val_labels is already numeric


    # Generate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Optionally, print a classification report for more metrics
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Star', 'Binary Star','Galaxy',  'AGN'], yticklabels=['Star', 'Binary Star','Galaxy',  'AGN'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
import numpy as np
import tensorflow as tf

class BalancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, limit_per_label=1600):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.limit_per_label = limit_per_label
        self.classes = np.unique(y)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indices = []
        for cls in self.classes:
            cls_indices = np.where(self.y == cls)
            if len(cls_indices) > self.limit_per_label:
                cls_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
            self.indices.extend(cls_indices)
        np.random.shuffle(self.indices)

# Load your data and preprocess it
X = pd.read_pickle("Pickles/fusionv0/all.pkl")
y = X["label"]
label_mapping = {'star': 0, 'binary_star': 1, 'galaxy': 2, 'agn': 3}

# Assuming y is a Pandas series or NumPy array, map the labels
y = y.map(label_mapping) if isinstance(y, pd.Series) else np.vectorize(label_mapping.get)(y)

X = X.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", "phot_g_mean_flux", "flagnopllx",
            "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "obsid", "label"], axis=1)

# Ensure data is in the correct shape for Conv1D
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to NumPy arrays to avoid index issues
X_train = np.expand_dims(X_train.to_numpy(), axis=-1)
y_train = y_train.to_numpy()
X_val = np.expand_dims(X_val.to_numpy(), axis=-1)
y_val = y_val.to_numpy()

# Clear memory
del X, y
gc.collect()

# Create data generators
train_generator = BalancedDataGenerator(X_train, y_train, batch_size=32, limit_per_label=1600)
val_generator = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

# Define input shape and number of classes
input_shape = (3748, 1)  # Since your data is 1D with 3748 columns
num_classes = len(np.unique(y_val))

# Create and train the model
model = create_convnet(input_shape, num_classes)
train_convnet_many_times(model, train_generator, val_generator, epochs_per_run=1, num_runs=10)
print_confusion_matrix(model, X_val, y_val)

# Custom BalancedDataGenerator class for training
class BalancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, limit_per_label=1600):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.limit_per_label = limit_per_label
        self.classes = np.unique(y)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indices = []
        for cls in self.classes:
            cls_indices = np.where(self.y == cls)[0]
            if len(cls_indices) > self.limit_per_label:
                cls_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
            self.indices.extend(cls_indices)
        np.random.shuffle(self.indices)

# Custom BalancedDataGenerator class for validation (400 per class)
class BalancedValidationGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, limit_per_label=400):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.limit_per_label = limit_per_label
        self.classes = np.unique(y)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indices = []
        for cls in self.classes:
            cls_indices = np.where(self.y == cls)[0]
            if len(cls_indices) > self.limit_per_label:
                cls_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
            self.indices.extend(cls_indices)
        np.random.shuffle(self.indices)

# Create the Conv1D model
def create_convnet(input_shape, num_classes, 
                   num_filters=[128, 128, 128, 128, 128, 128, 128, 128], 
                   kernel_size=9,
                   dense_units1=256, 
                   dense_units2=128,
                   dense_units3=64,
                   dropout_rate=0.2,
                   padding='same'):
    model = tf.keras.models.Sequential()
    
    # First convolutional layer
    model.add(tf.keras.layers.Conv1D(filters=num_filters[0], kernel_size=kernel_size, 
                                     activation='relu', input_shape=input_shape, padding=padding))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    
    # Additional convolutional layers
    for filters in num_filters[1:]:
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                         activation='relu', padding=padding))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    
    # Flatten the output and add dense layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=dense_units1, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    # Adding another dense layer
    if dense_units2:
        model.add(tf.keras.layers.Dense(units=dense_units2, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    # Adding another dense layer
    if dense_units3:
        model.add(tf.keras.layers.Dense(units=dense_units3, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    # Optimizer and loss function
    optimizer_ = tf.keras.optimizers.AdamW(learning_rate=1e-4) 

    # Compile the model
    model.compile(optimizer=optimizer_, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Function to train the model
def train_convnet(model, train_dataset, val_dataset, limit_per_label=1600, epochs=1, batch_size=32, patience=5):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Fit the model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping])
    
    return history

# Function to train the model multiple times
def train_convnet_many_times(model, train_dataset, val_dataset, epochs_per_run=1, batch_size=32, num_runs=10, limit_per_label=1600):
    histories = []
    for i in range(num_runs):
        print(f"Training run {i+1}/{num_runs}...")
        history = train_convnet(model, train_dataset, val_dataset, limit_per_label=limit_per_label, epochs=epochs_per_run, batch_size=batch_size)
        histories.append(history)
    
    return histories

# Function to print confusion matrix and classification report
def print_confusion_matrix(convnet_model, val_spectra, val_labels):   
    val_predictions = convnet_model.predict(val_spectra)
    predicted_labels = np.argmax(val_predictions, axis=1)
    true_labels = np.array(val_labels)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Star', 'Binary Star','Galaxy', 'AGN'], yticklabels=['Star', 'Binary Star','Galaxy', 'AGN'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Load and preprocess data
X = pd.read_pickle("Pickles/fusionv0/all.pkl")
y = X["label"]
label_mapping = {'star': 0, 'binary_star': 1, 'galaxy': 2, 'agn': 3}
y = y.map(label_mapping) if isinstance(y, pd.Series) else np.vectorize(label_mapping.get)(y)

X = X.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
            "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
            "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "obsid", "label"], axis=1)

# Train the model
num_epochs = 200
lr = 1e-4
patience = num_epochs
batch_size = 512
dropout_rate = 0.2
kernel_size = 9

# Start an MLflow run
with mlflow.start_run(log_system_metrics=True):
    # Log parameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("lr", lr)
    mlflow.log_param("patience", patience)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_filters", filters)
    mlflow.log_param("dense_units", dense)
    mlflow.log_param("dropout_rate", dropout_rate)
    mlflow.log_param("kernel_size", kernel_size)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=patience)

    # Evaluate the model
    print_confusion_matrix(trained_model, val_loader)

    # Save the model
    mlflow.pytorch.log_model(trained_model, "model")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import gc
import psutil
import GPUtil


# Custom Dataset for handling balanced data
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

# Define the Conv1D PyTorch model
class ConvNet(nn.Module):
    def __init__(self, input_shape, num_classes, 
                 num_filters=[128, 128, 128, 128, 128, 128, 128, 128], 
                 kernel_size=9,
                 dense_units=[256, 256, 256, 128, 128, 128, 64, 64, 64],
                 dropout_rate=0.2, padding='same'):
        super(ConvNet, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        in_channels = 1  # Since it's a 1D input
        
        # Add convolutional layers
        for filters in num_filters:
            conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, padding=kernel_size//2)
            self.conv_layers.append(conv_layer)
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = filters
        
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        
        # Compute the flattened output size (based on input shape and pooling)
        # Assumption: input_shape[0] is the sequence length
        final_seq_len = input_shape[0] // (2 ** len(num_filters))  # After all pooling layers
        
        # Add dense layers
        dense_input_units = num_filters[-1] * final_seq_len
        self.dense_layers = nn.ModuleList()
        for units in dense_units:
            self.dense_layers.append(nn.Linear(dense_input_units, units))
            dense_input_units = units
        
        # Output layer
        self.output_layer = nn.Linear(dense_input_units, num_classes)
    
    def forward(self, x):
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            x = pool_layer(torch.relu(conv_layer(x)))
            x = self.dropout(x)
        x = self.flatten(x)
        for dense_layer in self.dense_layers:
            x = torch.relu(dense_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return torch.softmax(x, dim=1)

def log_system_metrics(epoch):
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    gpus = GPUtil.getGPUs()
    
    if gpus:
        for i, gpu in enumerate(gpus):
            mlflow.log_metric(f"gpu_{i}_usage", gpu.load * 100, step=epoch)
            mlflow.log_metric(f"gpu_{i}_memory_used", gpu.memoryUsed, step=epoch)
            mlflow.log_metric(f"gpu_{i}_memory_total", gpu.memoryTotal, step=epoch)
    
    mlflow.log_metric("cpu_usage", cpu_usage, step=epoch)
    mlflow.log_metric("memory_usage", memory_info.percent, step=epoch)

# Updated train_model function with system metrics logging
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, patience=5, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Re-sample the training dataset at the start of each epoch
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
            train_accuracy = (outputs.argmax(dim=1) == y_batch).float().mean()
            train_loss += loss.item() * X_batch.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_accuracy = (outputs.argmax(dim=1) == y_val).float().mean()
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        # Log metrics to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy.item(), step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy.item(), step=epoch)
        
        # Log system metrics
        log_system_metrics(epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break
    
    return model
# Confusion matrix and classification report
def print_confusion_matrix(model, val_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.extend(np.argmax(preds, axis=1))
            all_labels.extend(y_batch.numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Star', 'Binary Star', 'Galaxy', 'AGN'], 
                yticklabels=['Star', 'Binary Star', 'Galaxy', 'AGN'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Main script to load data and train the model
if __name__ == "__main__":
    # Load and preprocess data
    X = pd.read_pickle("Pickles/fusionv0/train.pkl")
    y = X["label"]
    label_mapping = {'star': 0, 'binary_star': 1, 'galaxy': 2, 'agn': 3}
    y = y.map(label_mapping).values
    
    X = X.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
                "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
                "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "label"], axis=1).values
    
    # Read test data
    X_test = pd.read_pickle("Pickles/fusionv0/test.pkl")
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
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# Initialize model, train, and evaluate
filters=[128, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
dense=[512, 256, 64]
model = ConvNet(input_shape=(3748,), num_classes=4, num_filters=filters, kernel_size=9, dense_units=dense, dropout_rate=0.2)



# model summary
print(model)

# Print number of parameters
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

num_epochs = 200
lr = 1e-4
patience = num_epochs
batch_size = 512
dropout_rate = 0.2
kernel_size = 9

# Start an MLflow run
with mlflow.start_run(log_system_metrics=True):
    # Log parameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("lr", lr)
    mlflow.log_param("patience", patience)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_filters", filters)
    mlflow.log_param("dense_units", dense)
    mlflow.log_param("dropout_rate", dropout_rate)
    mlflow.log_param("kernel_size", kernel_size)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=patience)

    # Evaluate the model
    print_confusion_matrix(trained_model, val_loader)

    # Save the model
    mlflow.pytorch.log_model(trained_model, "model")

    import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import gc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.tensorflow
# Enable autologging
mlflow.tensorflow.autolog()
mlflow.set_tracking_uri(uri="file:///C:/Users/jcwin/OneDrive - University of Southampton/_Southampton/2024-25/Star-Classifier/mlflow")
mlflow.set_experiment("star-classifier")



class BalancedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, limit_per_label=1600):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.limit_per_label = limit_per_label
        self.classes = np.unique(y)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indices = []
        for cls in self.classes:
            cls_indices = np.where(self.y == cls)[0]
            if len(cls_indices) > self.limit_per_label:
                cls_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
            self.indices.extend(cls_indices)
        np.random.shuffle(self.indices)

# Custom BalancedDataGenerator class for validation (400 per class)
class BalancedValidationGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, limit_per_label=400):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.limit_per_label = limit_per_label
        self.classes = np.unique(y)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indices = []
        for cls in self.classes:
            cls_indices = np.where(self.y == cls)[0]
            if len(cls_indices) > self.limit_per_label:
                cls_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
            self.indices.extend(cls_indices)
        np.random.shuffle(self.indices)

# Create the Conv1D model
def create_convnet(input_shape, num_classes, 
                   num_filters=[128, 128, 128, 128, 128, 128, 128, 128], 
                   kernel_size=9,
                   dense_units=[256, 256, 256, 128, 128, 128, 64, 64, 64],
                   dropout_rate=0.2,
                   padding='same'):
    model = tf.keras.models.Sequential()
    
    # First convolutional layer
    model.add(tf.keras.layers.Conv1D(filters=num_filters[0], kernel_size=kernel_size, 
                                     activation='relu', input_shape=input_shape, padding=padding))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    
    # Additional convolutional layers
    for filters in num_filters[1:]:
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                         activation='relu', padding=padding))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    
    # Flatten the output and add dense layers
    model.add(tf.keras.layers.Flatten())

    for units in dense_units:
        model.add(tf.keras.layers.Dense(units=units, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    # Optimizer and loss function
    optimizer_ = tf.keras.optimizers.AdamW(learning_rate=1e-4) 

    # Compile the model
    model.compile(optimizer=optimizer_, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Function to train the model
def train_convnet(model, train_dataset, val_dataset, limit_per_label=1600, epochs=1, batch_size=32, patience=5):
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Start an MLflow run
    with mlflow.start_run():
        # Fit the model
        history = model.fit(train_dataset,
                            validation_data=val_dataset,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping])
    
    return history

# Function to train the model multiple times
def train_convnet_many_times(model, train_dataset, val_dataset, epochs_per_run=1, batch_size=512, num_runs=10, limit_per_label=1600):
    histories = []
    for i in range(num_runs):
        print(f"Training run {i+1}/{num_runs}...")
        history = train_convnet(model, train_dataset, val_dataset, limit_per_label=limit_per_label, epochs=epochs_per_run, batch_size=batch_size)
        histories.append(history)
    
    return histories

# Function to print confusion matrix and classification report
def print_confusion_matrix(convnet_model, val_spectra, val_labels):   
    val_predictions = convnet_model.predict(val_spectra)
    predicted_labels = np.argmax(val_predictions, axis=1)
    true_labels = np.array(val_labels)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Star', 'Binary Star','Galaxy', 'AGN'], yticklabels=['Star', 'Binary Star','Galaxy', 'AGN'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Define input shape and number of classes
input_shape = (3748, 1)
num_classes = len(np.unique(y_val))
batchsize = 512


# Load and preprocess data
X = pd.read_pickle("Pickles/fusionv0/train.pkl")
y = X["label"]
label_mapping = {'star': 0, 'binary_star': 1, 'galaxy': 2, 'agn': 3}
y = y.map(label_mapping) if isinstance(y, pd.Series) else np.vectorize(label_mapping.get)(y)

X = X.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
            "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
            "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "label"], axis=1)

# Read test data
X_test = pd.read_pickle("Pickles/fusionv0/test.pkl")
y_test = X_test["label"]
label_mapping = {'star': 0, 'binary_star': 1, 'galaxy': 2, 'agn': 3}
y_test = y_test.map(label_mapping) if isinstance(y_test, pd.Series) else np.vectorize(label_mapping.get)(y_test)

X_test = X_test.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error",
                        "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux",
                        "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "label"], axis=1)



# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train.to_numpy(), axis=-1)
y_train = y_train.to_numpy()
X_val = np.expand_dims(X_val.to_numpy(), axis=-1)
y_val = y_val.to_numpy()

# Clear memory
del X, y
gc.collect()

# Create data generators
train_generator = BalancedDataGenerator(X_train, y_train, batch_size=batchsize, limit_per_label=1600)
val_generator = BalancedValidationGenerator(X_val, y_val, batch_size=batchsize, limit_per_label=400)

# Create and train the model
filters=[64, 128, 128, 256, 256, 512, 512, 512, 1024, 1024]
dense=[512, 256, 64]
model = create_convnet(input_shape, num_classes, num_filters=filters, kernel_size=(9,), dense_units=dense)
model.summary()

histories = train_convnet_many_times(model, train_generator, val_generator, epochs_per_run=1, num_runs=10)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Initialize model with Gaia input size
gaia_input_size = X_train_gaia.shape[1]
print(f"Gaia input size: {gaia_input_size}")
filters=[128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 1024]
dense=[2048, 2048 , 1024, 512, 256, 64]

model = ConvNetFusion(input_shape=(3748,), num_classes=4, gaia_input_size=gaia_input_size, 
                      num_filters=filters, kernel_size=9, dense_units=dense, dropout_rate=0.2, gaia_fusion_units=1024)
print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


# MAYBE ADD THIS TO THE MAIN SCRIPT
# Initialize model with Gaia input size
gaia_input_size = X_train_gaia.shape[1]
print(f"Gaia input size: {gaia_input_size}")
filters=[128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 1024]
dense=[2048, 2048 , 1024, 512, 256, 64]

model = ConvNetFusion(input_shape=(3748,), num_classes=4, gaia_input_size=gaia_input_size, 
                      num_filters=filters, kernel_size=9, dense_units=dense, dropout_rate=0.2, gaia_fusion_units=1024)
print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Initialize the model
gaia_input_size = X_train_gaia.shape[1]
filters = [128, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
dense = [1024, 1024, 512, 256, 64]

model = ConvNetFusion(input_shape=(3748,), num_classes=4, gaia_input_size=gaia_input_size, 
                      num_filters=filters, kernel_size=9, dense_units=dense, dropout_rate=0.2, gaia_fusion_units=512)
print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Training parameters
num_epochs = 200
lr = 1e-4
patience = 10
batch_size = 512

# Start an MLflow run
with mlflow.start_run(log_system_metrics=True):
    # Log parameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("lr", lr)
    mlflow.log_param("patience", patience)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_filters", filters)
    mlflow.log_param("dense_units", dense)
    mlflow.log_param("dropout_rate", 0.2)
    mlflow.log_param("kernel_size", 20)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=patience)

    # Evaluate the model
    print_confusion_matrix(trained_model, val_loader)
    # Save the model in MLflow
    #mlflow.pytorch.log_model(trained_model, "model")

    import torch
import torch.nn as nn

class ConvNetFusion(nn.Module):
    def __init__(self, input_shape, num_classes, gaia_input_size, 
                 num_filters=[128, 128, 128, 128],  # Reduced for simplicity
                 kernel_size=9,
                 dense_units=[256, 128, 64, 32],
                 dropout_rate=0.2, 
                 gaia_fusion_units=10000):
        super(ConvNetFusion, self).__init__()

        # Spectra input path: Conv1D layers and pooling layers
        for i, filters in enumerate(num_filters):
            if i == 0:
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size, padding=kernel_size//2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2)
                )
            else:
                self.conv_layers.add_module(f"conv_{i}", nn.Conv1d(in_channels=num_filters[i-1], out_channels=filters, kernel_size=kernel_size, padding=kernel_size//2))
                self.conv_layers.add_module(f"relu_{i}", nn.ReLU())
                self.conv_layers.add_module(f"pool_{i}", nn.MaxPool1d(kernel_size=2))
        
        # Compute the flattened output size (based on input shape and pooling)
        final_seq_len = input_shape[0] // (2 ** len(num_filters))  # After all pooling layers
        spectra_out_size = num_filters[-1] * final_seq_len

        # Gaia input path: Dense layer to project Gaia input features
        self.gaia_branch = nn.Sequential(
            nn.Linear(gaia_input_size, gaia_fusion_units),
            nn.ReLU()
        )
        
        # Concatenation of Spectra and Gaia features
        fused_input_size = spectra_out_size + gaia_fusion_units
        
        # Fully connected layers after fusion
        for i, units in enumerate(dense_units):
            if i == 0:
                self.dense_layers = nn.Sequential(
                    nn.Linear(fused_input_size, units),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                )
            else:
                self.dense_layers.add_module(f"dense_{i}", nn.Linear(dense_units[i-1], units))
                self.dense_layers.add_module(f"relu_{i}", nn.ReLU())
                self.dense_layers.add_module(f"dropout_{i}", nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(dense_units[-1], num_classes)
    
    def forward(self, x_conv, x_gaia):
        # Spectra branch forward pass
        x_conv = self.spectra_branch(x_conv)
        x_conv = torch.flatten(x_conv, start_dim=1)  # Flatten for dense layers
        
        # Gaia branch forward pass
        x_gaia = self.gaia_branch(x_gaia)
        
        # Concatenate the two branches
        x_fused = torch.cat((x_conv, x_gaia), dim=1)
        
        # Pass through dense layers
        x = self.dense_layers(x_fused)
        
        # Output layer
        x = self.output_layer(x)
        
        return torch.softmax(x, dim=1)

# Initialize toy model
gaia_input_size = X_train_gaia.shape[1]
print(f"Gaia input size: {gaia_input_size}")
filters = [16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128]
dense=[256, 256 , 256 ,128, 64, 32]

model = ConvNetFusion(input_shape=(3748,), num_classes=4, gaia_input_size=gaia_input_size, 
                      num_filters=filters, kernel_size=9, dense_units=dense, dropout_rate=0.2, gaia_fusion_units=128)
# Save the model
torch.save(model, "Models/toyv0.pth")
# Print model summary
print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# Training parameters
num_epochs = 20
lr = 1e-4
patience = 20
batch_size = 512

# Start an MLflow run
with mlflow.start_run(log_system_metrics=True):
    # Log parameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("lr", lr)
    mlflow.log_param("patience", patience)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_filters", filters)
    mlflow.log_param("dense_units", dense)
    mlflow.log_param("dropout_rate", 0.2)
    mlflow.log_param("kernel_size", 20)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, test_loader, num_epochs=num_epochs, lr=lr, patience=patience)

    # Evaluate the model
    print_confusion_matrix(trained_model, val_loader)
class ConvNetFusion(nn.Module):
    def __init__(self, input_shape, num_classes, gaia_input_size, 
                 num_filters=[128, 128, 128, 128, 128, 128, 128, 128], 
                 kernel_size=9,
                 dense_units=[256, 256, 256, 128, 128, 128, 64, 64, 64],
                 dropout_rate=0.2, 
                 gaia_fusion_units=128,
                 padding='same'):
        super(ConvNetFusion, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        in_channels = 1  # Since it's a 1D input
        
        # Convolutional layers for spectra
        for filters in num_filters:
            conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, padding=kernel_size//2)
            self.conv_layers.append(conv_layer)
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = filters
        
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        
        # Compute the flattened output size (based on input shape and pooling)
        final_seq_len = input_shape[0] // (2 ** len(num_filters))  # After all pooling layers
        
        # Separate input for Gaia features
        self.gaia_input_layer = nn.Linear(gaia_input_size, gaia_fusion_units)

        # Add dense layers
        dense_input_units = num_filters[-1] * final_seq_len + gaia_fusion_units  # Combine convnet output and Gaia features
        self.dense_layers = nn.ModuleList()
        for units in dense_units:
            self.dense_layers.append(nn.Linear(dense_input_units, units))
            dense_input_units = units
        
        # Output layer
        self.output_layer = nn.Linear(dense_input_units, num_classes)
    
    def forward(self, x_conv, x_gaia):
        # Separate path for convolutional spectra data
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            x_conv = pool_layer(torch.relu(conv_layer(x_conv)))
            x_conv = self.dropout(x_conv)
        x_conv = self.flatten(x_conv)  # Flatten after conv layers

        # Separate path for Gaia features
        x_gaia = torch.relu(self.gaia_input_layer(x_gaia))

        # Concatenate both the spectra and Gaia features
        x = torch.cat((x_conv, x_gaia), dim=1)
        
        # Pass through dense layers
        for dense_layer in self.dense_layers:
            x = torch.relu(dense_layer(x))
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        return torch.softmax(x, dim=1)

# Initialize toy model
gaia_input_size = X_train_gaia.shape[1]
print(f"Gaia input size: {gaia_input_size}")
filters = [16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128]
dense=[256, 256 , 256 ,128, 64, 32]

model = ConvNetFusion(input_shape=(3748,), num_classes=4, gaia_input_size=gaia_input_size, 
                      num_filters=filters, kernel_size=9, dense_units=dense, dropout_rate=0.2, gaia_fusion_units=128)
# Save the model
torch.save(model, "Models/toyv0.pth")
# Print model summary
print(model)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
import os
import tempfile
import torch
from sklearn.metrics import confusion_matrix
import mlflow

def print_confusion_matrix(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_conv, X_gaia, y in val_loader:
            outputs = model(X_conv, X_gaia)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Log confusion matrix
    temp_file_path = save_confusion_matrix_to_file(cm)
    mlflow.log_artifact(temp_file_path, artifact_path="confusion_matrix")
    os.remove(temp_file_path)  # Clean up the temporary file

def save_confusion_matrix_to_file(cm):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    xticklabels = ['Star', 'Binary Star', 'Galaxy', 'AGN']
    yticklabels = ['Star', 'Binary Star', 'Galaxy', 'AGN']
    
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=xticklabels, rotation=45)
    plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=yticklabels, rotation=0)

    # Save the figure to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, format='png')
    temp_file.close()
    
    return temp_file.name
import os
import tempfile
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
import mlflow

def print_confusion_matrix(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_conv, X_gaia, y in val_loader:
            outputs = model(X_conv, X_gaia)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Calculate and print additional metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")

    # Log confusion matrix
    temp_file_path = save_confusion_matrix_to_file(cm)
    mlflow.log_artifact(temp_file_path, artifact_path="confusion_matrix")
    os.remove(temp_file_path)  # Clean up the temporary file

def save_confusion_matrix_to_file(cm):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    xticklabels = ['Star', 'Binary Star', 'Galaxy', 'AGN']
    yticklabels = ['Star', 'Binary Star', 'Galaxy', 'AGN']
    
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=xticklabels, rotation=45)
    plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=yticklabels, rotation=0)

    # Save the figure to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name, format='png')
    temp_file.close()
    
    return temp_file.name   

def train_model(model, train_loader, val_loader, test_loader, num_epochs=200, lr=1e-4, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    early_stopping_counter = 0
    best_test_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    for epoch in range(num_epochs):
        # Re-sample the training dataset at the start of each epoch
        train_loader.dataset.re_sample()
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        # Training loop
        for X_conv, X_gaia, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X_conv, X_gaia)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_conv.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y).sum().item()
            total_train += y.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = correct_train / total_train

        # Validation loop
        model.eval()
        val_loss, correct_val, total_val, test_loss, correct_test, total_test = 0, 0, 0, 0, 0, 0

        with torch.no_grad():
            for X_conv, X_gaia, y in val_loader:
                outputs = model(X_conv, X_gaia)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X_conv.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == y).sum().item()
                total_val += y.size(0)
        # Test loop
        with torch.no_grad():
            for X_conv, X_gaia, y in test_loader:
                outputs = model(X_conv, X_gaia)
                loss = criterion(outputs, y)
                test_loss += loss.item() * X_conv.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_test += (predicted == y).sum().item()
                total_test += y.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val
        test_loss /= len(test_loader.dataset)
        test_acc = correct_test / total_test

        # Log metrics for MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_acc", test_acc, step=epoch)


        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


    # Load the best model weights before returning
    model.load_state_dict(best_model)
    return model

    # Define the Gaia Branch as a separate module
class GaiaBranch(nn.Module):
    def __init__(self, gaia_input_size, gaia_fusion_units):
        super(GaiaBranch, self).__init__()
        self.fc = nn.Linear(gaia_input_size, gaia_fusion_units)

    def forward(self, x):
        return torch.relu(self.fc(x))
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import gc
import psutil
import GPUtil

# Enable MLflow autologging for PyTorch
mlflow.pytorch.autolog()
mlflow.set_tracking_uri(uri="file:///C:/Users/jcwin/OneDrive - University of Southampton/_Southampton/2024-25/Star-Classifier/mlflow")
mlflow.set_experiment("LAMOST_ConVnet")

# Custom Dataset for handling balanced data
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

# Define the Conv1D PyTorch model
class ConvNet(nn.Module):
    def __init__(self, input_shape, num_classes, 
                 num_filters=[128, 128, 128, 128, 128, 128, 128, 128], 
                 kernel_size=9,
                 dense_units=[256, 256, 256, 128, 128, 128, 64, 64, 64],
                 dropout_rate=0.2, padding='same'):
        super(ConvNet, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        in_channels = 1  # Since it's a 1D input
        
        # Add convolutional layers
        for filters in num_filters:
            conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, padding=kernel_size//2)
            self.conv_layers.append(conv_layer)
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = filters
        
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        
        # Compute the flattened output size (based on input shape and pooling)
        # Assumption: input_shape[0] is the sequence length
        final_seq_len = input_shape[0] // (2 ** len(num_filters))  # After all pooling layers
        
        # Add dense layers
        dense_input_units = num_filters[-1] * final_seq_len
        self.dense_layers = nn.ModuleList()
        for units in dense_units:
            self.dense_layers.append(nn.Linear(dense_input_units, units))
            dense_input_units = units
        
        # Output layer
        self.output_layer = nn.Linear(dense_input_units, num_classes)
    
    def forward(self, x):
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            x = pool_layer(torch.relu(conv_layer(x)))
            x = self.dropout(x)
        x = self.flatten(x)
        for dense_layer in self.dense_layers:
            x = torch.relu(dense_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return torch.softmax(x, dim=1)


def log_system_metrics(epoch):
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    gpus = GPUtil.getGPUs()
    
    if gpus:
        for i, gpu in enumerate(gpus):
            mlflow.log_metric(f"gpu_{i}_usage", gpu.load * 100, step=epoch)
            mlflow.log_metric(f"gpu_{i}_memory_used", gpu.memoryUsed, step=epoch)
            mlflow.log_metric(f"gpu_{i}_memory_total", gpu.memoryTotal, step=epoch)
    
    mlflow.log_metric("cpu_usage", cpu_usage, step=epoch)
    mlflow.log_metric("memory_usage", memory_info.percent, step=epoch)

# Updated train_model function with system metrics logging
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, patience=5, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Re-sample the training dataset at the start of each epoch
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
            train_accuracy = (outputs.argmax(dim=1) == y_batch).float().mean()
            train_loss += loss.item() * X_batch.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                val_accuracy = (outputs.argmax(dim=1) == y_val).float().mean()
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        # Log metrics to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy.item(), step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy.item(), step=epoch)
        
        # Log system metrics
        log_system_metrics(epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break
    
    return model
# Confusion matrix and classification report
def print_confusion_matrix(model, val_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.extend(np.argmax(preds, axis=1))
            all_labels.extend(y_batch.numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Star', 'Binary Star', 'Galaxy', 'AGN'], 
                yticklabels=['Star', 'Binary Star', 'Galaxy', 'AGN'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Main script to load data and train the model
if __name__ == "__main__":
    # Load and preprocess data
    X = pd.read_pickle("Pickles/fusionv0/train.pkl")
    y = X["label"]
    label_mapping = {'star': 0, 'binary_star': 1, 'galaxy': 2, 'agn': 3}
    y = y.map(label_mapping).values
    
    X = X.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
                "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
                "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "label"], axis=1).values
    
    # Read test data
    X_test = pd.read_pickle("Pickles/fusionv0/test.pkl")
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
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# Initialize model, train, and evaluate
filters=[128, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512]
dense=[512, 256, 64]
model = ConvNet(input_shape=(3748,), num_classes=4, num_filters=filters, kernel_size=9, dense_units=dense, dropout_rate=0.2)



# model summary
print(model)

# Print number of parameters
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
num_epochs = 200
lr = 1e-4
patience = num_epochs
batch_size = 512
dropout_rate = 0.2
kernel_size = 9

# Start an MLflow run
with mlflow.start_run(log_system_metrics=True):
    # Log parameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("lr", lr)
    mlflow.log_param("patience", patience)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_filters", filters)
    mlflow.log_param("dense_units", dense)
    mlflow.log_param("dropout_rate", dropout_rate)
    mlflow.log_param("kernel_size", kernel_size)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=patience)

    # Evaluate the model
    print_confusion_matrix(trained_model, val_loader)

    # Save the model
    mlflow.pytorch.log_model(trained_model, "model")
# Evaluate with confusion matrix and classification report
test_loader = DataLoader(BalancedValidationDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
                                                    torch.tensor(y_test, dtype=torch.long)),
                         batch_size=512, shuffle=False)
print_confusion_matrix(trained_model, test_loader)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import gc
from sklearn.model_selection import train_test_split
import psutil
import GPUtil

# Enable MLflow autologging
mlflow.pytorch.autolog()

# Define the Vision Transformer model
class VisionTransformer1D(nn.Module):
    def __init__(self, input_size=3748, num_classes=4, patch_size=16, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1):
        super(VisionTransformer1D, self).__init__()
        
        # Parameters
        self.num_patches = input_size // patch_size
        self.patch_size = patch_size
        self.dim = dim
        
        # Patch Embedding layer
        self.patch_embed = nn.Linear(patch_size, dim)
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        
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
        # Convert input into patches and embed them
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_patches, self.patch_size)
        x = self.patch_embed(x) + self.pos_embedding
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Classify based on the first token representation
        x = self.fc(x[:, 0])
        
        return x

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

# Training function (using a similar approach to your ConvNet setup)
def train_model_vit(model, train_loader, val_loader, num_epochs=10, lr=1e-4, patience=5, device='cuda'):
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
        
        # Log metrics
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    return model

# Use your confusion matrix function to evaluate the model
def print_confusion_matrix_vit(model, val_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.extend(np.argmax(preds, axis=1))
            all_labels.extend(y_batch.numpy())
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:", conf_matrix)
    print("Classification Report:", classification_report(all_labels, all_preds))

# Main training script
if __name__ == "__main__":
    # Load and preprocess your data
    X = pd.read_pickle("Pickles/fusionv0/train.pkl")
    y = X["label"]
    label_mapping = {'star': 0, 'binary_star': 1, 'galaxy': 2, 'agn': 3}
    y = y.map(label_mapping).values
    
    X = X.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
                "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
                "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "label"], axis=1).values
    
    # Read test data
    X_test = pd.read_pickle("Pickles/fusionv0/test.pkl")
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
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Initialize Vision Transformer
    model_vit = VisionTransformer1D()
    num_epochs = 200
    lr = 1e-4
    patience = 10

    with mlflow.start_run():
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("learning_rate", lr)
        trained_model_vit = train_model_vit(model_vit, train_loader, val_loader, num_epochs, lr, patience)
        print_confusion_matrix_vit(trained_model_vit, val_loader)
class VisionTransformer1D(nn.Module):
    def __init__(self, input_size=3748, num_classes=4, patch_size=5, dim=128, depth=12, heads=16, mlp_dim=256, dropout=0.2):
        super(VisionTransformer1D, self).__init__()

        # Store patch size and dimensionality for embedding
        self.patch_size = patch_size
        self.dim = dim

        # Patch Embedding layer
        self.patch_embed = nn.Linear(patch_size, dim)

        # Positional Encoding (initialize to a reasonable size, but we’ll adjust it dynamically)
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
        batch_size, channels, seq_len = x.shape
        x = x.squeeze(1) if channels == 1 else x
        pad_length = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        x = nn.functional.pad(x, (0, pad_length))
        x = x.view(batch_size, -1, self.patch_size)

        x = self.patch_embed(x)
        if torch.isnan(x).any():
            print("NaN detected after patch embedding")
        x += self.pos_embedding
        
        x = self.transformer(x)
        if torch.isnan(x).any():
            print("NaN detected after transformer layer")

        x = self.fc(x[:, 0])
        if torch.isnan(x).any():
            print("NaN detected in final output layer")

        return x



def train_model_vit(model, train_loader, val_loader, test_loader, num_epochs=500, lr=1e-5, patience=5, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loader.dataset.re_sample()
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            if torch.isnan(outputs).any():
                print("NaN detected in model outputs")
                continue
            loss = criterion(outputs, y_batch)
            if torch.isnan(loss).any():
                print("NaN loss detected")
                continue
            loss.backward()
            for param in model.parameters():
                if torch.isnan(param.grad).any():
                    print("NaN detected in gradients")
                    continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_accuracy = (outputs.argmax(dim=1) == y_batch).float().mean()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item() * X_val.size(0)
                val_accuracy = (outputs.argmax(dim=1) == y_val).float().mean()
        
        test_loss = 0.0
        test_accuracy = 0.0
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                outputs = model(X_test)
                loss = criterion(outputs, y_test)
                test_loss += loss.item() * X_test.size(0)
                test_accuracy = (outputs.argmax(dim=1) == y_test).float().mean()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        test_loss /= len(test_loader.dataset)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch, 
                   "train_accuracy": train_accuracy.item(), "val_accuracy": val_accuracy.item(), 
                   "test_accuracy": test_accuracy.item(), "test_loss": test_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    return model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
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
            pos_embedding = nn.Parameter(torch.randn(1, max_patches, dim))
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
        batch_size, seq_len = x.shape  # channels is 1
        branch_outputs = []
        
        # Extract patches, embed, and process with transformer for each branch
        for branch in self.branches:
            patch_size = branch['patch_embed'].in_features
            stride = int(patch_size * (1 - self.overlap))
            num_patches = (seq_len - patch_size) // stride + 1
            patches = [x[:, i * stride : i * stride + patch_size] for i in range(num_patches)]
            x_branch = torch.stack(patches, dim=1)  # Shape: (batch_size, num_patches, patch_size)
            x_branch = branch['patch_embed'](x_branch) + branch['pos_embedding'][:, :num_patches, :]
            x_branch = branch['transformer'](x_branch)
            branch_outputs.append(x_branch)

        # Apply cross-attention to combine the representations from each branch
        x_fused = torch.cat(branch_outputs, dim=1)  # Concatenate along sequence dimension
        x_fused = self.cross_attention(x_fused)

        # Classification based on the first token representation
        x = self.fc(x_fused[:, 0])
        return x
    
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Mlp, Block

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
        batch_size, seq_len = x.shape
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

# Define the hyperparameters and train as before




# Define the hyperparameters
num_classes = 4
patch_sizes=[1, 17]
dim = 64
depth = 7
heads = 8
mlp_dim = 64
dropout = 0.3
batch_size = 512
lr = 0.0001
patience = 30
num_epochs = 200


# Define the config dictionary object
config = {"num_classes": num_classes, "patch_size": patch_sizes, "dim": dim, "depth": depth, "heads": heads, "mlp_dim": mlp_dim, 
          "dropout": dropout, "batch_size": batch_size, "lr": lr, "patience": patience}

# Initialize WandB project
wandb.init(project="gaia-crossvit", entity="joaoc-university-of-southampton", config=config)
# Initialize and train the model
model_vit = VisionTransformer1D(input_size=17, num_classes=num_classes, patch_sizes=patch_sizes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, overlap=0)
trained_model = train_model_vit(model_vit, train_loader, val_loader, test_loader, num_epochs=num_epochs, lr=lr, max_patience=patience)

# Save the model and finish WandB session
wandb.finish()

def init_rope_frequencies(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    mag = 1 / (theta ** (torch.arange(0, dim // 2, 2).float() / dim))
    angles = torch.rand(num_heads, 1) * 2 * torch.pi if rotate else torch.zeros(num_heads, 1)
    freq_x = mag * torch.cat([torch.cos(angles), torch.cos(torch.pi/2 + angles)], dim=-1)
    freq_y = mag * torch.cat([torch.sin(angles), torch.sin(torch.pi/2 + angles)], dim=-1)
    return torch.stack([freq_x, freq_y], dim=0)

def apply_rotary_position_embeddings(freqs, q, k):
    cos, sin = freqs[0], freqs[1]
    q_rot = (q * cos) + (torch.roll(q, shifts=1, dims=-1) * sin)
    k_rot = (k * cos) + (torch.roll(k, shifts=1, dims=-1) * sin)
    return q_rot, k_rot 

class VisionTransformer1D(nn.Module):
    def __init__(self, input_size, num_classes=4, patch_sizes=[20, 40], overlap=0, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.2, theta=10.0):
        super(VisionTransformer1D, self).__init__()
        self.num_branches = len(patch_sizes)
        self.dim = dim
        self.overlap = overlap
        self.branches = nn.ModuleList()
        
        # Set up branches for different patch sizes
        for patch_size in patch_sizes:
            stride = int(patch_size * (1 - overlap))
            patch_embed = nn.Linear(patch_size, dim)
            transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout), depth
            )
            self.branches.append(nn.ModuleDict({
                'patch_embed': patch_embed,
                'transformer': transformer
            }))
        
        # Cross-Attention with RoPE for fusion
        self.cross_attention = CrossAttentionBlock(dim, heads, theta=theta)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len = x.shape
        branch_outputs = []
        
        for branch in self.branches:
            patch_size = branch['patch_embed'].in_features
            stride = int(patch_size * (1 - self.overlap))
            num_patches = (seq_len - patch_size) // stride + 1
            patches = [x[:, i * stride : i * stride + patch_size] for i in range(num_patches)]
            x_branch = torch.stack(patches, dim=1)
            x_branch = branch['patch_embed'](x_branch)
            x_branch = branch['transformer'](x_branch)
            branch_outputs.append(x_branch)

        x_fused = torch.cat(branch_outputs, dim=1)
        x_fused = self.cross_attention(x_fused)
        x = self.fc(x_fused[:, 0])
        return x
    
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
    def __init__(self, input_size, num_classes=4, patch_sizes=[20, 40], overlap=0.5, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.2):
        super(VisionTransformer1D, self).__init__()
        if type(patch_sizes) is int:
            patch_sizes = [patch_sizes]
        self.num_branches = len(patch_sizes)
        self.dim = dim
        self.overlap = overlap
        self.branches = nn.ModuleList()
        
        # Set up branches for different patch sizes
        for patch_size in patch_sizes:
            stride = int(patch_size * (1 - overlap))
            max_patches = (input_size - patch_size) // stride + 1
            max_patches = (input_size // patch_size) ** 2
            patch_embed = nn.Linear(patch_size, dim)
            pos_embedding = nn.Embedding(max_patches + 1, dim)  # Changed to nn.Embedding to act as a module
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
        batch_size, seq_len = x.shape
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
    

            # Extract patches, embed, and process with transformer for each branch
        for branch in self.branches:
            patch_size = branch['patch_embed'].in_features
            stride = int(patch_size)
            num_patches = (seq_len - patch_size) // stride + 1
            patches = [x[:, i * stride : i * stride + patch_size] for i in range(num_patches)]
            x_branch = torch.stack(patches, dim=1)
            x_branch = branch['patch_embed'](x_branch) + branch['pos_embedding'].to(x.device)  # Add positional encoding
            x_branch = branch['transformer'](x_branch)
            branch_outputs.append(x_branch)
        def forward(self, x):


# Based on SpectraFM: Tuning into Stellar Foundation Models
import torch
import torch.nn as nn

class VisionTransformer1D(nn.Module):
    def __init__(self, input_size, num_classes, patch_sizes, dim, depth, heads, mlp_dim, dropout, lambda_min=4000, lambda_max=7000):
        super(VisionTransformer1D, self).__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.branches = nn.ModuleList()

        for patch_size in patch_sizes:
            stride = patch_size
            max_patches = (input_size - patch_size) // stride + 1
            print("max_patches", max_patches)
            patch_embed = nn.Linear(patch_size, dim)
            pos_embedding = nn.Parameter(self.create_positional_encoding(max_patches, dim), requires_grad=False)  # Not in ModuleDict
            transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout), depth
            )
            
            # Store branch elements directly as attributes
            branch = nn.ModuleDict({
                'patch_embed': patch_embed,
                'transformer': transformer
            })
            branch.pos_embedding = pos_embedding  # Assign pos_embedding as an attribute
            self.branches.append(branch)

        # Cross-Attention for fusion of multiple patch sizes
        self.cross_attention = CrossAttentionBlock(dim, heads)

    def create_positional_encoding(self, num_patches, d_model):
        pos_encoding = torch.zeros((num_patches, d_model))
        for pos in range(num_patches):
            lambda_norm = (pos - self.lambda_min) / (self.lambda_max - self.lambda_min)
            for k in range(d_model):
                angle = torch.tensor(1000 * lambda_norm / (10000 ** (k / d_model)))
                if k % 2 == 0:
                    pos_encoding[pos, k] = torch.sin(angle)
                else:
                    pos_encoding[pos, k] = torch.cos(angle)
        return pos_encoding



    def forward(self, x):
        batch_size, channels, seq_len = x.shape  # Assuming x has 3 dimensions
        x = x.squeeze(1) if channels == 1 else x  # Remove channel dimension if it's 1
        branch_outputs = []

        print("branch.self", self.branches)

        for branch in self.branches:
            patch_embed = branch['patch_embed']
            transformer = branch['transformer']
            pos_embedding = branch.pos_embedding  # Access `pos_embedding` directly

            # Create patches for each branch
            print(seq_len)
            stride = patch_embed.weight.shape[1]  # Adjust based on patch size
            print("patch_embed.weight.shape[1]", patch_embed.weight.shape[1])
            print("stride", stride)
            print("self.lambda_min", self.lambda_min)
            print(stride)
            num_patches = seq_len // stride
            print(num_patches)
            patches = [x[:, i * stride : i * stride + patch_embed.weight.shape[1]] for i in range(num_patches)]
            x_branch = torch.stack(patches, dim=1)

            # Apply patch embedding and add positional encoding
            x_branch = patch_embed(x_branch) + pos_embedding.to(x.device)  # Use direct attribute access
            x_branch = transformer(x_branch)
            branch_outputs.append(x_branch)

            # Apply cross-attention to combine the representations from each branch
            x_fused = torch.cat(branch_outputs, dim=1)
            x_fused = self.cross_attention(x_fused)

            # Classification based on the first token representation
            x = self.fc(x_fused[:, 0])
            return x
class VisionTransformer1D(nn.Module):
    def __init__(self, input_size, num_classes, patch_sizes, dim, depth, heads, mlp_dim, dropout, lambda_min=4000, lambda_max=7000):
        super(VisionTransformer1D, self).__init__()
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.branches = nn.ModuleList()
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))  # Initialize class token

        for patch_size in patch_sizes:
            stride = patch_size
            max_patches = (input_size - patch_size) // stride + 1
            print("max_patches", max_patches)
            patch_embed = nn.Linear(patch_size, dim)
            pos_embedding = nn.Parameter(self.create_positional_encoding(max_patches, dim), requires_grad=False)  # Not in ModuleDict
            transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(dim, heads, mlp_dim, dropout), depth
            )
            
            # Store branch elements directly as attributes
            branch = nn.ModuleDict({
                'patch_embed': patch_embed,
                'transformer': transformer
            })
            branch.pos_embedding = pos_embedding  # Assign pos_embedding as an attribute
            self.branches.append(branch)

        # Cross-Attention for fusion of multiple patch sizes
        self.cross_attention = CrossAttentionBlock(dim, heads)

    def create_positional_encoding(self, num_patches, d_model):
        pos_encoding = torch.zeros((num_patches, d_model))
        for pos in range(num_patches):
            lambda_norm = (pos - self.lambda_min) / (self.lambda_max - self.lambda_min)
            for k in range(d_model):
                angle = torch.tensor(1000 * lambda_norm / (10000 ** (k / d_model)))
                if k % 2 == 0:
                    pos_encoding[pos, k] = torch.sin(angle)
                else:
                    pos_encoding[pos, k] = torch.cos(angle)
        return pos_encoding

    def forward(self, x):
        batch_size, channels, seq_len = x.shape  # Assuming x has 3 dimensions
        x = x.squeeze(1) if channels == 1 else x  # Remove channel dimension if it's 1
        branch_outputs = []

        for branch in self.branches:
            patch_embed = branch['patch_embed']
            transformer = branch['transformer']
            pos_embedding = branch.pos_embedding  # Access `pos_embedding` directly

            # Create patches for each branch
            stride = patch_embed.weight.shape[1]  # Adjust based on patch size
            num_patches = seq_len // stride
            patches = [x[:, i * stride : i * stride + patch_embed.weight.shape[1]] for i in range(num_patches)]
            x_branch = torch.stack(patches, dim=1)

            # Apply patch embedding and add positional encoding
            x_branch = patch_embed(x_branch) + pos_embedding.to(x.device)  # Use direct attribute access
            x_branch = transformer(x_branch)
            branch_outputs.append(x_branch)

        # Concatenate branch outputs and add class token
        x = torch.cat(branch_outputs, dim=1)
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)

        # Apply cross-attention
        x = self.cross_attention(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., theta):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.theta = theta

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias) 
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize rotary positional encoding
        self.rotary = Rotary(dim // num_heads, base=theta)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).view(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Apply rotary position embedding
        cos, sin = self.rotary(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention calculation with rotated embeddings
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
import torch
import matplotlib.pyplot as plt

# Define the Rotary class
class Rotary(torch.nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached

# Initialize the Rotary positional embedding
dim = 16
base = 30
rotary_pos_emb = Rotary(dim=dim, base=base)

# Create a dummy input tensor
seq_len = 50
batch_size = 1
x = torch.zeros((batch_size, seq_len, dim))

# Get the cos and sin embeddings
cos_emb, sin_emb = rotary_pos_emb(x)

# Plot the angles for the first dimension of the positional embeddings
plt.figure(figsize=(12, 6))
plt.plot(cos_emb[0, :, 0, 0].cpu().numpy(), label='cos')
plt.plot(sin_emb[0, :, 0, 0].cpu().numpy(), label='sin')
plt.title('Rotary Positional Embedding Angles')
plt.xlabel('Sequence Position')
plt.ylabel('Angle')
plt.legend()
plt.show()
import numpy as np
import torch
import matplotlib.pyplot as plt

# Parameters
base = 10000
dim = 256

# Calculate inverse frequencies
i = np.arange(0, dim, 2)
inv_freq = 1.0 / (base ** (i / dim))
inv_freq = torch.tensor(inv_freq)

# Generate sequence positions
t = torch.arange(4000)

# Calculate frequencies
freqs = torch.einsum("i,j->ij", t, inv_freq)

# Calculate embeddings
emb = torch.cat((freqs, freqs), dim=-1)

# Calculate cos and sin
cos = emb.cos()
sin = emb.sin()

# Plot the angles for the first dimension
plt.figure(figsize=(12, 6))
plt.plot(cos[:, 1].numpy(), label='cos')
plt.plot(sin[:, 1].numpy(), label='sin')
plt.title('Rotary Positional Embedding Angles for Dimension 0')
plt.xlabel('Sequence Position')
plt.ylabel('Angle')
plt.legend()
plt.show()



    import torch
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate rotation angles
def compute_rotation_angles(seq_len, dim, base=10000):
    positions = torch.arange(seq_len).unsqueeze(1).float()  # Shape: (seq_len, 1)
    per_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, per_dim).float() / per_dim))  # Shape: (dim/2,)
    angles = positions * inv_freq  # Broadcasting: Shape (seq_len, dim/2)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

def init_rope_frequencies(dim, num_heads, base=10000):
    per_head_dim = dim // num_heads
    inv_freq = 1.0 / (base ** (torch.arange(0, per_head_dim, 2).float() / per_head_dim))
    return torch.cat((torch.cos(inv_freq), torch.sin(inv_freq)), dim=-1)
# Example configuration
seq_len = 3748  # Number of positions in the sequence
dim = 1024 # Dimensionality of embedding vector
num_heads = 1  # Number of attention heads
theta = 10000 # Hyperparameter controlling frequency scaling

# Initialize frequencies
freqs = init_rope_frequencies(dim, num_heads, theta)

# Compute rotation angles
angles = compute_rotation_angles(seq_len, dim, base=theta)

#print(angles[:10, :10])  # Print the first few rows and columns

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.imshow(angles, aspect='auto', cmap='viridis', extent=[0, seq_len, 0, dim // num_heads])
plt.colorbar(label="Rotation Angle (radians)")
plt.title("Rotary Position Embedding Rotation Angles")
plt.xlabel("Sequence Position")
plt.ylabel("Embedding Dimension (per head)")
plt.show()class DualMambaClassifier(nn.Module):
    def __init__(self, gaia_dim, spectra_dim, d_model, num_classes, d_state=64, d_conv=4, n_layers=6):
        super(DualMambaClassifier, self).__init__()
        # MAMBA model for Gaia data
        self.gaia_model = StarClassifierMAMBA(d_model, num_classes, d_state, d_conv, gaia_dim, n_layers)
        # MAMBA model for spectra data
        self.spectra_model = StarClassifierMAMBA(d_model, num_classes, d_state, d_conv, spectra_dim, n_layers)
        # Cross attention block
        self.gaia_model.input_projection = nn.Linear(17, d_model)  # Gaia input
        self.spectra_model.input_projection = nn.Linear(3749, d_model)  # Spectra input

        print("Shape of Gaia input projection: ", self.gaia_model.input_projection)
        print("Shape of Spectra input projection: ", self.spectra_model.input_projection)
        self.cross_attention = CrossAttentionBlock(dim=d_model*2, num_heads=8)
        print("Shape of cross attention: ", self.cross_attention)


        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, gaia_x, spectra_x):
        gaia_features = self.gaia_model.mamba_layer(self.gaia_model.input_projection(gaia_x).unsqueeze(1))
        spectra_features = self.spectra_model.mamba_layer(self.spectra_model.input_projection(spectra_x).unsqueeze(1))

        # Cross attention: allowing information sharing between modalities
        combined_features = torch.cat([gaia_features, spectra_features], dim=1)
        print(f'Combined features shape: {combined_features.shape}')

        fused_features = self.cross_attention(combined_features)
        print(f'Fused features shape: {fused_features.shape}')

        # Global average pooling and classification
        pooled_features = fused_features.mean(dim=1)
        print(f'Pooled features shape: {pooled_features.shape}')

        output = self.classifier(pooled_features)

        return output
    def train_model_mamba_fusion(
    model, train_loader, val_loader, test_loader, 
    num_epochs=500, lr=1e-4, max_patience=20, device='cuda'
):
    # Move model to device
    model = model.to(device)

    # Define optimizer, scheduler, and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=int(max_patience / 3), verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = max_patience

    for epoch in range(num_epochs):
        # Resample training and validation data if needed
        train_loader.dataset.re_sample()
        val_loader.dataset.balance_classes()

        # Training phase
        model.train()
        train_loss, train_accuracy = 0.0, 0.0

        for spectra_batch, gaia_batch, y_batch in train_loader:
            spectra_batch, gaia_batch, y_batch = (
                spectra_batch.to(device),
                gaia_batch.to(device),
                y_batch.to(device)
            )
            optimizer.zero_grad()
            outputs = model(spectra_batch, gaia_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * spectra_batch.size(0)
            train_accuracy += (outputs.argmax(dim=1) == y_batch).float().mean().item()

        # Validation phase
        model.eval()
        val_loss, val_accuracy = 0.0, 0.0
        with torch.no_grad():
            for spectra_val, gaia_val, y_val in val_loader:
                spectra_val, gaia_val, y_val = (
                    spectra_val.to(device),
                    gaia_val.to(device),
                    y_val.to(device)
                )
                outputs = model(spectra_val, gaia_val)
                loss = criterion(outputs, y_val)

                val_loss += loss.item() * spectra_val.size(0)
                val_accuracy += (outputs.argmax(dim=1) == y_val).float().mean().item()

        # Test phase and metric collection
        test_loss, test_accuracy = 0.0, 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for spectra_test, gaia_test, y_test in test_loader:
                spectra_test, gaia_test, y_test = (
                    spectra_test.to(device),
                    gaia_test.to(device),
                    y_test.to(device)
                )
                outputs = model(spectra_test, gaia_test)
                loss = criterion(outputs, y_test)

                test_loss += loss.item() * spectra_test.size(0)
                test_accuracy += (outputs.argmax(dim=1) == y_test).float().mean().item()
                y_true.extend(y_test.cpu().numpy())
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

        # Update scheduler
        scheduler.step(val_loss / len(val_loader.dataset))

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
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=y_pred, class_names=np.unique(y_true)
            ),
            "classification_report": classification_report(
                y_true, y_pred, target_names=[str(i) for i in range(len(np.unique(y_true)))]
            )
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            best_model = model.state_dict()
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    # Load the best model weights
    model.load_state_dict(best_model)
    return model
class DualMambaClassifier(nn.Module):
    def __init__(self, gaia_dim, spectra_dim, d_model, num_classes, d_state=64, d_conv=4, n_layers=6):
        super(DualMambaClassifier, self).__init__()
        # MAMBA model for Gaia data
        self.gaia_model = StarClassifierMAMBA(d_model, num_classes, d_state, d_conv, gaia_dim, n_layers)
        # MAMBA model for spectra data
        self.spectra_model = StarClassifierMAMBA(d_model, num_classes, d_state, d_conv, spectra_dim, n_layers)
        # Cross attention block
        self.gaia_model.input_projection = nn.Linear(gaia_dim, d_model)  # Gaia input
        self.spectra_model.input_projection = nn.Linear(spectra_dim, d_model)  # Spectra input

        print("Shape of Gaia input projection: ", self.gaia_model.input_projection)
        print("Shape of Spectra input projection: ", self.spectra_model.input_projection)
        self.cross_attention = CrossAttentionBlock(dim=d_model*2, num_heads=8)
        print("Shape of cross attention: ", self.cross_attention)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, gaia_x, spectra_x):
        print(f'Gaia input shape: {gaia_x.shape}')
        print(f'Spectra input shape: {spectra_x.shape}')
        
        # Ensure the inputs are correctly aligned
        if gaia_x.shape[1] != 17 or spectra_x.shape[1] != 3748:
            raise ValueError("Input dimensions do not match the expected dimensions for Gaia and spectra data.")

        gaia_features = self.gaia_model.mamba_layer(self.gaia_model.input_projection(gaia_x).unsqueeze(1))
        spectra_features = self.spectra_model.mamba_layer(self.spectra_model.input_projection(spectra_x).unsqueeze(1))

        # Cross attention: allowing information sharing between modalities
        combined_features = torch.cat([gaia_features, spectra_features], dim=1)
        print(f'Combined features shape: {combined_features.shape}')

        fused_features = self.cross_attention(combined_features)
        print(f'Fused features shape: {fused_features.shape}')

        # Global average pooling and classification
        pooled_features = fused_features.mean(dim=1)
        print(f'Pooled features shape: {pooled_features.shape}')

        output = self.classifier(pooled_features)

        return outputdef train_model_mamba_fusion(
    model, train_loader, val_loader, test_loader, 
    num_epochs=500, lr=1e-4, max_patience=20, device='cuda'
):
    # Move model to device
    model = model.to(device)

    # Define optimizer, scheduler, and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=int(max_patience / 3), verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = max_patience

    for epoch in range(num_epochs):
        # Resample training and validation data if needed
        train_loader.dataset.re_sample()
        val_loader.dataset.balance_classes()

        # Training phase
        model.train()
        train_loss, train_accuracy = 0.0, 0.0

        for spectra_batch, gaia_batch, y_batch in train_loader:
            spectra_batch, gaia_batch, y_batch = (
                spectra_batch.to(device),
                gaia_batch.to(device),
                y_batch.to(device)
            )
            optimizer.zero_grad()
            # Ensure the correct order of inputs
            outputs = model(gaia_batch, spectra_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * spectra_batch.size(0)
            train_accuracy += (outputs.argmax(dim=1) == y_batch).float().mean().item()

        # Validation phase
        model.eval()
        val_loss, val_accuracy = 0.0, 0.0
        with torch.no_grad():
            for spectra_val, gaia_val, y_val in val_loader:
                spectra_val, gaia_val, y_val = (
                    spectra_val.to(device),
                    gaia_val.to(device),
                    y_val.to(device)
                )
                outputs = model(gaia_val, spectra_val)
                loss = criterion(outputs, y_val)

                val_loss += loss.item() * spectra_val.size(0)
                val_accuracy += (outputs.argmax(dim=1) == y_val).float().mean().item()

        # Test phase and metric collection
        test_loss, test_accuracy = 0.0, 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for spectra_test, gaia_test, y_test in test_loader:
                spectra_test, gaia_test, y_test = (
                    spectra_test.to(device),
                    gaia_test.to(device),
                    y_test.to(device)
                )
                outputs = model(gaia_test, spectra_test)
                loss = criterion(outputs, y_test)

                test_loss += loss.item() * spectra_test.size(0)
                test_accuracy += (outputs.argmax(dim=1) == y_test).float().mean().item()
                y_true.extend(y_test.cpu().numpy())
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

        # Update scheduler
        scheduler.step(val_loss / len(val_loader.dataset))

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
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=y_pred, class_names=np.unique(y_true)
            ),
            "classification_report": classification_report(
                y_true, y_pred, target_names=[str(i) for i in range(len(np.unique(y_true)))]
            )
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            best_model = model.state_dict()
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    # Load the best model weights
    model.load_state_dict(best_model)
    return model

def train_model_mamba(
    model, train_loader, val_loader, test_loader, 
    num_epochs=500, lr=1e-4, max_patience=20, device='cuda'
):
    # Move model to device
    model = model.to(device)

    # Define optimizer, scheduler, and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=int(max_patience / 3), verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = max_patience

    for epoch in range(num_epochs):
        # Resample training and validation data
        train_loader.dataset.re_sample()
        val_loader.dataset.balance_classes()

        # Training phase
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

        # Test phase and metric collection
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

        # Update scheduler
        scheduler.step(val_loss / len(val_loader.dataset))

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
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=y_pred, class_names=np.unique(y_true)
            ),
            "classification_report": classification_report(
                y_true, y_pred, target_names=[str(i) for i in range(len(np.unique(y_true)))]
            )
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            best_model = model.state_dict()
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    # Load the best model weights
    model.load_state_dict(best_model)
    return model

# Training loop
def train_model_mamba_fusion(
    model, train_loader, val_loader, test_loader, 
    num_epochs=500, lr=1e-4, max_patience=20, device='cuda'
):
    # Move model to device
    model = model.to(device)

    # Define optimizer, scheduler, and loss function
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=int(max_patience / 3), verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = max_patience

    for epoch in range(num_epochs):
        # Resample training and validation data if needed
        train_loader.dataset.re_sample()
        val_loader.dataset.balance_classes()

        # Training phase
        model.train()
        train_loss, train_accuracy = 0.0, 0.0

        for spectra_batch, gaia_batch, y_batch in train_loader:
            spectra_batch, gaia_batch, y_batch = (
                spectra_batch.to(device),
                gaia_batch.to(device),
                y_batch.to(device)
            )
            optimizer.zero_grad()
            # Ensure the correct order of inputs
            outputs = model(gaia_batch, spectra_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * spectra_batch.size(0)
            train_accuracy += (outputs.argmax(dim=1) == y_batch).float().mean().item()

        # Validation phase
        model.eval()
        val_loss, val_accuracy = 0.0, 0.0
        with torch.no_grad():
            for spectra_val, gaia_val, y_val in val_loader:
                spectra_val, gaia_val, y_val = (
                    spectra_val.to(device),
                    gaia_val.to(device),
                    y_val.to(device)
                )
                outputs = model(gaia_val, spectra_val)
                loss = criterion(outputs, y_val)

                val_loss += loss.item() * spectra_val.size(0)
                val_accuracy += (outputs.argmax(dim=1) == y_val).float().mean().item()

        # Test phase and metric collection
        test_loss, test_accuracy = 0.0, 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for spectra_test, gaia_test, y_test in test_loader:
                spectra_test, gaia_test, y_test = (
                    spectra_test.to(device),
                    gaia_test.to(device),
                    y_test.to(device)
                )
                outputs = model(gaia_test, spectra_test)
                loss = criterion(outputs, y_test)

                test_loss += loss.item() * spectra_test.size(0)
                test_accuracy += (outputs.argmax(dim=1) == y_test).float().mean().item()
                y_true.extend(y_test.cpu().numpy())
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

        # Update scheduler
        scheduler.step(val_loss / len(val_loader.dataset))

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
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=y_pred, class_names=np.unique(y_true)
            ),
            "classification_report": classification_report(
                y_true, y_pred, target_names=[str(i) for i in range(len(np.unique(y_true)))]
            )
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            best_model = model.state_dict()
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    # Load the best model weights
    model.load_state_dict(best_model)
    return model

class DualMambaClassifier(nn.Module):
    def __init__(self, gaia_dim, spectra_dim, d_model, num_classes, d_state=64, d_conv=4, n_layers=6):
        super(DualMambaClassifier, self).__init__()
        # MAMBA model for Gaia data
        self.gaia_model = StarClassifierMAMBA(d_model, num_classes, d_state, d_conv, gaia_dim, n_layers)
        # MAMBA model for spectra data
        self.spectra_model = StarClassifierMAMBA(d_model, num_classes, d_state, d_conv, spectra_dim, n_layers)
        # Cross attention block
        self.gaia_model.input_projection = nn.Linear(gaia_dim, d_model)  # Gaia input
        self.spectra_model.input_projection = nn.Linear(spectra_dim, d_model)  # Spectra input

        print("Shape of Gaia input projection: ", self.gaia_model.input_projection)
        print("Shape of Spectra input projection: ", self.spectra_model.input_projection)
        self.cross_attention = CrossAttentionBlock(dim=d_model*2, num_heads=8)
        print("Shape of cross attention: ", self.cross_attention)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, gaia_x, spectra_x):
        print(f'Gaia input shape: {gaia_x.shape}')
        print(f'Spectra input shape: {spectra_x.shape}')
        
        
        # Ensure the inputs are correctly aligned
        if gaia_x.shape[1] != 17 or spectra_x.shape[2] != 3748:
            raise ValueError("Input dimensions do not match the expected dimensions for Gaia and spectra data.")
        


        gaia_features = self.gaia_model.mamba_layer(self.gaia_model.input_projection(gaia_x).unsqueeze(1))
        spectra_features = self.spectra_model.mamba_layer(self.spectra_model.input_projection(spectra_x.squeeze(1)))
        # Forward pass through the MAMBA models
        gaia_features = self.gaia_model(gaia_x)
        spectra_features = self.spectra_model(spectra_x)


        # Combine the representations from each branch
        branch_outputs = [gaia_features, spectra_features]
        combined_features = torch.cat(branch_outputs, dim=1)
        print(f'Combined features shape: {combined_features.shape}')
        
        # Apply cross-attention
        fused_features = self.cross_attention(combined_features)
        print(f'Fused features shape: {fused_features.shape}')
        
        # Global average pooling and classification
        pooled_features = fused_features.mean(dim=1)
        print(f'Pooled features shape: {pooled_features.shape}')
        
        output = self.classifier(pooled_features)
        return output
# Select the relevant gaia columns to normalize
gaia_cols = ["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error",
                "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux",
                "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "flagnoflux"]

# Get the train and test data for gaia normalization
train_gaia = train_data[gaia_cols]

# Initialize the PowerTransformer for Yeo-Johnson transformation
pt = PowerTransformer(method='yeo-johnson', standardize=True)

# Fit the PowerTransformer for each column and store the lambda values
lambdas = {}
for column in train_gaia.columns:
    pt.fit(train_gaia[[column]])
    lambdas[column] = pt.lambdas_

# Display the lambda values for each column
for column, lambda_value in lambdas.items():
    print(f"Lambda value for {column}: {lambda_value}")

print(column, lambda_value)

print(lambdas.items())
print(train_gaia.columns)

# Apply the Yeo-Johnson transformation to the train data with the lambda values
for column in train_gaia.columns:
    print(column)
    train_gaia[column] = pt.transform(train_gaia[[column]])


batch_size = 16

# If X exists, delete it
if 'X' in locals():   
    del X, y
gc.collect()

# Example usage
if __name__ == "__main__":
    # Load and preprocess your data (example from original script)
    # Load and preprocess data
    X = pd.read_pickle("Pickles/train_data_transformed.pkl")
    classes = pd.read_pickle("Pickles/List_of_Classes.pkl")
    y = X[classes]

    #label_mapping = {'star': 0, 'binary_star': 1, 'galaxy': 2, 'agn': 3}
    #y = y.map(label_mapping).values

    # Drop gaia data
    X = X.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
                "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
                "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "obsid"], axis=1).values
    # Drop labels
    X = X.drop(classes, axis=1).values

    print(X.shape, y.shape)
    
    # Read test data
    X_test = pd.read_pickle("Pickles/testv2.pkl")
    y_test = X_test[classes]
    X_test = X_test.drop(["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", "pmra", "pmdec", "pmra_error", "pmdec_error", 
                "phot_g_mean_flux", "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", "phot_rp_mean_flux", 
                "phot_bp_mean_flux_error", "phot_rp_mean_flux_error", "obsid"], axis=1).values
    X_test = X_test.drop(classes, axis=1).values

    print(X_test.shape, y_test.shape)
    
    # Split validation data
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
    # Define the model with your parameters
d_model = 2048 # Embedding dimension
num_classes = 4  # Star classification categories
input_dim = 3647

# Define the training parameters
num_epochs = 500
lr = 2e-7
patience = 50   
depth = 6

# Define the config dictionary object
config = {"num_classes": num_classes, "batch_size": batch_size, "lr": lr, "patience": patience, "num_epochs": num_epochs, "d_model": d_model, "depth": depth}

# Initialize WandB project
wandb.init(project="lamost-mamba-test", entity="joaoc-university-of-southampton", config=config)
# Initialize and train the model
# Train the model using your `train_model_vit` or an adjusted training loop
model_mamba = StarClassifierMAMBA(d_model=d_model, num_classes=num_classes, input_dim=input_dim, n_layers=depth)
print(model_mamba)
# print number of parameters per layer
for name, param in model_mamba.named_parameters():
    print(name, param.numel())
print("Total number of parameters:", sum(p.numel() for p in model_mamba.parameters() if p.requires_grad))

# Move the model to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_mamba = model_mamba.to(device)

# Train the model using your `train_model_vit` or an adjusted training loop
trained_model = train_model_mamba(
    model=model_mamba,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_epochs=num_epochs,
    lr=lr,
    max_patience=patience,
    device=device
)
# Save the model and finish WandB session
wandb.finish()
    
# Load the filtered DataFrame and the updated list of label columns
filtered_df = pd.read_pickle("Pickles/filtered_multi_hot_encoded.pkl")
updated_label_columns = pd.read_pickle("Pickles/Updated_List_of_Classes.pkl")

# Step 4: Prepare X (features) and y (labels) for stratified splitting
X = filtered_df.drop(columns=updated_label_columns)  # Features
y = filtered_df[updated_label_columns]  # Multi-labels (multi-hot encoding)

# Step 5: Use MultilabelStratifiedKFold to split the data
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in mskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    break  # Take only the first split to simulate a train-test split

# Recombine the train and test DataFrames if needed
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save train and test sets if needed
train_df.to_pickle("Pickles/train_df.pkl")
test_df.to_pickle("Pickles/test_df.pkl")



class BalancedDataset(Dataset):
    def __init__(self, X, y, limit_per_label=201):
        self.X = X
        self.y = y
        self.limit_per_label = limit_per_label
        self.classes = np.unique(y)
        self.indices = self.balance_classes()

    def balance_classes(self):
        indices = []
        for cls in self.classes:
            cls_indices = np.where(self.y == cls)[0]
            # Set limit per label except for the * label
            if cls == "*":
                print("Skipping limit for * label")
            elif len(cls_indices) > self.limit_per_label:
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
    def __init__(self, X, y, limit_per_label=100):
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
    
    # Load the merged data and the list of classes
df = pd.read_pickle("Pickles/merged_multi_hot_encoded_no_star.pkl")
label_columns = pd.read_pickle("Pickles/List_of_Classes_no_star.pkl")

# Step 1: Count the occurrences of each label (sum of 1s in each column)
label_counts = df[label_columns].sum()

# Step 2: Drop label columns (classes) that have fewer than 2 occurrences
frequent_labels = label_counts[label_counts >= 2].index
#filtered_df = df[frequent_labels.union(df.columns.difference(label_columns))]  # Keep non-label columns
non_label_columns = df.columns.difference(label_columns)
filtered_df = df[list(frequent_labels) + list(non_label_columns)]

# Step 3: Remove samples (rows) that have no active labels
mask = filtered_df[frequent_labels].sum(axis=1) > 0  # Keep only rows with at least one active label
filtered_df = filtered_df.loc[mask]

# Update the list of label columns (since some columns were dropped)
updated_label_columns = [col for col in frequent_labels if col in filtered_df.columns]

# Save the updated list of label columns and the filtered DataFrame
pd.to_pickle(updated_label_columns, "Pickles/Updated_List_of_Classes.pkl")
filtered_df.to_pickle("Pickles/filtered_multi_hot_encoded.pkl")

def calculate_sample_weights(y):
    """
    Compute per-sample weights based on the inverse frequency of the labels in a multi-label setting.

    Args:
    - y (numpy.ndarray): Multi-hot encoded labels (num_samples, num_classes)

    Returns:
    - sample_weights (numpy.ndarray): Weight per sample (num_samples,)
    """
    class_counts = np.sum(y, axis=0)  # Number of times each class appears
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0)  # Inverse class frequency

    # Compute per-sample weight as the sum of its class weights
    sample_weights = np.sum(y * class_weights, axis=1)
    return sample_weights


class BalancedMultiLabelDataset(Dataset):
    def __init__(self, X, y, limit_per_label=201):
        self.X = X
        self.y = y
        self.limit_per_label = limit_per_label
        self.num_classes = y.shape[1]
        self.indices = self.balance_classes()
        self.sample_weights = torch.tensor(calculate_sample_weights(y.numpy()), dtype=torch.float)

    def balance_classes(self):
        indices = []
        for cls in range(self.num_classes):
            cls_indices = np.where(self.y[:, cls] == 1)[0]
            if len(cls_indices) < self.limit_per_label:
                extra_indices = np.random.choice(cls_indices, self.limit_per_label - len(cls_indices), replace=True)
                cls_indices = np.concatenate([cls_indices, extra_indices])
            elif len(cls_indices) > self.limit_per_label:
                cls_indices = np.random.choice(cls_indices, self.limit_per_label, replace=False)
            indices.extend(cls_indices)
        np.random.shuffle(indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.X[index], self.y[index], self.sample_weights[index]  # Include per-sample weight
    
    
def train_model_mamba(
    model, train_loader, val_loader, test_loader, 
    num_epochs=500, lr=1e-4, max_patience=20, device='cuda'
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=int(max_patience / 5)
    )
    
    best_val_loss = float('inf')
    patience = max_patience

    for epoch in range(num_epochs):
        train_loader.dataset.re_sample()  # Resample training data
        
        # Compute sample weights
        all_labels = []
        for _, y_batch in train_loader:
            all_labels.extend(y_batch.cpu().numpy())

        sample_weights = calculate_sample_weights(np.array(all_labels))
        sample_weights = torch.tensor(sample_weights, dtype=torch.float).to(device)

        # Define weighted loss function
        criterion = nn.BCEWithLogitsLoss(reduction='none')  # No reduction yet, as we will weight manually

        model.train()
        train_loss = 0.0

        for X_batch, y_batch, w_batch in zip(train_loader, sample_weights):
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)  # Shape: (batch_size, num_classes)

            # Apply per-sample weights (element-wise multiplication)
            loss = (loss * w_batch.unsqueeze(1)).mean()  # Apply weight per sample and average

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        # Validation loop (no per-sample weighting needed)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val).mean()
                val_loss += loss.item() * X_val.size(0)

        scheduler.step(val_loss / len(val_loader.dataset))

        # Log results
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader.dataset),
            "val_loss": val_loss / len(val_loader.dataset),
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            best_model = model.state_dict()
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)
    return model

def predict_star_labels(gaia_ids, model_path, lamost_catalog, output_pickle="predicted_labels.pkl"):
    """
    Given a list of Gaia DR3 IDs, this function:
    1) Queries Gaia for star parameters.
    2) Cross-matches with LAMOST spectra.
    3) Downloads and processes LAMOST spectra.
    4) Normalizes both Gaia and LAMOST data.
    5) Applies a trained StarClassifierFusion model to predict labels.
    
    Args:
        gaia_ids (list): List of Gaia DR3 source IDs.
        model_path (str): Path to the trained model file.
        lamost_catalog (str): Path to the LAMOST catalog CSV.
        output_pickle (str): Path to save the final predictions.
    
    Returns:
        DataFrame with Gaia IDs and predicted multi-label classifications.
    """

    print("\n🚀 Step 1: Querying Gaia data...")
    df_gaia = query_gaia_data(gaia_ids)
    if df_gaia.empty:
        print("⚠️ No Gaia data found. Exiting.")
        return None

    print("\n🔄 Step 2: Cross-matching with LAMOST catalog...")
    df_matched = crossmatch_lamost(df_gaia, lamost_catalog)
    if df_matched.empty:
        print("⚠️ No LAMOST matches found. Exiting.")
        return None

    print("\n📥 Step 3: Downloading LAMOST spectra (if needed)...")
    obsids = df_matched["obsid"].unique()
    spectra_folder = "lamost_spectra"
    downloaded_obsids = download_lamost_spectra(obsids, save_folder=spectra_folder, num_workers=50)

    print("\n🔧 Step 4: Processing LAMOST spectra...")
    df_interpolated, failed_files = process_lamost_spectra(spectra_folder)

    if df_interpolated.empty:  # ✅ Now correctly checking the DataFrame
        print("⚠️ No processed LAMOST spectra found. Exiting.")
        return None

    print("\n📊 Step 5: Normalizing LAMOST and Gaia features...")
    lamost_normalized = normalize_lamost_spectra(df_interpolated)
    gaia_normalized = apply_gaia_transforms(df_gaia, gaia_transformers)

    print("\n🔗 Step 6: Merging normalized Gaia and LAMOST data...")
    gaia_lamost_match = df_matched[["source_id", "obsid"]]
    lamost_normalized["obsid"] = lamost_normalized["obsid"].astype(int)
    gaia_lamost_match["obsid"] = gaia_lamost_match["obsid"].astype(int)

    lamost_normalized["source_id"] = lamost_normalized["obsid"].map(gaia_lamost_match.set_index("obsid")["source_id"])
    gaia_lamost_merged = pd.merge(gaia_normalized, lamost_normalized, on="source_id", how="inner")

    if gaia_lamost_merged.empty:
        print("⚠️ No valid data after merging. Exiting.")
        return None

    print("\n🤖 Step 7: Predicting labels using the trained model...")
    predictions = process_star_data_fusion(
        model_path=model_path,
        data_path=gaia_lamost_merged,
        classes_path="Pickles/Updated_list_of_Classes.pkl",
        sigmoid_constant=0.5
    )

    print("\n💾 Step 8: Saving predictions...")
    df_predictions = pd.DataFrame(predictions, columns=pd.read_pickle("Pickles/Updated_list_of_Classes.pkl"))
    df_predictions["source_id"] = gaia_lamost_merged["source_id"].values
    df_predictions.to_pickle(output_pickle)

    print(f"\n✅ Predictions saved to {output_pickle}")
    return df_predictions


def query_gaia_data(gaia_id_list):
    """
    Given a list of Gaia DR3 source IDs, queries the Gaia archive
    for the relevant columns used during training.
    Returns a concatenated DataFrame of results.
    """
    # Columns you actually need (adapt to match your pipeline!)
    # e.g. ra, dec, pmra, pmdec, phot_g_mean_flux, ...
    desired_cols = [
        "source_id", "ra", "ra_error", "dec", "dec_error",
        "pmra", "pmra_error", "pmdec", "pmdec_error",
        "parallax", "parallax_error",
        "phot_g_mean_flux", "phot_g_mean_flux_error",
        "phot_bp_mean_flux", "phot_bp_mean_flux_error",
        "phot_rp_mean_flux", "phot_rp_mean_flux_error"
    ]

    all_dfs = []
    print(f"🔍 Unique Gaia IDs before query: {len(set(gaia_id_list))}")
    chunks = split_ids_into_chunks(gaia_id_list, chunk_size=30000)
    for chunk in chunks:
        query = f"""
        SELECT {', '.join(desired_cols)}
        FROM gaiadr3.gaia_source
        WHERE source_id IN ({chunk})
        """
        job = Gaia.launch_job_async(query)
        tbl = job.get_results()
        df_tmp = tbl.to_pandas()
        all_dfs.append(df_tmp)

    # Print a warning if some IDs were not found
    all_ids = pd.concat(all_dfs)["source_id"].values
    print(f"🔍 Found {len(all_ids)} IDs in Gaia DR3.")
    missing_ids = set(gaia_id_list) - set(all_ids)
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs not found in Gaia DR3.")
        print(f"Missing IDs: {missing_ids}")

    print(set(all_ids).difference(set(gaia_id_list)))  # Extra IDs
    print(set(gaia_id_list).difference(set(all_ids)))  # Missing IDs


    if not all_dfs:
        return pd.DataFrame(columns=desired_cols)
    else:
        return pd.concat(all_dfs, ignore_index=True)        
    

        gaia_lamost_match = df_matched[["source_id", "obsid"]]

    # Troubleshooting duplicate obsids
    duplicate_obsids = gaia_lamost_match["obsid"].duplicated().sum()
    print(f"⚠️ Found {duplicate_obsids} duplicated obsid values in Gaia-LAMOST match.")
    print(gaia_lamost_match["obsid"].value_counts().head(10))


    spectrum_normalized["source_id"] = spectrum_normalized["obsid"].astype(int).map(gaia_lamost_match.set_index("obsid")["source_id"])
    gaia_lamost_merged = pd.merge(gaia_normalized, spectrum_normalized, on="source_id", how="inner")

    if gaia_lamost_merged.empty:
        print("⚠️ No valid data after merging. Exiting.")
        return None
    
    
def crossmatch_lamost(gaia_df, lamost_df, match_radius=3*u.arcsec):
    """
    Cross-matches Gaia sources with a local LAMOST catalogue.
    Returns a merged DataFrame of matched objects.
    """

    # Ensure RA/Dec are numeric
    gaia_df['ra'] = pd.to_numeric(gaia_df['ra'], errors='coerce')
    gaia_df['dec'] = pd.to_numeric(gaia_df['dec'], errors='coerce')
    lamost_df['ra'] = pd.to_numeric(lamost_df['ra'], errors='coerce')
    lamost_df['dec'] = pd.to_numeric(lamost_df['dec'], errors='coerce')

    # Drop NaN values
    gaia_df = gaia_df.dropna(subset=['ra', 'dec'])
    lamost_df = lamost_df.dropna(subset=['ra', 'dec'])

    print("🔍 Gaia RA/Dec sample:", gaia_df[['ra', 'dec']].head())
    print("🔍 LAMOST RA/Dec sample:", lamost_df[['ra', 'dec']].head())


    print(f"After NaN removal: Gaia={gaia_df.shape}, LAMOST={lamost_df.shape}")

    # Check if LAMOST coordinates are in arcseconds (convert if necessary)
    if lamost_df['ra'].max() > 360:  # RA should not exceed 360 degrees
        print("⚠️ LAMOST RA/Dec seem to be in arcseconds. Converting to degrees.")
        lamost_df['ra'] /= 3600
        lamost_df['dec'] /= 3600

    # Convert to SkyCoord objects in arcseconds (ensuring same frame)
    gaia_coords = SkyCoord(ra=gaia_df['ra'].values*u.deg,
                           dec=gaia_df['dec'].values*u.deg,
                           frame='icrs')

    lamost_coords = SkyCoord(ra=lamost_df['ra'].values*u.deg,
                             dec=lamost_df['dec'].values*u.deg,
                             frame='icrs')

    # Perform crossmatch
    idx, d2d, _ = gaia_coords.match_to_catalog_sky(lamost_coords)

    # Apply matching radius filter
    matches = d2d < match_radius

    # Print match distances only for successful matches
    print(f"Maximum match distance (arcsec): {d2d[matches].max()*3600}")
    print(f"Mean match distance (arcsec): {d2d[matches].mean()*3600}")

    if matches.sum() == 0:
        print("⚠️ No matches found! Try increasing `match_radius`.")
        return pd.DataFrame()

    # Extract matched rows correctly
    gaia_matched = gaia_df.iloc[matches].copy().reset_index(drop=True)
    #lamost_matched = lamost_df.iloc[idx[matches]].copy().reset_index(drop=True) 
    lamost_matched = lamost_df.iloc[np.where(matches)].copy().reset_index(drop=True)


    print(f"Matched Gaia Objects: {gaia_matched.shape}")
    print(f"Matched LAMOST Objects: {lamost_matched.shape}")

    # Merge matches into final DataFrame
    final = pd.concat([gaia_matched, lamost_matched], axis=1)

    return final

def crossmatch_lamost(gaia_df, lamost_df, match_radius=3*u.arcsec):
    """
    Cross-match the Gaia DataFrame with a local LAMOST catalogue CSV
    (which must have 'ra' and 'dec' columns).
    Returns a merged DataFrame containing only matched objects, plus LAMOST obsid, etc.
    """
    # Basic cleaning
    lamost_df = lamost_df.dropna(subset=['ra','dec'])
    gaia_df = gaia_df.dropna(subset=['ra','dec'])

    # Create astropy SkyCoord objects
    gaia_coords   = SkyCoord(ra=gaia_df['ra'].values*u.deg,
                             dec=gaia_df['dec'].values*u.deg)
    lamost_coords = SkyCoord(ra=lamost_df['ra'].values*u.deg,
                             dec=lamost_df['dec'].values*u.deg)
    # Print the coordinates
    print(f"Gaia Coords: {gaia_coords[:3]}")
    print(f"LAMOST Coords: {lamost_coords[:3]}")

    # Match to catalog
    idx, sep2d, _ = gaia_coords.match_to_catalog_sky(lamost_coords)
    matches = sep2d < match_radius

    # Subset
    gaia_matched   = gaia_df.iloc[matches].copy().reset_index(drop=True)
    lamost_matched = lamost_df.iloc[idx[matches]].copy().reset_index(drop=True)

    # Merge into single DataFrame
    final = pd.concat([gaia_matched, lamost_matched], axis=1)
    return final


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.io import fits

def plot_spectrum_with_gaia(source_id, gaia_lamost_merged, spectra_folder="lamost_spectra_uniques"):
    """
    Plots the LAMOST spectrum from FITS files and displays Gaia parameters below it.
    
    :param source_id: Gaia Source ID of the incorrectly classified source
    :param gaia_lamost_merged: DataFrame containing Gaia and LAMOST cross-matched data
    :param spectra_folder: Path to the folder containing LAMOST FITS spectra
    """
    try:
        # Ensure 'obsid' column exists
        if 'obsid' not in gaia_lamost_merged.columns:
            print(f"⚠️ 'obsid' column not found in gaia_lamost_merged.")
            return
        
        match = gaia_lamost_merged.loc[gaia_lamost_merged['source_id'] == source_id]
        if match.empty:
            print(f"⚠️ No LAMOST match found for source_id {source_id}.")
            return
        
        obsid = int(match.iloc[0]['obsid'])  # Ensure obsid is an integer
        print(f"Found match: Source ID {source_id} -> ObsID {obsid}")
        
        # Construct FITS file path
        fits_path = f"{spectra_folder}/{int(obsid)}"
        
        # Load FITS data
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            if data is None or data.shape[0] < 3:
                print(f"⚠️ Skipping {obsid}: Data not found or incorrect format.")
                return
            
            flux = data[0]  # First row is flux
            wavelength = data[2]  # Third row is wavelength
            
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot Spectrum
            ax[0].plot(wavelength, flux, color='blue', alpha=0.7, lw=1)
            ax[0].set_xlabel("Wavelength (Å)")
            ax[0].set_ylabel("Flux")
            ax[0].set_title(f"LAMOST Spectrum for Gaia Source ID: {source_id} (LAMOST ObsID: {obsid})")
            ax[0].grid()
            
            # Show Gaia Data in a bar plot
            gaia_info = match.drop(columns=["source_id", "obsid"], errors='ignore')
            # drop columns with flux_ prefix
            gaia_info = gaia_info.loc[:, ~gaia_info.columns.str.startswith("flux_")]
            if not gaia_info.empty:
                ax[1].barh(gaia_info.columns, gaia_info.values[0], color='skyblue')
                ax[1].set_title("Gaia Parameters")
            else:
                ax[1].text(0.5, 0.5, "No Gaia Data Available", ha='center', va='center', fontsize=12)
                ax[1].axis("off")
            plt.show()
    except Exception as e:
        print(f"Error loading {fits_path}: {e}")

gaia_lamost_merged['obsid'] = gaia_lamost_merged['obsid'].astype(int)
gaia_lamost_merged['source_id'] = gaia_lamost_merged['source_id'].astype(int)
#print(gaia_lamost_merged.head())
incorrect_gaia_ids2 = incorrect_gaia_ids.astype(int)

# Loop through incorrectly classified sources and plot spectra
for source_id in incorrect_gaia_ids2:
    plot_spectrum_with_gaia(source_id, gaia_lamost_merged)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.io import fits

def plot_spectrum_with_gaia(source_id, gaia_lamost_merged, spectra_folder="lamost_spectra_uniques"):
    """
    Plots the LAMOST spectrum from FITS files and displays Gaia parameters below it.
    
    :param source_id: Gaia Source ID of the incorrectly classified source
    :param gaia_lamost_merged: DataFrame containing Gaia and LAMOST cross-matched data
    :param spectra_folder: Path to the folder containing LAMOST FITS spectra
    """
    try:
        # Ensure 'obsid' column exists
        if 'obsid' not in gaia_lamost_merged.columns:
            print(f"⚠️ 'obsid' column not found in gaia_lamost_merged.")
            return
        
        match = gaia_lamost_merged.loc[gaia_lamost_merged['source_id'] == source_id]
        if match.empty:
            print(f"⚠️ No LAMOST match found for source_id {source_id}.")
            return
        
        obsid = match.iloc[0]['obsid']
        print(f"Found match: Source ID {source_id} -> ObsID {obsid}")
        
        # Construct FITS file path
        fits_path = f"{spectra_folder}/{int(obsid)}"
        
        # Load FITS data
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            if data is None or data.shape[0] < 3:
                print(f"⚠️ Skipping {obsid}: Data not found or incorrect format.")
                return
            
            flux = data[0]  # First row is flux
            wavelength = data[2]  # Third row is wavelength
            
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot Spectrum
            ax[0].plot(wavelength, flux, color='blue', alpha=0.7, lw=1)
            ax[0].set_xlabel("Wavelength (Å)")
            ax[0].set_ylabel("Flux")
            ax[0].set_title(f"LAMOST Spectrum for Source ID: {source_id} (ObsID: {obsid})")
            ax[0].grid()
            
            gaia_info = match.drop(columns=["source_id", "obsid"], errors='ignore')
            if not gaia_info.empty:
                table_data = gaia_info.iloc[0].to_dict()
                column_labels = list(table_data.keys())
                cell_values = [[table_data[col]] for col in column_labels]
                
                ax[1].axis("tight")
                ax[1].axis("off")
                ax[1].table(cellText=cell_values, colLabels=column_labels, cellLoc='center', loc='center')
            else:
                ax[1].text(0.5, 0.5, "No Gaia Data Available", ha='center', va='center', fontsize=12)
                ax[1].axis("off")
            
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Error loading {fits_path}: {e}")
gaia_lamost_merged['obsid'] = gaia_lamost_merged['obsid'].astype(int)
gaia_lamost_merged['source_id'] = gaia_lamost_merged['source_id'].astype(int)
#print(gaia_lamost_merged.head())
incorrect_gaia_ids2 = incorrect_gaia_ids.astype(int)
bunda

# Loop through incorrectly classified sources and plot spectra
for source_id in incorrect_gaia_ids2:
    plot_spectrum_with_gaia(source_id, gaia_lamost_merged)


def plot_spectrum_with_gaia(source_id, gaia_lamost_merged, spectra_folder="lamost_spectra_uniques"):
    """
    Plots the LAMOST spectrum from FITS files and displays Gaia parameters below it.
    
    :param source_id: Gaia Source ID of the incorrectly classified source
    :param gaia_lamost_merged: DataFrame containing Gaia and LAMOST cross-matched data
    :param spectra_folder: Path to the folder containing LAMOST FITS spectra
    """
    try:
        # Ensure 'obsid' column exists
        if 'obsid' not in gaia_lamost_merged.columns:
            print(f"⚠️ 'obsid' column not found in gaia_lamost_merged.")
            return
        
        match = gaia_lamost_merged.loc[gaia_lamost_merged['source_id'] == source_id]
        if match.empty:
            print(f"⚠️ No LAMOST match found for source_id {source_id}.")
            return
        
        obsid = int(match.iloc[0]['obsid'])  # Ensure obsid is an integer
        print(f"Found match: Source ID {source_id} -> ObsID {obsid}")
        
        # Construct FITS file path
        fits_path = f"{spectra_folder}/{int(obsid)}"
        
        # Load FITS data
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            if data is None or data.shape[0] < 3:
                print(f"⚠️ Skipping {obsid}: Data not found or incorrect format.")
                return
            
            flux = data[0]  # First row is flux
            wavelength = data[2]  # Third row is wavelength
            
            fig, ax = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 3]})
            
            # Plot Spectrum
            ax[0].plot(wavelength, flux, color='blue', alpha=0.7, lw=1)
            ax[0].set_xlabel("Wavelength (Å)")
            ax[0].set_ylabel("Flux")
            ax[0].set_title(f"LAMOST Spectrum for Source ID: {source_id} (ObsID: {obsid})")
            ax[0].grid()
            
            # Show Gaia Data in a Horizontal Bar Plot
            gaia_info = match.iloc[[0]].drop(columns=["source_id", "obsid"], errors='ignore')
            # drop columns with flux_ prefix
            gaia_info = gaia_info.loc[:, ~gaia_info.columns.str.startswith("flux_")]
            if not gaia_info.empty:
                gaia_data = gaia_info.to_dict(orient='records')[0]
                labels = list(gaia_data.keys())
                values = list(gaia_data.values())
                
                ax[1].bar(labels, values, color='skyblue')
                ax[1].tick_params("x", labelrotation=90)
                ax[1].set_xlabel("Value")
                ax[1].set_title("Gaia Parameters")
            else:
                ax[1].text(0.5, 0.5, "No Gaia Data Available", ha='center', va='center', fontsize=12)
                ax[1].axis("off")
            
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Error loading {fits_path}: {e}")

gaia_lamost_merged['obsid'] = gaia_lamost_merged['obsid'].astype(int)
gaia_lamost_merged['source_id'] = gaia_lamost_merged['source_id'].astype(int)
#print(gaia_lamost_merged.head())
incorrect_gaia_ids2 = incorrect_gaia_ids.astype(int)
bunda
# Loop through incorrectly classified sources and plot spectra
for source_id in incorrect_gaia_ids2:
    plot_spectrum_with_gaia(source_id, gaia_lamost_merged)

def process_lamost_fits_files(folder_path="lamost_spectra_uniques", output_file="Pickles/lamost_data.csv", batch_size=10000):
    """
    Processes LAMOST FITS spectra by extracting flux and frequency data.
    Saves data in a CSV file with batching to optimize memory usage.

    Args:
        folder_path (str): Path to the folder containing FITS files.
        output_file (str): Path to the output CSV file.
        batch_size (int): Number of records to process before writing to the CSV.
    """

    print("\n📂 Processing LAMOST FITS files...")

    # Create output folder if necessary and delete existing file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Define column headers
    columns = [f'col_{i}' for i in range(3748)] + ['file_name', 'row']

    # Initialize the CSV file with headers
    with open(output_file, 'w') as f:
        pd.DataFrame(columns=columns).to_csv(f, index=False)
        f.close()

    # Count total files for progress tracking
    total_files = sum([len(files) for _, _, files in os.walk(folder_path)])

    batch_list = []

    # Process FITS files
    with tqdm(total=total_files, desc='Processing FITS files') as pbar:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with fits.open(file_path) as hdul:
                    data = hdul[0].data[:3, :3748]  # Extract first 3 rows and 3748 columns
                    
                    for i, row_data in enumerate(data):
                        data_dict = {f'col_{j}': value for j, value in enumerate(row_data)}
                        data_dict['file_name'] = filename
                        data_dict['row'] = i  # Track which row from the FITS file
                        batch_list.append(data_dict)
                
                # Write batch to CSV
                if len(batch_list) >= batch_size:
                    pd.DataFrame(batch_list).to_csv(output_file, mode='a', header=False, index=False)
                    batch_list.clear()

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

            pbar.update(1)

        # Write any remaining data
        if batch_list:
            pd.DataFrame(batch_list).to_csv(output_file, mode='a', header=False, index=False)


def download_lamost_spectra(obsid_list, save_folder="lamost_spectra_uniques", num_workers=50):
    """
    Downloads LAMOST spectra by obsid in parallel.
    
    :param obsid_list: List of obsids to download
    :param save_folder: Folder where spectra will be saved
    :param num_workers: Number of parallel download threads
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        print(f"📂 Folder '{save_folder}' already exists. Existing files will be deleted")
        #shutil.rmtree(save_folder)
        

    # Create a requests Session with Retry to handle transient errors
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # Use ThreadPoolExecutor to download in parallel
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_obsid = {
            executor.submit(download_one_spectrum, obsid, session, save_folder): obsid 
            for obsid in obsid_list
        }

        # Wrap with tqdm for progress bar
        for future in tqdm(as_completed(future_to_obsid), total=len(future_to_obsid), desc="Downloading Spectra"):
            obsid = future_to_obsid[future]
            try:
                obsid, success, error_msg = future.result()
                results.append((obsid, success, error_msg))
            except Exception as e:
                results.append((obsid, False, str(e)))

    # Print any failures
    failures = [r for r in results if not r[1]]
    if failures:
        print(f"❌ Failed to download {len(failures)} spectra:")
        for (obsid, _, err) in failures[:10]:  # show first 10 errors
            print(f"  obsid={obsid} => Error: {err}")

    # Return list of successfully downloaded obsids for reference
    downloaded_obsids = [r[0] for r in results if r[1]]
    return downloaded_obsids

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def calculate_precision_per_class(true_labels, predicted_labels):
    """
    Calculates the precision for each class.
    """
    precisions = []
    for i in range(true_labels.shape[1]):
        true_positives = np.sum((true_labels[:, i] == 1) & (predicted_labels[:, i] == 1))
        false_positives = np.sum((true_labels[:, i] == 0) & (predicted_labels[:, i] == 1))

        if true_positives + false_positives == 0:
            precision = 0  # Avoid division by zero
        else:
            precision = true_positives / (true_positives + false_positives)

        precisions.append(precision)
    return precisions

def calculate_f1_score_per_class(true_labels, predicted_labels):
    """Calculate the F1 score for each class."""
    f1_scores = []
    for i in range(true_labels.shape[1]):
        f1 = f1_score(true_labels[:, i], predicted_labels[:, i])
        f1_scores.append(f1)
    return f1_scores

def calculate_sample_size_per_class(true_labels):
    """Calculates the sample size for each class."""
    return np.sum(true_labels, axis=0)

def plot_metrics_per_class(true_labels, predicted_labels, class_names, log_scale=False):
    """
    Plots precision and F1 score against sample size for each class.
    """
    sample_sizes = calculate_sample_size_per_class(true_labels)
    precisions = calculate_precision_per_class(true_labels, predicted_labels)
    f1_scores = calculate_f1_score_per_class(true_labels, predicted_labels)

    if log_scale:
        sample_sizes = np.log10(sample_sizes + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Precision plot
    axes[0].scatter(sample_sizes, precisions, color='steelblue', s=100, edgecolors='k', alpha=0.7)
    for i, class_name in enumerate(class_names):
        axes[0].text(sample_sizes[i], precisions[i], class_name, fontsize=9, ha='right')

    axes[0].set_xlabel('Sample Size (Total Number of Samples)')
    axes[0].set_ylabel('Precision (Correct Guesses / Total Predictions)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_xlim(0, np.max(sample_sizes) * 1.05)
    axes[0].set_ylim(-0.0, 1.0)
    axes[0].set_title("Sample Size vs Precision")

    # F1 Score plot
    axes[1].scatter(sample_sizes, f1_scores, color='steelblue', s=100, edgecolors='k', alpha=0.7)
    for i, class_name in enumerate(class_names):
        axes[1].text(sample_sizes[i], f1_scores[i], class_name, fontsize=9, ha='right')

    axes[1].set_xlabel('Sample Size (Total Number of Samples)')
    axes[1].set_ylabel('F1 Score (2 * Precision * Recall / Precision + Recall)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_xlim(0, np.max(sample_sizes) * 1.05)
    axes[1].set_ylim(-0.0, 1.0)
    axes[1].set_title("Sample Size vs F1 Score")

    plt.tight_layout()
    plt.show()

# Load saved predictions
y_cpu = np.load("Results/mamba_fused_v3_y_cpu.npy")
predicted_cpu = np.load("Results/mamba_fused_v3_predicted_cpu.npy")

# Load class names
import pandas as pd
classes = pd.read_pickle("Pickles/Updated_List_of_Classes_ubuntu.pkl")

# Plot the results
plot_metrics_per_class(y_cpu, predicted_cpu, classes, log_scale=False)


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

import torch
import torch.nn as nn
from functools import partial

# Import the needed components from your MambaOut implementation
from timm.models.layers import DropPath

class GatedCNNBlock(nn.Module):
    """Adaptation of GatedCNNBlock for sequence data"""
    def __init__(self, dim, d_state=256, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(expand * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        
        # Use 1D convolution for sequence data with same padding
        # Ensure padding is properly set to maintain sequence length
        padding = (d_conv - 1) // 2  # This ensures 'same' padding for odd kernel sizes
        self.conv = nn.Conv1d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=d_conv,
            padding=padding,
            groups=hidden  # Depthwise convolution
        )
        
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Input shape: [B, seq_len, dim]
        shortcut = x
        x = self.norm(x)
        
        # Split the channels for gating mechanism
        x = self.fc1(x)  # [B, seq_len, hidden*2]
        chunks = torch.chunk(x, 2, dim=-1)  # Creates two tensors
        g, c = chunks  # Each: [B, seq_len, hidden]
        
        # Apply 1D convolution on c (preserving sequence length)
        batch_size, seq_len, channels = c.shape
        c_permuted = c.permute(0, 2, 1)  # [B, hidden, seq_len]
        c_conv = self.conv(c_permuted)  # [B, hidden, seq_len]
        
        # Ensure c_conv has the right sequence length
        if c_conv.shape[2] != seq_len:
            # If sequence length changed (due to even kernel size)
            # Either pad or trim to match original sequence length
            if c_conv.shape[2] < seq_len:
                # Pad if too short
                padding = torch.zeros(batch_size, channels, seq_len - c_conv.shape[2], 
                                      device=c_conv.device)
                c_conv = torch.cat([c_conv, padding], dim=2)
            else:
                # Trim if too long
                c_conv = c_conv[:, :, :seq_len]
        
        c_final = c_conv.permute(0, 2, 1)  # [B, seq_len, hidden]
        
        # Gating mechanism
        x = self.fc2(self.act(g) * c_final)  # [B, seq_len, dim]
        
        x = self.drop_path(x)
        return x + shortcut

class SequenceMambaOut(nn.Module):
    """Adaptation of MambaOut for sequence data with a single stage"""
    def __init__(self, d_model, d_state=256, d_conv=4, expand=2, depth=1, drop_path=0.):
        super().__init__()
        
        # Create a sequence of GatedCNNBlocks
        self.blocks = nn.Sequential(
            *[GatedCNNBlock(
                dim=d_model,
                d_state=d_state,
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

class StarClassifierFusion(nn.Module):
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
            n_layers (int): depth for each MAMBA
            use_cross_attention (bool): whether to use cross-attention
            n_cross_attn_heads (int): number of heads for cross-attention
        """
        super().__init__()

        # --- MambaOut for spectra ---
        self.mamba_spectra = nn.Sequential(
            *[SequenceMambaOut(
                d_model=d_model_spectra,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                depth=1,  # Each SequenceMambaOut has depth 1
                drop_path=0.1 if i > 0 else 0.0,  # Optional: add some dropout for regularization
            ) for i in range(n_layers)]
        )
        self.input_proj_spectra = nn.Linear(input_dim_spectra, d_model_spectra)

        # --- MambaOut for gaia ---
        self.mamba_gaia = nn.Sequential(
            *[SequenceMambaOut(
                d_model=d_model_gaia,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                depth=1,  # Each SequenceMambaOut has depth 1
                drop_path=0.1 if i > 0 else 0.0,  # Optional: add some dropout for regularization
            ) for i in range(n_layers)]
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
    
    def forward(self, x_spectra, x_gaia):
        """
        x_spectra : (batch_size, input_dim_spectra) or (batch_size, seq_len_spectra, input_dim_spectra)
        x_gaia    : (batch_size, input_dim_gaia) or (batch_size, seq_len_gaia, input_dim_gaia)
        """
        # For MambaOut, we expect shape: (B, seq_len, d_model). 
        # If input is just (B, d_in), we turn it into (B, 1, d_in).
        
        # --- Project to d_model and add sequence dimension if needed ---
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

        # --- MambaOut encoding (each modality separately) ---
        x_spectra = self.mamba_spectra(x_spectra)  # (B, seq_len, d_model_spectra)
        x_gaia = self.mamba_gaia(x_gaia)           # (B, seq_len, d_model_gaia)

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
    
    import torch
import torch.nn as nn
from functools import partial

# Import the needed components from your MambaOut implementation
from timm.models.layers import DropPath

class GatedCNNBlock(nn.Module):
    """Adaptation of GatedCNNBlock for sequence data"""
    def __init__(self, dim, d_state=256, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(expand * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        
        # Use 1D convolution for sequence data
        self.conv = nn.Conv1d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=hidden  # Depthwise convolution
        )
        
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Input shape: [B, seq_len, dim]
        shortcut = x
        x = self.norm(x)
        
        x = self.fc1(x)  # [B, seq_len, hidden*2]
        g, c = torch.chunk(x, 2, dim=-1)  # Each: [B, seq_len, hidden]
        
        # Apply 1D convolution on c
        c = c.permute(0, 2, 1)  # [B, hidden, seq_len]
        c = self.conv(c)  # [B, hidden, seq_len]
        c = c.permute(0, 2, 1)  # [B, seq_len, hidden]
        
        # Gating mechanism
        x = self.fc2(self.act(g) * c)  # [B, seq_len, dim]
        
        x = self.drop_path(x)
        return x + shortcut

class SequenceMambaOut(nn.Module):
    """Adaptation of MambaOut for sequence data with a single stage"""
    def __init__(self, d_model, d_state=256, d_conv=4, expand=2, depth=1, drop_path=0.):
        super().__init__()
        
        # Create a sequence of GatedCNNBlocks
        self.blocks = nn.Sequential(
            *[GatedCNNBlock(
                dim=d_model,
                d_state=d_state,
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

class StarClassifierFusion(nn.Module):
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
            n_layers (int): depth for each MAMBA
            use_cross_attention (bool): whether to use cross-attention
            n_cross_attn_heads (int): number of heads for cross-attention
        """
        super().__init__()

        # --- MambaOut for spectra ---
        self.mamba_spectra = nn.Sequential(
            *[SequenceMambaOut(
                d_model=d_model_spectra,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                depth=1,  # Each SequenceMambaOut has depth 1
                drop_path=0.1 if i > 0 else 0.0,  # Optional: add some dropout for regularization
            ) for i in range(n_layers)]
        )
        self.input_proj_spectra = nn.Linear(input_dim_spectra, d_model_spectra)

        # --- MambaOut for gaia ---
        self.mamba_gaia = nn.Sequential(
            *[SequenceMambaOut(
                d_model=d_model_gaia,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                depth=1,  # Each SequenceMambaOut has depth 1
                drop_path=0.1 if i > 0 else 0.0,  # Optional: add some dropout for regularization
            ) for i in range(n_layers)]
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
    
    def forward(self, x_spectra, x_gaia):
        """
        x_spectra : (batch_size, input_dim_spectra) or (batch_size, seq_len_spectra, input_dim_spectra)
        x_gaia    : (batch_size, input_dim_gaia) or (batch_size, seq_len_gaia, input_dim_gaia)
        """
        # For MambaOut, we expect shape: (B, seq_len, d_model). 
        # If input is just (B, d_in), we turn it into (B, 1, d_in).
        
        # --- Project to d_model and add sequence dimension if needed ---
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

        # --- MambaOut encoding (each modality separately) ---
        x_spectra = self.mamba_spectra(x_spectra)  # (B, seq_len, d_model_spectra)
        x_gaia = self.mamba_gaia(x_gaia)           # (B, seq_len, d_model_gaia)

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
    
    def train_model_fusion(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs=100,
    lr=1e-4,
    max_patience=20,
    device='cuda'
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=int(max_patience / 5)
    )

    # We assume the datasets are MultiModalBalancedMultiLabelDataset
    # that returns (X_spectra, X_gaia, y).
    # You can keep the class weighting logic as in train_model_mamba.
    all_labels = []
    for _, _, y_batch in train_loader:
        all_labels.extend(y_batch.cpu().numpy())
    
    class_weights = calculate_class_weights(np.array(all_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    best_val_loss = float('inf')
    patience = max_patience

    for epoch in range(num_epochs):
        # Resample training data
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
        for X_spc, X_ga, y_batch in train_loader:
            X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_spc, X_ga)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_spc.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predicted == y_batch).float()
            train_acc += correct.mean(dim=1).mean().item()

        # --- Validation ---
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for X_spc, X_ga, y_batch in val_loader:
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                outputs = model(X_spc, X_ga)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_spc.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct = (predicted == y_batch).float()
                val_acc += correct.mean(dim=1).mean().item()

        # --- Test metrics (optional or do after training) ---
        test_loss, test_acc = 0.0, 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_spc, X_ga, y_batch in test_loader:
                X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
                outputs = model(X_spc, X_ga)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_spc.size(0)
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct = (predicted == y_batch).float()
                test_acc += correct.mean(dim=1).mean().item()

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Compute multi-label metrics as before
        all_metrics = calculate_metrics(np.array(y_true), np.array(y_pred))

        # Logging example
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader.dataset),
            "val_loss": val_loss / len(val_loader.dataset),
            "train_acc": train_acc / len(train_loader),
            "val_acc": val_acc / len(val_loader),
            "test_loss": test_loss / len(test_loader.dataset),
            "test_acc": test_acc / len(test_loader),
            **all_metrics
        })

        # Scheduler
        scheduler.step(val_loss / len(val_loader.dataset))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = max_patience
            best_model = model.state_dict()
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)
    return model

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
    """Transformer-based feature extractor for multi-modal fusion of spectra and Gaia data."""
    def __init__(
        self,
        d_model_spectra,
        d_model_gaia,
        num_classes,
        input_dim_spectra,
        input_dim_gaia,
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
            n_layers (int): depth for each Transformer
            n_heads (int): number of attention heads
            use_cross_attention (bool): whether to use cross-attention
            n_cross_attn_heads (int): number of heads for cross-attention
            dropout (float): dropout rate
        """
        super().__init__()

        # --- Transformer for spectra ---
        self.transformer_spectra = TransformerFeatureExtractor(
            d_model=d_model_spectra,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        self.input_proj_spectra = nn.Linear(input_dim_spectra, d_model_spectra)

        # --- Transformer for gaia ---
        self.transformer_gaia = TransformerFeatureExtractor(
            d_model=d_model_gaia,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
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
        # Project to embedding space
        if x_spectra.dim() == 2:
            # [batch_size, input_dim] -> [batch_size, d_model]
            x_spectra = self.input_proj_spectra(x_spectra)
            # Add sequence dimension: [batch_size, d_model] -> [batch_size, 1, d_model]
            x_spectra = x_spectra.unsqueeze(1)
        else:
            # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, d_model]
            x_spectra = self.input_proj_spectra(x_spectra)
        
        if x_gaia.dim() == 2:
            # [batch_size, input_dim] -> [batch_size, d_model]
            x_gaia = self.input_proj_gaia(x_gaia)
            # Add sequence dimension: [batch_size, d_model] -> [batch_size, 1, d_model]
            x_gaia = x_gaia.unsqueeze(1)
        else:
            # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, d_model]
            x_gaia = self.input_proj_gaia(x_gaia)

        # Process through transformers
        x_spectra = self.transformer_spectra(x_spectra)  # [batch_size, seq_len, d_model]
        x_gaia = self.transformer_gaia(x_gaia)          # [batch_size, seq_len, d_model]

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
    
class StarClassifierFusionMambaOut(nn.Module):
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
            n_layers (int): depth for each MAMBA
            use_cross_attention (bool): whether to use cross-attention
            n_cross_attn_heads (int): number of heads for cross-attention
        """
        super().__init__()

        # --- MambaOut for spectra ---
        self.mamba_spectra = nn.Sequential(
            *[SequenceMambaOut(
                d_model=d_model_spectra,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                depth=1,  # Each SequenceMambaOut has depth 1
                drop_path=0.1 if i > 0 else 0.0,  # Optional: add some dropout for regularization
            ) for i in range(n_layers)]
        )
        self.input_proj_spectra = nn.Linear(input_dim_spectra, d_model_spectra)

        # --- MambaOut for gaia ---
        self.mamba_gaia = nn.Sequential(
            *[SequenceMambaOut(
                d_model=d_model_gaia,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                depth=1,  # Each SequenceMambaOut has depth 1
                drop_path=0.1 if i > 0 else 0.0,  # Optional: add some dropout for regularization
            ) for i in range(n_layers)]
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
    
    def forward(self, x_spectra, x_gaia):
        """
        x_spectra : (batch_size, input_dim_spectra) or (batch_size, seq_len_spectra, input_dim_spectra)
        x_gaia    : (batch_size, input_dim_gaia) or (batch_size, seq_len_gaia, input_dim_gaia)
        """
        # For MambaOut, we expect shape: (B, seq_len, d_model). 
        # If input is just (B, d_in), we turn it into (B, 1, d_in).
        
        # --- Project to d_model and add sequence dimension if needed ---
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

        # --- MambaOut encoding (each modality separately) ---
        x_spectra = self.mamba_spectra(x_spectra)  # (B, seq_len, d_model_spectra)
        x_gaia = self.mamba_gaia(x_gaia)           # (B, seq_len, d_model_gaia)

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
                d_state=d_state,
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
                d_state=d_state,
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
            model_type = 'MAMBA'
        elif 'transformer' in model_path.lower():
            model_type = 'Transformer'
        elif 'gated_cnn' in model_path.lower() or 'mambaout' in model_path.lower():
            model_type = 'MambaOut'
        else:
            print(f"Unable to determine model type from filename: {model_path}")
            return None
        
        # Initialize parameters based on the model configurations in the original code
        # Define model architecture configurations based on token config and model type
        
        # Default parameters that will be overridden based on token config
        n_layers = 20 if model_type in ['MAMBA', 'MambaOut'] else 10  # Different default layers by model type
        n_heads = 8
        d_state = 32 if model_type == 'MAMBA' else None  # Only for MAMBA models
        d_conv = 4
        expand = 2
        
        # Determine token config from filename
        token_config = None
        if '1_token' in model_path.lower():
            token_config = '1 Token'
            d_model_spectra = 2048
            d_model_gaia = 2048
            token_dim_spectra = input_dim_spectra  # All features in one token (3647)
            token_dim_gaia = input_dim_gaia        # All features in one token (18)
            if model_type == 'MAMBA':
                d_conv = 2  # Special case for MAMBA 1 Token configuration
                
        elif 'balanced' in model_path.lower():
            token_config = 'Balanced'
            d_model_spectra = 2048
            d_model_gaia = 2048
            token_dim_spectra = 192  # Will create ~19 tokens for spectra (3647/192)
            token_dim_gaia = 1       # Will create 18 tokens for gaia (18/1)
            
        else:  # max_tokens
            token_config = 'Max Tokens'
            d_model_spectra = 1536
            d_model_gaia = 1536
            token_dim_spectra = 7    # Will create 522 tokens for spectra (3647/7)
            token_dim_gaia = 1       # Will create 18 tokens for gaia (18/1)
            if model_type == 'MAMBA':
                d_state = 16  # Reduced state dimension for the Max Tokens MAMBA configuration
        
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
    


    import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, roc_auc_score, average_precision_score
import os
import json
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score


# Import your model architectures
from Fusion_Models import StarClassifierFusionMambaOut, StarClassifierFusionTransformer, StarClassifierFusionMambaTokenized

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



def load_data():
    """Load and preprocess the data"""
    print("Loading datasets...")
    
    # Load classes
    with open("Pickles/Updated_List_of_Classes_ubuntu.pkl", "rb") as f:
        classes = pickle.load(f)
    
    # Load test data
    with open("Pickles/test_data_transformed_ubuntu.pkl", "rb") as f:
        X_test_full = pickle.load(f)
    
    # Extract labels
    y_test = X_test_full[classes]
    
    # Drop labels from both datasets
    X_test_full.drop(classes, axis=1, inplace=True)
    
    # Define Gaia columns
    gaia_columns = ["parallax", "ra", "dec", "ra_error", "dec_error", "parallax_error", 
                   "pmra", "pmdec", "pmra_error", "pmdec_error", "phot_g_mean_flux", 
                   "flagnopllx", "phot_g_mean_flux_error", "phot_bp_mean_flux", 
                   "phot_rp_mean_flux", "phot_bp_mean_flux_error", 
                   "phot_rp_mean_flux_error", "flagnoflux"]
    
    # Split data into spectra and gaia parts
    X_test_spectra = X_test_full.drop(columns={"otype", "obsid", *gaia_columns})
    
    X_test_gaia = X_test_full[gaia_columns]
    
    # Free up memory
    del X_test_full
    gc.collect()
    
    # Convert to PyTorch tensors
    X_test_spectra_tensor = torch.tensor(X_test_spectra.values, dtype=torch.float32)

    X_test_gaia_tensor = torch.tensor(X_test_gaia.values, dtype=torch.float32)
    
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    return (X_test_spectra_tensor, X_test_gaia_tensor, y_test_tensor)

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate a model on test data and return comprehensive metrics"""
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    y_true, y_pred, y_prob = [], [], []
    
    # Compute class weights for loss function
    all_labels = []
    for _, _, y_batch in test_loader:
        all_labels.extend(y_batch.cpu().numpy())
    
    class_weights = calculate_class_weights(np.array(all_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Evaluation loop
    with torch.no_grad():
        for X_spc, X_ga, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_spc, X_ga, y_batch = X_spc.to(device), X_ga.to(device), y_batch.to(device)
            outputs = model(X_spc, X_ga)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_spc.size(0)
            
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct = (predicted == y_batch).float()
            test_acc += correct.mean(dim=1).mean().item()

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    y_prob_array = np.array(y_prob)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_array, y_pred_array)
    
    # Add average metrics
    metrics["avg_loss"] = test_loss / len(test_loader.dataset)
    metrics["avg_accuracy"] = test_acc / len(test_loader)
    
    # Calculate AUROC if possible
    try:
        class_aurocs = []
        for i in range(y_true_array.shape[1]):
            if len(np.unique(y_true_array[:, i])) > 1:
                class_auroc = roc_auc_score(y_true_array[:, i], y_prob_array[:, i])
                class_aurocs.append(class_auroc)
        
        if class_aurocs:
            metrics["macro_auroc"] = np.mean(class_aurocs)
    except Exception as e:
        print(f"Error calculating AUROC: {e}")
        metrics["macro_auroc"] = float('nan')
    
    return metrics

def main():
    # Create results directory
    results_dir = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("Models", exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    (X_test_spectra, X_test_gaia, y_test) = load_data()
    
    # Create datasets and dataloaders
    batch_size = 16
    batch_limit = int(batch_size / 2.5)
    

    test_dataset = MultiModalBalancedMultiLabelDataset(
        X_test_spectra, X_test_gaia, y_test, limit_per_label=batch_limit
    )
    

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    

    print(f"Test samples: {len(test_dataset)}")
    # Define model configurations to evaluate
    model_configs = [
        # MambaOut Models
        {
            "name": "MambaOut_1token",
            "model_class": StarClassifierFusionMambaOut,
            "params": {
                "d_model_spectra": 2048,
                "d_model_gaia": 2048,
                "num_classes": 55,
                "input_dim_spectra": 3647,
                "input_dim_gaia": 18,
                "token_dim_spectra": 3647,  # 1 token
                "token_dim_gaia": 18,       # 1 token
                "n_layers": 20,
                "d_conv": 1,
                "use_cross_attention": True,
                "n_cross_attn_heads": 8
            },
            "checkpoint": "Comparing_Mambas_Trans/gated_cnn_(mambaout)_1_token.pth"
        },
        {
            "name": "MambaOut_19_18token",
            "model_class": StarClassifierFusionMambaOut,
            "params": {
                "d_model_spectra": 2048,
                "d_model_gaia": 2048,
                "num_classes": 55,
                "input_dim_spectra": 3647,
                "input_dim_gaia": 18,
                "token_dim_spectra": 192,  # ~19 tokens
                "token_dim_gaia": 1,       # 18 tokens
                "n_layers": 20,
                "d_conv": 4,
                "use_cross_attention": True,
                "n_cross_attn_heads": 8
            },
            "checkpoint": "Comparing_Mambas_Trans/gated_cnn_(mambaout)_balanced.pth"
        },
        {
            "name": "MambaOut_522_18token",
            "model_class": StarClassifierFusionMambaOut,
            "params": {
                "d_model_spectra": 1536,
                "d_model_gaia": 1536,
                "num_classes": 55,
                "input_dim_spectra": 3647,
                "input_dim_gaia": 18,
                "token_dim_spectra": 7,    # ~522 tokens
                "token_dim_gaia": 1,       # 18 tokens
                "n_layers": 20,
                "d_conv": 32,
                "use_cross_attention": True,
                "n_cross_attn_heads": 8
            },
            "checkpoint": "Comparing_Mambas_Trans/gated_cnn_(mambaout)_max_tokens.pth"
        },
        
        # Transformer Models
        {
            "name": "Transformer_1token",
            "model_class": StarClassifierFusionTransformer,
            "params": {
                "d_model_spectra": 2048,
                "d_model_gaia": 2048,
                "num_classes": 55,
                "input_dim_spectra": 3647,
                "input_dim_gaia": 18,
                "token_dim_spectra": 3647,  # 1 token
                "token_dim_gaia": 18,       # 1 token
                "n_layers": 10,
                "n_heads": 8,
                "use_cross_attention": True,
                "n_cross_attn_heads": 8,
                "dropout": 0.1
            },
            "checkpoint": "Comparing_Mambas_Trans/transformer_1_token.pth"
        },
        {
            "name": "Transformer_19_18token",
            "model_class": StarClassifierFusionTransformer,
            "params": {
                "d_model_spectra": 2048,
                "d_model_gaia": 2048,
                "num_classes": 55,
                "input_dim_spectra": 3647,
                "input_dim_gaia": 18,
                "token_dim_spectra": 192,  # ~19 tokens
                "token_dim_gaia": 1,       # 18 tokens
                "n_layers": 10,
                "n_heads": 8,
                "use_cross_attention": True,
                "n_cross_attn_heads": 8,
                "dropout": 0.1
            },
            "checkpoint": "Comparing_Mambas_Trans/transformer_balanced.pth"
        },
        {
            "name": "Transformer_522_18token",
            "model_class": StarClassifierFusionTransformer,
            "params": {
                "d_model_spectra": 1536,
                "d_model_gaia": 1536,
                "num_classes": 55,
                "input_dim_spectra": 3647,
                "input_dim_gaia": 18,
                "token_dim_spectra": 7,    # ~522 tokens
                "token_dim_gaia": 1,       # 18 tokens
                "n_layers": 10,
                "n_heads": 8,
                "use_cross_attention": True,
                "n_cross_attn_heads": 8,
                "dropout": 0.1
            },
            "checkpoint": "Comparing_Mambas_Trans/transformer_max_tokens.pth"
        },
        
        # Mamba2 Tokenized Models
        {
            "name": "Mamba2_1token",
            "model_class": StarClassifierFusionMambaTokenized,
            "params": {
                "d_model_spectra": 2048,
                "d_model_gaia": 2048,
                "num_classes": 55,
                "input_dim_spectra": 3647,
                "input_dim_gaia": 18,
                "token_dim_spectra": 3647,  # 1 token
                "token_dim_gaia": 18,       # 1 token
                "n_layers": 20,
                "d_state": 32,
                "d_conv": 2,
                "expand": 2,
                "use_cross_attention": True,
                "n_cross_attn_heads": 8
            },
            "checkpoint": "Comparing_Mambas_Trans/mamba_1_token.pth"
        },
        {
            "name": "Mamba2_19_18token",
            "model_class": StarClassifierFusionMambaTokenized,
            "params": {
                "d_model_spectra": 2048,
                "d_model_gaia": 2048,
                "num_classes": 55,
                "input_dim_spectra": 3647,
                "input_dim_gaia": 18,
                "token_dim_spectra": 192,  # ~19 tokens
                "token_dim_gaia": 1,       # 18 tokens
                "n_layers": 20,
                "d_state": 32,
                "d_conv": 4,
                "expand": 2,
                "use_cross_attention": True,
                "n_cross_attn_heads": 8
            },
            "checkpoint": "Comparing_Mambas_Trans/mamba_balanced.pth"
        },
        {
            "name": "Mamba2_522_18token",
            "model_class": StarClassifierFusionMambaTokenized,
            "params": {
                "d_model_spectra": 1536,
                "d_model_gaia": 1536,
                "num_classes": 55,
                "input_dim_spectra": 3647,
                "input_dim_gaia": 18,
                "token_dim_spectra": 7,    # ~522 tokens
                "token_dim_gaia": 1,       # 18 tokens
                "n_layers": 20,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "use_cross_attention": True,
                "n_cross_attn_heads": 8
            },
            "checkpoint": "Comparing_Mambas_Trans/mamba_max_tokens.pth"
        }
    ]
    
    # Store results
    results = {}
    
    # Evaluate each model
    for config in model_configs:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {config['name']}")
        print(f"{'='*50}")
        
        # Create model instance
        model = config["model_class"](**config["params"])
        
        # Load checkpoint if exists
        checkpoint_path = config["checkpoint"]
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Checkpoint {checkpoint_path} not found. Skipping this model.")
            continue
        
        # Move model to device
        model = model.to(device)
        
        # Print model statistics
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params:,}")
        
        # Calculate model size in MB
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024**2)
        print(f"Model size: {size_mb:.2f} MB")
        
        # Evaluate model
        metrics = evaluate_model(model, test_loader, device)
        
        # Add model info to metrics
        metrics["model_name"] = config["name"]
        metrics["num_parameters"] = num_params
        metrics["model_size_mb"] = size_mb
        
        # Get token counts for analysis
        spectra_tokens = (config["params"]["input_dim_spectra"] + config["params"]["token_dim_spectra"] - 1) // config["params"]["token_dim_spectra"]
        gaia_tokens = (config["params"]["input_dim_gaia"] + config["params"]["token_dim_gaia"] - 1) // config["params"]["token_dim_gaia"]
        metrics["spectra_tokens"] = spectra_tokens
        metrics["gaia_tokens"] = gaia_tokens
        metrics["total_tokens"] = spectra_tokens + gaia_tokens
        
        # Print key metrics
        print("\nTest Metrics:")
        print(f"  Loss: {metrics['avg_loss']:.4f}")
        print(f"  Accuracy: {metrics['avg_accuracy']:.4f}")
        print(f"  Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"  Macro AUROC: {metrics.get('macro_auroc', 'N/A')}")
        
        # Store results
        results[config["name"]] = metrics
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save results to JSON
    results_file = os.path.join(results_dir, "model_comparison_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Save DataFrame to CSV
    csv_file = os.path.join(results_dir, "model_comparison_results.csv")
    results_df.to_csv(csv_file)
    
    # Create model family summary
    model_families = {
        "MambaOut": [m for m in results_df.index if m.startswith("MambaOut")],
        "Transformer": [m for m in results_df.index if m.startswith("Transformer")],
        "Mamba2": [m for m in results_df.index if m.startswith("Mamba2")]
    }

    # Print debug information to help identify the issue
    print("Model families:")
    for family, models in model_families.items():
        print(f"  {family}: {models}")

    family_results = {}
    for family, models in model_families.items():
        if models:
            # Check if the models list contains actual model names
            if all(isinstance(m, str) and m in results_df.index for m in models):
                # Use only numeric columns for the mean calculation
                numeric_cols = results_df.select_dtypes(include=['number']).columns
                family_results[family] = results_df.loc[models, numeric_cols].mean()
            else:
                print(f"Warning: Invalid model list for {family}: {models}")
                # Skip this family or provide default values

    family_df = pd.DataFrame.from_dict(family_results, orient='index')
    family_csv = os.path.join(results_dir, "model_family_summary.csv")
    family_df.to_csv(family_csv)
    
    # Create comparative visualizations
    
    # 1. Performance by model metrics bar chart
    key_metrics = ['micro_f1', 'macro_f1', 'weighted_f1', 'macro_auroc']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(key_metrics):
        if metric in results_df.columns:
            plt.subplot(2, 2, i+1)
            sns.barplot(x=results_df.index, y=results_df[metric])
            plt.title(f"{metric.replace('_', ' ').title()}")
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "key_metrics_comparison.png"))
    
    # 2. Performance vs Model Size scatter plot
    plt.figure(figsize=(12, 8))
    
    for family, models in model_families.items():
        if models:
            sns.scatterplot(
                x=results_df.loc[models, 'model_size_mb'], 
                y=results_df.loc[models, 'macro_f1'],
                label=family,
                s=100
            )
    
    for i, model in enumerate(results_df.index):
        plt.annotate(
            model,
            (results_df.loc[model, 'model_size_mb'], results_df.loc[model, 'macro_f1']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Macro F1 Score")
    plt.title("Model Performance vs Model Size")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(results_dir, "performance_vs_size.png"))
    
    # 3. Performance vs Number of Tokens scatter plot
    plt.figure(figsize=(12, 8))
    
    for family, models in model_families.items():
        if models:
            sns.scatterplot(
                x=results_df.loc[models, 'total_tokens'], 
                y=results_df.loc[models, 'macro_f1'],
                label=family,
                s=100
            )
    
    for i, model in enumerate(results_df.index):
        plt.annotate(
            model,
            (results_df.loc[model, 'total_tokens'], results_df.loc[model, 'macro_f1']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.xscale('log')
    plt.xlabel("Total Number of Tokens (log scale)")
    plt.ylabel("Macro F1 Score")
    plt.title("Model Performance vs Number of Tokens")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(results_dir, "performance_vs_tokens.png"))
    
    print(f"\nResults saved to {results_dir}/")
    print(f"Summary: {family_csv}")
    
    # Generate a text summary report
    with open(os.path.join(results_dir, "summary_report.txt"), 'w') as f:
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("======================\n\n")
        
        f.write("Best Models by Metric:\n")
        for metric in ['micro_f1', 'macro_f1', 'weighted_f1', 'macro_auroc']:
            if metric in results_df.columns:
                best_model = results_df[metric].idxmax()
                f.write(f"  Best {metric}: {best_model} ({results_df.loc[best_model, metric]:.4f})\n")
        
        f.write("\nModel Family Comparison:\n")
        for family, metrics in family_results.items():
            f.write(f"  {family}:\n")
            f.write(f"    Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"    Average Size: {metrics['model_size_mb']:.2f} MB\n")
        
        f.write("\nToken Configuration Analysis:\n")
        token_configs = ["1token", "19_18token", "522_18token"]
        for config in token_configs:
            models = [m for m in results_df.index if config in m]
            if models:
                config_df = results_df.loc[models]
                f.write(f"  {config}:\n")
                f.write(f"    Average Macro F1: {config_df['macro_f1'].mean():.4f}\n")
                f.write(f"    Best Model: {config_df['macro_f1'].idxmax()} ({config_df['macro_f1'].max():.4f})\n")
        
        f.write("\nDetailed Model Rankings:\n")
        for rank, (model, metrics) in enumerate(results_df.sort_values('macro_f1', ascending=False).iterrows(), 1):
            f.write(f"  {rank}. {model}:\n")
            f.write(f"     Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"     Size: {metrics['model_size_mb']:.2f} MB\n")
            f.write(f"     Parameters: {metrics['num_parameters']:,}\n")
            f.write(f"     Token Config: {metrics['spectra_tokens']} spectra, {metrics['gaia_tokens']} gaia\n")
    
    print(f"Summary report generated: {results_dir}/summary_report.txt")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.colors as mcolors
from graphviz import Digraph
import os

# Set the output directory for saving diagrams
os.makedirs('thesis_figures', exist_ok=True)

def visualize_torchviz(model):
    """
    Visualize model using torchviz (requires torchviz package).
    Note: This shows computational graph, not architecture layout.
    """
    try:
        from torchviz import make_dot
        
        # Create dummy inputs matching your model's expected input shape
        dummy_spectra = torch.randn(1, 3647)
        dummy_gaia = torch.randn(1, 18)
        
        # Forward pass to get the graph
        y = model(dummy_spectra, dummy_gaia)
        
        # Create and save dot graph
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.format = 'pdf'
        dot.render('thesis_figures/model_computational_graph')
        print("Computational graph saved as 'thesis_figures/model_computational_graph.pdf'")
    except ImportError:
        print("torchviz not installed. Install with: pip install torchviz")

def visualize_graphviz(model):
    """
    Create a custom visualization using Graphviz
    """
    dot = Digraph(comment='StarClassifierFusion Architecture', 
                 format='pdf', 
                 node_attr={'shape': 'box', 'style': 'filled', 'fontname': 'Arial'})
    
    # Add title
    dot.attr(label='StarClassifierFusion Neural Network Architecture', labelloc='t', fontsize='20')
    
    # Create subgraphs for each section
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs', style='filled', color='lightblue', fontcolor='black')
        c.node('spectra_input', 'Spectral Input\n(B, 3647)', fillcolor='white')
        c.node('gaia_input', 'Gaia Input\n(B, 18)', fillcolor='white')
    
    # Projection layers
    dot.node('spectra_proj', 'Linear Projection\n(B, 2048)', fillcolor='#F8CECC')
    dot.node('gaia_proj', 'Linear Projection\n(B, 2048)', fillcolor='#F8CECC')
    
    # Connections from inputs to projections
    dot.edge('spectra_input', 'spectra_proj')
    dot.edge('gaia_input', 'gaia_proj')
    
    # Reshape (add sequence dimension)
    dot.node('spectra_reshape', 'Unsqueeze\n(B, 1, 2048)', fillcolor='#D5E8D4')
    dot.node('gaia_reshape', 'Unsqueeze\n(B, 1, 2048)', fillcolor='#D5E8D4')
    
    # Connections to reshape
    dot.edge('spectra_proj', 'spectra_reshape')
    dot.edge('gaia_proj', 'gaia_reshape')
    
    # Mamba layers
    with dot.subgraph(name='cluster_mamba_spectra') as c:
        c.attr(label='Mamba2 Encoder (Spectra)', style='filled', color='#DAE8FC', fontcolor='black')
        c.node('mamba_spectra', f'Mamba2 Layers × {model.mamba_spectra.__len__()}\n(B, 1, 2048)', fillcolor='white')
    
    with dot.subgraph(name='cluster_mamba_gaia') as c:
        c.attr(label='Mamba2 Encoder (Gaia)', style='filled', color='#DAE8FC', fontcolor='black')
        c.node('mamba_gaia', f'Mamba2 Layers × {model.mamba_gaia.__len__()}\n(B, 1, 2048)', fillcolor='white')
    
    # Connections to Mamba
    dot.edge('spectra_reshape', 'mamba_spectra')
    dot.edge('gaia_reshape', 'mamba_gaia')
    
    # Cross-attention blocks (if used)
    if model.use_cross_attention:
        with dot.subgraph(name='cluster_cross_attn') as c:
            c.attr(label='Cross-Attention', style='filled', color='#FFE6CC', fontcolor='black')
            c.node('cross_attn_spectra', 'Cross-Attention\n(Spectra → Gaia)', fillcolor='white')
            c.node('cross_attn_gaia', 'Cross-Attention\n(Gaia → Spectra)', fillcolor='white')
        
        # Connections to cross-attention
        dot.edge('mamba_spectra', 'cross_attn_spectra')
        dot.edge('mamba_gaia', 'cross_attn_gaia')
        dot.edge('mamba_gaia', 'cross_attn_spectra', style='dashed', color='gray')
        dot.edge('mamba_spectra', 'cross_attn_gaia', style='dashed', color='gray')
        
        # Mean pooling after cross-attention
        dot.node('pool_spectra', 'Mean Pooling\n(B, 2048)', fillcolor='#D5E8D4')
        dot.node('pool_gaia', 'Mean Pooling\n(B, 2048)', fillcolor='#D5E8D4')
        
        # Connections to pooling
        dot.edge('cross_attn_spectra', 'pool_spectra')
        dot.edge('cross_attn_gaia', 'pool_gaia')
    else:
        # Direct pooling without cross-attention
        dot.node('pool_spectra', 'Mean Pooling\n(B, 2048)', fillcolor='#D5E8D4')
        dot.node('pool_gaia', 'Mean Pooling\n(B, 2048)', fillcolor='#D5E8D4')
        
        # Connections to pooling
        dot.edge('mamba_spectra', 'pool_spectra')
        dot.edge('mamba_gaia', 'pool_gaia')
    
    # Fusion
    dot.node('fusion', 'Concatenation\n(B, 4096)', fillcolor='#FFF2CC')
    
    # Connections to fusion
    dot.edge('pool_spectra', 'fusion')
    dot.edge('pool_gaia', 'fusion')
    
    # Final classification
    dot.node('layer_norm', 'Layer Normalization', fillcolor='#F8CECC')
    dot.node('classifier', 'Linear Classifier\n(B, 55)', fillcolor='#F8CECC')
    dot.node('output', 'Output Logits', fillcolor='#E1D5E7')
    
    # Final connections
    dot.edge('fusion', 'layer_norm')
    dot.edge('layer_norm', 'classifier')
    dot.edge('classifier', 'output')
    
    # Save the diagram
    dot.render('thesis_figures/model_architecture')
    print("Architecture diagram saved as 'thesis_figures/model_architecture.pdf'")

def plot_network_matplotlib(model):
    """
    Create a custom visualization using matplotlib
    This gives more control over the appearance
    """
    # Set up the figure with a light gray background
    fig, ax = plt.subplots(figsize=(12, 16), facecolor='#F8F9F9')
    ax.set_facecolor('#F8F9F9')
    
    # Remove axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#D4F1F9',
        'projection': '#F8FAD7',
        'mamba': '#E8F8F5',
        'cross_attn': '#F5EEF8',
        'pooling': '#D5E8D4',
        'fusion': '#FADBD8',
        'output': '#FCF3CF',
        'arrow': '#05386B'
    }
    
    # Define box dimensions
    box_width = 2.5
    box_height = 0.6
    
    # Function to add a box with a label
    def add_box(x, y, label, color, alpha=1.0):
        rect = Rectangle((x, y), box_width, box_height, 
                         facecolor=color, edgecolor='#05386B', 
                         alpha=alpha, linewidth=1.5, zorder=1)
        ax.add_patch(rect)
        ax.text(x + box_width/2, y + box_height/2, label, 
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, fontweight='bold', zorder=2)
        return (x + box_width/2, y + box_height/2)  # return center point
    
    # Function to add an arrow between boxes
    def add_arrow(start, end, color='#05386B', linestyle='-', linewidth=1.5):
        arrow = FancyArrowPatch(start, end, 
                                arrowstyle='->', color=color, 
                                linewidth=linewidth, linestyle=linestyle,
                                connectionstyle='arc3,rad=0.1', zorder=0)
        ax.add_patch(arrow)
    
    # Title
    plt.title('StarClassifierFusion: Multimodal Architecture with Mamba2 and Cross-Attention', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add subtitle/description
    ax.text(5, 15.5, 'A multimodal deep learning model for stellar classification using spectral and Gaia data',
           horizontalalignment='center', fontsize=12, fontstyle='italic')
    
    # Input layers
    input_spectra = add_box(1.5, 14, 'Spectral Input\n(B, 3647)', colors['input'])
    input_gaia = add_box(6, 14, 'Gaia Input\n(B, 18)', colors['input'])
    
    # Projection layers
    proj_spectra = add_box(1.5, 13, 'Linear Projection\n(B, 2048)', colors['projection'])
    proj_gaia = add_box(6, 13, 'Linear Projection\n(B, 2048)', colors['projection'])
    
    # Reshape layers
    reshape_spectra = add_box(1.5, 12, 'Unsqueeze\n(B, 1, 2048)', colors['projection'])
    reshape_gaia = add_box(6, 12, 'Unsqueeze\n(B, 1, 2048)', colors['projection'])
    
    # Mamba blocks
    # Create a rectangle area for the Mamba subgraph
    spectra_background = Rectangle((1, 9), 3.5, 2.5, facecolor='#D4E6F1', alpha=0.3, edgecolor='#2874A6', linewidth=1)
    gaia_background = Rectangle((5.5, 9), 3.5, 2.5, facecolor='#D4E6F1', alpha=0.3, edgecolor='#2874A6', linewidth=1)
    ax.add_patch(spectra_background)
    ax.add_patch(gaia_background)
    
    # Label for the mamba sections
    ax.text(2.75, 11.3, f'Mamba2 Encoder (Spectra)', 
            horizontalalignment='center', fontsize=10, fontstyle='italic')
    ax.text(7.25, 11.3, f'Mamba2 Encoder (Gaia)', 
            horizontalalignment='center', fontsize=10, fontstyle='italic')
    
    # Individual Mamba layers
    mamba_spectra1 = add_box(1.5, 10.8, f'Mamba2 Layer 1', colors['mamba'])
    mamba_spectra2 = add_box(1.5, 10.2, f'Mamba2 Layer 2', colors['mamba'])
    mamba_spectra_dots = add_box(1.5, 9.6, f'...', 'none')
    mamba_spectraN = add_box(1.5, 9, f'Mamba2 Layer {model.mamba_spectra.__len__()}', colors['mamba'])
    
    mamba_gaia1 = add_box(6, 10.8, f'Mamba2 Layer 1', colors['mamba'])
    mamba_gaia2 = add_box(6, 10.2, f'Mamba2 Layer 2', colors['mamba'])
    mamba_gaia_dots = add_box(6, 9.6, f'...', 'none')
    mamba_gaiaN = add_box(6, 9, f'Mamba2 Layer {model.mamba_gaia.__len__()}', colors['mamba'])
    
    # Cross-attention (if used) or direct pooling
    if model.use_cross_attention:
        # Create a rectangle area for the cross-attention
        cross_attn_background = Rectangle((1, 7), 8, 1.5, facecolor='#E8DAEF', alpha=0.3, edgecolor='#8E44AD', linewidth=1)
        ax.add_patch(cross_attn_background)
        ax.text(5, 8.3, 'Cross-Attention Fusion', horizontalalignment='center', fontsize=10, fontstyle='italic')
        
        cross_attn_spectra = add_box(1.5, 7.5, 'Cross-Attention\n(Spectra → Gaia)', colors['cross_attn'])
        cross_attn_gaia = add_box(6, 7.5, 'Cross-Attention\n(Gaia → Spectra)', colors['cross_attn'])
        
        # Add dashed lines to show cross-connections
        add_arrow((2.75, 9), (7.75, 7.5), linestyle='--', linewidth=1)
        add_arrow((7.25, 9), (2.75, 7.5), linestyle='--', linewidth=1)
        
        # Pooling layers
        pool_spectra = add_box(1.5, 6.5, 'Mean Pooling\n(B, 2048)', colors['pooling'])
        pool_gaia = add_box(6, 6.5, 'Mean Pooling\n(B, 2048)', colors['pooling'])
        
        # Connections
        add_arrow(input_spectra, proj_spectra)
        add_arrow(input_gaia, proj_gaia)
        add_arrow(proj_spectra, reshape_spectra)
        add_arrow(proj_gaia, reshape_gaia)
        add_arrow(reshape_spectra, mamba_spectra1)
        add_arrow(reshape_gaia, mamba_gaia1)
        add_arrow(mamba_spectra1, mamba_spectra2)
        add_arrow(mamba_gaia1, mamba_gaia2)
        add_arrow(mamba_spectra2, mamba_spectra_dots)
        add_arrow(mamba_gaia2, mamba_gaia_dots)
        add_arrow(mamba_spectra_dots, mamba_spectraN)
        add_arrow(mamba_gaia_dots, mamba_gaiaN)
        add_arrow(mamba_spectraN, cross_attn_spectra)
        add_arrow(mamba_gaiaN, cross_attn_gaia)
        add_arrow(cross_attn_spectra, pool_spectra)
        add_arrow(cross_attn_gaia, pool_gaia)
    else:
        # Direct pooling without cross-attention
        pool_spectra = add_box(1.5, 8, 'Mean Pooling\n(B, 2048)', colors['pooling'])
        pool_gaia = add_box(6, 8, 'Mean Pooling\n(B, 2048)', colors['pooling'])
        
        # Connections
        add_arrow(input_spectra, proj_spectra)
        add_arrow(input_gaia, proj_gaia)
        add_arrow(proj_spectra, reshape_spectra)
        add_arrow(proj_gaia, reshape_gaia)
        add_arrow(reshape_spectra, mamba_spectra1)
        add_arrow(reshape_gaia, mamba_gaia1)
        add_arrow(mamba_spectra1, mamba_spectra2)
        add_arrow(mamba_gaia1, mamba_gaia2)
        add_arrow(mamba_spectra2, mamba_spectra_dots)
        add_arrow(mamba_gaia2, mamba_gaia_dots)
        add_arrow(mamba_spectra_dots, mamba_spectraN)
        add_arrow(mamba_gaia_dots, mamba_gaiaN)
        add_arrow(mamba_spectraN, pool_spectra)
        add_arrow(mamba_gaiaN, pool_gaia)
    
    # Fusion and classification
    fusion = add_box(3.8, 5.5, 'Concatenation\n(B, 4096)', colors['fusion'])
    norm = add_box(3.8, 4.5, 'Layer Normalization', colors['projection'])
    classifier = add_box(3.8, 3.5, 'Linear Classifier\n(B, 55)', colors['projection'])
    output = add_box(3.8, 2.5, 'Output Logits', colors['output'])
    
    # Final connections
    add_arrow(pool_spectra, fusion)
    add_arrow(pool_gaia, fusion)
    add_arrow(fusion, norm)
    add_arrow(norm, classifier)
    add_arrow(classifier, output)
    
    # Add a legend for network components
    legend_x = 7.5
    legend_y = 4
    legend_spacing = 0.7
    
    # Legend title
    ax.text(legend_x, legend_y + 1.2, 'Model Components:', 
            fontsize=10, fontweight='bold')
    
    # Create legend entries
    legend_items = [
        ('Input Layer', colors['input']),
        ('Projection Layer', colors['projection']),
        ('Mamba2 Block', colors['mamba']),
        ('Cross-Attention', colors['cross_attn']),
        ('Pooling Layer', colors['pooling']),
        ('Fusion Layer', colors['fusion']),
        ('Output Layer', colors['output'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y - i * legend_spacing
        # Add colored square
        square = Rectangle((legend_x - 0.4, y_pos - 0.1), 0.3, 0.3, 
                          facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(square)
        # Add label
        ax.text(legend_x, y_pos, label, fontsize=9, 
                verticalalignment='center')
    
    # Add model parameters information
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ax.text(5, 1.5, f'Total Parameters: {param_count:,}', 
            horizontalalignment='center', fontsize=10, fontweight='bold')
    
    # Add university logo/attribution placeholder
    ax.text(5, 0.5, 'Your University / Institution Name', 
            horizontalalignment='center', fontsize=10, fontweight='bold')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('thesis_figures/model_architecture_matplotlib.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('thesis_figures/model_architecture_matplotlib.png', dpi=300, bbox_inches='tight')
    print("Matplotlib architecture visualization saved as 'thesis_figures/model_architecture_matplotlib.pdf/png'")

def generate_model_description():
    """
    Generate a professional description of the model architecture for the thesis.
    """
    description = """
## 3.5.1 Model Architecture Overview: StarClassifierFusion

The **StarClassifierFusion** model is a multimodal deep learning architecture designed to leverage both spectral and Gaia astrometric data for stellar classification. The model employs parallel processing branches for different data modalities, with optional cross-modal interaction capabilities, followed by late fusion for final classification.

### Key Components

1. **Dual Modality Encoders**: 
   * **Spectral Branch**: Processes high-dimensional spectral data (3,647 features) through a dedicated encoder.
   * **Gaia Branch**: Processes lower-dimensional astrometric data (18 features) through a separate encoder.

2. **Mamba2 Backbone**:
   * Both branches utilize Mamba2 layers, a state-of-the-art sequence modeling architecture that combines the efficiency of linear recurrent models with the expressivity of attention-based models.
   * Each branch consists of {n_layers} stacked Mamba2 layers with {d_model} hidden dimensions.

3. **Cross-Attention Mechanism**:
   * Facilitates information exchange between the two modality branches.
   * **Spectra-to-Gaia Attention**: The spectral branch attends to relevant features in the Gaia branch.
   * **Gaia-to-Spectra Attention**: The Gaia branch attends to relevant features in the spectral branch.
   * Each cross-attention block employs multi-head attention with {n_heads} heads and residual connections with layer normalization.

4. **Late Fusion**:
   * Concatenation of the processed features from both branches results in a joint representation with {fusion_dim} dimensions.
   * Layer normalization is applied to the concatenated features for stable training.

5. **Classification Head**:
   * A linear layer maps the fused representation to {num_classes} output logits corresponding to different stellar classes.

### Technical Specifications

* **Input Dimensions**:
  * Spectral data: B × {input_dim_spectra} (where B is batch size)
  * Gaia data: B × {input_dim_gaia}
  
* **Embedding Dimensions**:
  * Spectral branch: {d_model_spectra}
  * Gaia branch: {d_model_gaia}
  
* **Mamba2 Configuration**:
  * State dimension: {d_state}
  * Convolution kernel size: {d_conv}
  * Expansion factor: {expand}

* **Model Size**:
  * Total parameters: {param_count:,}
  * Model size: {model_size:.2f} MB

### Design Rationale

This architecture was designed to effectively capture both the fine-grained spectral features and the complementary astrometric information from Gaia, while enabling cross-modal interaction through the attention mechanism. The Mamba2 backbone was selected for its efficiency in processing sequence data compared to Transformer models, while maintaining competitive performance.

The cross-attention fusion strategy allows the model to learn which features from one modality are most relevant to the other, enabling more effective multimodal learning than simple late fusion approaches. This is particularly important for stellar classification where certain spectral features may be more informative when considered in conjunction with specific astrometric properties.
"""
    
    # Save the description to a file
    with open('thesis_figures/model_description.md', 'w') as f:
        f.write(description)
    
    print("Model description saved as 'thesis_figures/model_description.md'")
    return description

def visualize_star_classifier_fusion(model):
    """
    Generate all visualizations and descriptions for the StarClassifierFusion model
    """
    # Create visualizations
    visualize_graphviz(model)
    plot_network_matplotlib(model)
    
    # Try to create computational graph if torchviz is available
    try:
        visualize_torchviz(model)
    except Exception as e:
        print(f"Could not create computational graph: {e}")
    
    # Generate description
    description = generate_model_description()
    
    # Replace placeholder values with actual model parameters
    description = description.format(
        n_layers=model.mamba_spectra.__len__(),
        d_model=max(model.input_proj_spectra.out_features, model.input_proj_gaia.out_features),
        d_model_spectra=model.input_proj_spectra.out_features,
        d_model_gaia=model.input_proj_gaia.out_features,
        n_heads=8,  # Assuming this from the model code
        fusion_dim=model.input_proj_spectra.out_features + model.input_proj_gaia.out_features,
        num_classes=model.classifier[-1].out_features,import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.colors as mcolors
from graphviz import Digraph
import os

# Set the output directory for saving diagrams
os.makedirs('thesis_figures', exist_ok=True)

def visualize_torchviz(model):
    """
    Visualize model using torchviz (requires torchviz package).
    Note: This shows computational graph, not architecture layout.
    """
    try:
        from torchviz import make_dot
        
        # Create dummy inputs matching your model's expected input shape
        dummy_spectra = torch.randn(1, 3647)
        dummy_gaia = torch.randn(1, 18)
        
        # Forward pass to get the graph
        y = model(dummy_spectra, dummy_gaia)
        
        # Create and save dot graph
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.format = 'pdf'
        dot.render('thesis_figures/model_computational_graph')
        print("Computational graph saved as 'thesis_figures/model_computational_graph.pdf'")
    except ImportError:
        print("torchviz not installed. Install with: pip install torchviz")

def visualize_graphviz(model):
    """
    Create a custom visualization using Graphviz
    """
    dot = Digraph(comment='StarClassifierFusion Architecture', 
                 format='pdf', 
                 node_attr={'shape': 'box', 'style': 'filled', 'fontname': 'Arial'})
    
    # Add title
    dot.attr(label='StarClassifierFusion Neural Network Architecture', labelloc='t', fontsize='20')
    
    # Create subgraphs for each section
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs', style='filled', color='lightblue', fontcolor='black')
        c.node('spectra_input', 'Spectral Input\n(B, 3647)', fillcolor='white')
        c.node('gaia_input', 'Gaia Input\n(B, 18)', fillcolor='white')
    
    # Projection layers
    dot.node('spectra_proj', 'Linear Projection\n(B, 2048)', fillcolor='#F8CECC')
    dot.node('gaia_proj', 'Linear Projection\n(B, 2048)', fillcolor='#F8CECC')
    
    # Connections from inputs to projections
    dot.edge('spectra_input', 'spectra_proj')
    dot.edge('gaia_input', 'gaia_proj')
    
    # Reshape (add sequence dimension)
    dot.node('spectra_reshape', 'Unsqueeze\n(B, 1, 2048)', fillcolor='#D5E8D4')
    dot.node('gaia_reshape', 'Unsqueeze\n(B, 1, 2048)', fillcolor='#D5E8D4')
    
    # Connections to reshape
    dot.edge('spectra_proj', 'spectra_reshape')
    dot.edge('gaia_proj', 'gaia_reshape')
    
    # Mamba layers
    with dot.subgraph(name='cluster_mamba_spectra') as c:
        c.attr(label='Mamba2 Encoder (Spectra)', style='filled', color='#DAE8FC', fontcolor='black')
        c.node('mamba_spectra', f'Mamba2 Layers × {model.mamba_spectra.__len__()}\n(B, 1, 2048)', fillcolor='white')
    
    with dot.subgraph(name='cluster_mamba_gaia') as c:
        c.attr(label='Mamba2 Encoder (Gaia)', style='filled', color='#DAE8FC', fontcolor='black')
        c.node('mamba_gaia', f'Mamba2 Layers × {model.mamba_gaia.__len__()}\n(B, 1, 2048)', fillcolor='white')
    
    # Connections to Mamba
    dot.edge('spectra_reshape', 'mamba_spectra')
    dot.edge('gaia_reshape', 'mamba_gaia')
    
    # Cross-attention blocks (if used)
    if model.use_cross_attention:
        with dot.subgraph(name='cluster_cross_attn') as c:
            c.attr(label='Cross-Attention', style='filled', color='#FFE6CC', fontcolor='black')
            c.node('cross_attn_spectra', 'Cross-Attention\n(Spectra → Gaia)', fillcolor='white')
            c.node('cross_attn_gaia', 'Cross-Attention\n(Gaia → Spectra)', fillcolor='white')
        
        # Connections to cross-attention
        dot.edge('mamba_spectra', 'cross_attn_spectra')
        dot.edge('mamba_gaia', 'cross_attn_gaia')
        dot.edge('mamba_gaia', 'cross_attn_spectra', style='dashed', color='gray')
        dot.edge('mamba_spectra', 'cross_attn_gaia', style='dashed', color='gray')
        
        # Mean pooling after cross-attention
        dot.node('pool_spectra', 'Mean Pooling\n(B, 2048)', fillcolor='#D5E8D4')
        dot.node('pool_gaia', 'Mean Pooling\n(B, 2048)', fillcolor='#D5E8D4')
        
        # Connections to pooling
        dot.edge('cross_attn_spectra', 'pool_spectra')
        dot.edge('cross_attn_gaia', 'pool_gaia')
    else:
        # Direct pooling without cross-attention
        dot.node('pool_spectra', 'Mean Pooling\n(B, 2048)', fillcolor='#D5E8D4')
        dot.node('pool_gaia', 'Mean Pooling\n(B, 2048)', fillcolor='#D5E8D4')
        
        # Connections to pooling
        dot.edge('mamba_spectra', 'pool_spectra')
        dot.edge('mamba_gaia', 'pool_gaia')
    
    # Fusion
    dot.node('fusion', 'Concatenation\n(B, 4096)', fillcolor='#FFF2CC')
    
    # Connections to fusion
    dot.edge('pool_spectra', 'fusion')
    dot.edge('pool_gaia', 'fusion')
    
    # Final classification
    dot.node('layer_norm', 'Layer Normalization', fillcolor='#F8CECC')
    dot.node('classifier', 'Linear Classifier\n(B, 55)', fillcolor='#F8CECC')
    dot.node('output', 'Output Logits', fillcolor='#E1D5E7')
    
    # Final connections
    dot.edge('fusion', 'layer_norm')
    dot.edge('layer_norm', 'classifier')
    dot.edge('classifier', 'output')
    
    # Save the diagram
    dot.render('thesis_figures/model_architecture')
    print("Architecture diagram saved as 'thesis_figures/model_architecture.pdf'")

def plot_network_matplotlib(model):
    """
    Create a custom visualization using matplotlib
    This gives more control over the appearance
    """
    # Set up the figure with a light gray background
    fig, ax = plt.subplots(figsize=(12, 16), facecolor='#F8F9F9')
    ax.set_facecolor('#F8F9F9')
    
    # Remove axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#D4F1F9',
        'projection': '#F8FAD7',
        'mamba': '#E8F8F5',
        'cross_attn': '#F5EEF8',
        'pooling': '#D5E8D4',
        'fusion': '#FADBD8',
        'output': '#FCF3CF',
        'arrow': '#05386B'
    }
    
    # Define box dimensions
    box_width = 2.5
    box_height = 0.6
    
    # Function to add a box with a label
    def add_box(x, y, label, color, alpha=1.0):
        rect = Rectangle((x, y), box_width, box_height, 
                         facecolor=color, edgecolor='#05386B', 
                         alpha=alpha, linewidth=1.5, zorder=1)
        ax.add_patch(rect)
        ax.text(x + box_width/2, y + box_height/2, label, 
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, fontweight='bold', zorder=2)
        return (x + box_width/2, y + box_height/2)  # return center point
    
    # Function to add an arrow between boxes
    def add_arrow(start, end, color='#05386B', linestyle='-', linewidth=1.5):
        arrow = FancyArrowPatch(start, end, 
                                arrowstyle='->', color=color, 
                                linewidth=linewidth, linestyle=linestyle,
                                connectionstyle='arc3,rad=0.1', zorder=0)
        ax.add_patch(arrow)
    
    # Title
    plt.title('StarClassifierFusion: Multimodal Architecture with Mamba2 and Cross-Attention', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add subtitle/description
    ax.text(5, 15.5, 'A multimodal deep learning model for stellar classification using spectral and Gaia data',
           horizontalalignment='center', fontsize=12, fontstyle='italic')
    
    # Input layers
    input_spectra = add_box(1.5, 14, 'Spectral Input\n(B, 3647)', colors['input'])
    input_gaia = add_box(6, 14, 'Gaia Input\n(B, 18)', colors['input'])
    
    # Projection layers
    proj_spectra = add_box(1.5, 13, 'Linear Projection\n(B, 2048)', colors['projection'])
    proj_gaia = add_box(6, 13, 'Linear Projection\n(B, 2048)', colors['projection'])
    
    # Reshape layers
    reshape_spectra = add_box(1.5, 12, 'Unsqueeze\n(B, 1, 2048)', colors['projection'])
    reshape_gaia = add_box(6, 12, 'Unsqueeze\n(B, 1, 2048)', colors['projection'])
    
    # Mamba blocks
    # Create a rectangle area for the Mamba subgraph
    spectra_background = Rectangle((1, 9), 3.5, 2.5, facecolor='#D4E6F1', alpha=0.3, edgecolor='#2874A6', linewidth=1)
    gaia_background = Rectangle((5.5, 9), 3.5, 2.5, facecolor='#D4E6F1', alpha=0.3, edgecolor='#2874A6', linewidth=1)
    ax.add_patch(spectra_background)
    ax.add_patch(gaia_background)
    
    # Label for the mamba sections
    ax.text(2.75, 11.3, f'Mamba2 Encoder (Spectra)', 
            horizontalalignment='center', fontsize=10, fontstyle='italic')
    ax.text(7.25, 11.3, f'Mamba2 Encoder (Gaia)', 
            horizontalalignment='center', fontsize=10, fontstyle='italic')
    
    # Individual Mamba layers
    mamba_spectra1 = add_box(1.5, 10.8, f'Mamba2 Layer 1', colors['mamba'])
    mamba_spectra2 = add_box(1.5, 10.2, f'Mamba2 Layer 2', colors['mamba'])
    mamba_spectra_dots = add_box(1.5, 9.6, f'...', 'none')
    mamba_spectraN = add_box(1.5, 9, f'Mamba2 Layer {model.mamba_spectra.__len__()}', colors['mamba'])
    
    mamba_gaia1 = add_box(6, 10.8, f'Mamba2 Layer 1', colors['mamba'])
    mamba_gaia2 = add_box(6, 10.2, f'Mamba2 Layer 2', colors['mamba'])
    mamba_gaia_dots = add_box(6, 9.6, f'...', 'none')
    mamba_gaiaN = add_box(6, 9, f'Mamba2 Layer {model.mamba_gaia.__len__()}', colors['mamba'])
    
    # Cross-attention (if used) or direct pooling
    if model.use_cross_attention:
        # Create a rectangle area for the cross-attention
        cross_attn_background = Rectangle((1, 7), 8, 1.5, facecolor='#E8DAEF', alpha=0.3, edgecolor='#8E44AD', linewidth=1)
        ax.add_patch(cross_attn_background)
        ax.text(5, 8.3, 'Cross-Attention Fusion', horizontalalignment='center', fontsize=10, fontstyle='italic')
        
        cross_attn_spectra = add_box(1.5, 7.5, 'Cross-Attention\n(Spectra → Gaia)', colors['cross_attn'])
        cross_attn_gaia = add_box(6, 7.5, 'Cross-Attention\n(Gaia → Spectra)', colors['cross_attn'])
        
        # Add dashed lines to show cross-connections
        add_arrow((2.75, 9), (7.75, 7.5), linestyle='--', linewidth=1)
        add_arrow((7.25, 9), (2.75, 7.5), linestyle='--', linewidth=1)
        
        # Pooling layers
        pool_spectra = add_box(1.5, 6.5, 'Mean Pooling\n(B, 2048)', colors['pooling'])
        pool_gaia = add_box(6, 6.5, 'Mean Pooling\n(B, 2048)', colors['pooling'])
        
        # Connections
        add_arrow(input_spectra, proj_spectra)
        add_arrow(input_gaia, proj_gaia)
        add_arrow(proj_spectra, reshape_spectra)
        add_arrow(proj_gaia, reshape_gaia)
        add_arrow(reshape_spectra, mamba_spectra1)
        add_arrow(reshape_gaia, mamba_gaia1)
        add_arrow(mamba_spectra1, mamba_spectra2)
        add_arrow(mamba_gaia1, mamba_gaia2)
        add_arrow(mamba_spectra2, mamba_spectra_dots)
        add_arrow(mamba_gaia2, mamba_gaia_dots)
        add_arrow(mamba_spectra_dots, mamba_spectraN)
        add_arrow(mamba_gaia_dots, mamba_gaiaN)
        add_arrow(mamba_spectraN, cross_attn_spectra)
        add_arrow(mamba_gaiaN, cross_attn_gaia)
        add_arrow(cross_attn_spectra, pool_spectra)
        add_arrow(cross_attn_gaia, pool_gaia)
    else:
        # Direct pooling without cross-attention
        pool_spectra = add_box(1.5, 8, 'Mean Pooling\n(B, 2048)', colors['pooling'])
        pool_gaia = add_box(6, 8, 'Mean Pooling\n(B, 2048)', colors['pooling'])
        
        # Connections
        add_arrow(input_spectra, proj_spectra)
        add_arrow(input_gaia, proj_gaia)
        add_arrow(proj_spectra, reshape_spectra)
        add_arrow(proj_gaia, reshape_gaia)
        add_arrow(reshape_spectra, mamba_spectra1)
        add_arrow(reshape_gaia, mamba_gaia1)
        add_arrow(mamba_spectra1, mamba_spectra2)
        add_arrow(mamba_gaia1, mamba_gaia2)
        add_arrow(mamba_spectra2, mamba_spectra_dots)
        add_arrow(mamba_gaia2, mamba_gaia_dots)
        add_arrow(mamba_spectra_dots, mamba_spectraN)
        add_arrow(mamba_gaia_dots, mamba_gaiaN)
        add_arrow(mamba_spectraN, pool_spectra)
        add_arrow(mamba_gaiaN, pool_gaia)
    
    # Fusion and classification
    fusion = add_box(3.8, 5.5, 'Concatenation\n(B, 4096)', colors['fusion'])
    norm = add_box(3.8, 4.5, 'Layer Normalization', colors['projection'])
    classifier = add_box(3.8, 3.5, 'Linear Classifier\n(B, 55)', colors['projection'])
    output = add_box(3.8, 2.5, 'Output Logits', colors['output'])
    
    # Final connections
    add_arrow(pool_spectra, fusion)
    add_arrow(pool_gaia, fusion)
    add_arrow(fusion, norm)
    add_arrow(norm, classifier)
    add_arrow(classifier, output)
    
    # Add a legend for network components
    legend_x = 7.5
    legend_y = 4
    legend_spacing = 0.7
    
    # Legend title
    ax.text(legend_x, legend_y + 1.2, 'Model Components:', 
            fontsize=10, fontweight='bold')
    
    # Create legend entries
    legend_items = [
        ('Input Layer', colors['input']),
        ('Projection Layer', colors['projection']),
        ('Mamba2 Block', colors['mamba']),
        ('Cross-Attention', colors['cross_attn']),
        ('Pooling Layer', colors['pooling']),
        ('Fusion Layer', colors['fusion']),
        ('Output Layer', colors['output'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y - i * legend_spacing
        # Add colored square
        square = Rectangle((legend_x - 0.4, y_pos - 0.1), 0.3, 0.3, 
                          facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(square)
        # Add label
        ax.text(legend_x, y_pos, label, fontsize=9, 
                verticalalignment='center')
    
    # Add model parameters information
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ax.text(5, 1.5, f'Total Parameters: {param_count:,}', 
            horizontalalignment='center', fontsize=10, fontweight='bold')
    
    # Add university logo/attribution placeholder
    ax.text(5, 0.5, 'Your University / Institution Name', 
            horizontalalignment='center', fontsize=10, fontweight='bold')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('thesis_figures/model_architecture_matplotlib.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('thesis_figures/model_architecture_matplotlib.png', dpi=300, bbox_inches='tight')
    print("Matplotlib architecture visualization saved as 'thesis_figures/model_architecture_matplotlib.pdf/png'")

def generate_model_description():
    """
    Generate a professional description of the model architecture for the thesis.
    """
    description = """
## 3.5.1 Model Architecture Overview: StarClassifierFusion

The **StarClassifierFusion** model is a multimodal deep learning architecture designed to leverage both spectral and Gaia astrometric data for stellar classification. The model employs parallel processing branches for different data modalities, with optional cross-modal interaction capabilities, followed by late fusion for final classification.

### Key Components

1. **Dual Modality Encoders**: 
   * **Spectral Branch**: Processes high-dimensional spectral data (3,647 features) through a dedicated encoder.
   * **Gaia Branch**: Processes lower-dimensional astrometric data (18 features) through a separate encoder.

2. **Mamba2 Backbone**:
   * Both branches utilize Mamba2 layers, a state-of-the-art sequence modeling architecture that combines the efficiency of linear recurrent models with the expressivity of attention-based models.
   * Each branch consists of {n_layers} stacked Mamba2 layers with {d_model} hidden dimensions.

3. **Cross-Attention Mechanism**:
   * Facilitates information exchange between the two modality branches.
   * **Spectra-to-Gaia Attention**: The spectral branch attends to relevant features in the Gaia branch.
   * **Gaia-to-Spectra Attention**: The Gaia branch attends to relevant features in the spectral branch.
   * Each cross-attention block employs multi-head attention with {n_heads} heads and residual connections with layer normalization.

4. **Late Fusion**:
   * Concatenation of the processed features from both branches results in a joint representation with {fusion_dim} dimensions.
   * Layer normalization is applied to the concatenated features for stable training.

5. **Classification Head**:
   * A linear layer maps the fused representation to {num_classes} output logits corresponding to different stellar classes.

### Technical Specifications

* **Input Dimensions**:
  * Spectral data: B × {input_dim_spectra} (where B is batch size)
  * Gaia data: B × {input_dim_gaia}
  
* **Embedding Dimensions**:
  * Spectral branch: {d_model_spectra}
  * Gaia branch: {d_model_gaia}
  
* **Mamba2 Configuration**:
  * State dimension: {d_state}
  * Convolution kernel size: {d_conv}
  * Expansion factor: {expand}

* **Model Size**:
  * Total parameters: {param_count:,}
  * Model size: {model_size:.2f} MB

### Design Rationale

This architecture was designed to effectively capture both the fine-grained spectral features and the complementary astrometric information from Gaia, while enabling cross-modal interaction through the attention mechanism. The Mamba2 backbone was selected for its efficiency in processing sequence data compared to Transformer models, while maintaining competitive performance.

The cross-attention fusion strategy allows the model to learn which features from one modality are most relevant to the other, enabling more effective multimodal learning than simple late fusion approaches. This is particularly important for stellar classification where certain spectral features may be more informative when considered in conjunction with specific astrometric properties.
"""
    
    # Save the description to a file
    with open('thesis_figures/model_description.md', 'w') as f:
        f.write(description)
    
    print("Model description saved as 'thesis_figures/model_description.md'")
    return description

def visualize_star_classifier_fusion(model):
    """
    Generate all visualizations and descriptions for the StarClassifierFusion model
    """
    # Create visualizations
    visualize_graphviz(model)
    plot_network_matplotlib(model)
    
    # Try to create computational graph if torchviz is available
    try:
        visualize_torchviz(model)
    except Exception as e:
        print(f"Could not create computational graph: {e}")
    
    # Generate description
    description = generate_model_description()
    
    # Replace placeholder values with actual model parameters
    description = description.format(
        n_layers=model.mamba_spectra.__len__(),
        d_model=max(model.input_proj_spectra.out_features, model.input_proj_gaia.out_features),
        d_model_spectra=model.input_proj_spectra.out_features,
        d_model_gaia=model.input_proj_gaia.out_features,
        n_heads=8,  # Assuming this from the model code
        fusion_dim=model.input_proj_spectra.out_features + model.input_proj_gaia.out_features,
        num_classes=model.classifier[-1].out_features,
        input_dim_spectra=model.input_proj_spectra.in_features,
        input_dim_gaia=model.input_proj_gaia.in_features,
        d_state=256,  # From model defaults
        d_conv=4,     # From model defaults
        expand=2,     # From model defaults
        param_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        model_size=sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    )
    
    # Save the formatted description
    with open('thesis_figures/model_description_formatted.md', 'w') as f:
        f.write(description)
    
    print("All visualizations and descriptions generated successfully!")
    return description

# Example usage (should be replaced with actual model instance)
# visualize_star_classifier_fusion(your_model_instance)

# If you want to use this script with your model, run:
# ----------------------------------------------------------------
# from your_model_file import StarClassifierFusion, CrossAttentionBlock
# 
# # Create model instance with your parameters
# model = StarClassifierFusion(
#     d_model_spectra=2048,
#     d_model_gaia=2048,
#     num_classes=55,
#     input_dim_spectra=3647,
#     input_dim_gaia=18,
#     n_layers=12,
#     use_cross_attention=True,
#     n_cross_attn_heads=8
# )
# 
# # Generate all visualizations
# visualize_star_classifier_fusion(model)
# ----------------------------------------------------------------
        param_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        model_size=sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    )
    
    # Save the formatted description
    with open('thesis_figures/model_description_formatted.md', 'w') as f:
        f.write(description)
    
    print("All visualizations and descriptions generated successfully!")
    return description

# Example usage (should be replaced with actual model instance)
# visualize_star_classifier_fusion(your_model_instance)

# If you want to use this script with your model, run:
# ----------------------------------------------------------------
# from your_model_file import StarClassifierFusion, CrossAttentionBlock
# 
# # Create model instance with your parameters
# model = StarClassifierFusion(
#     d_model_spectra=2048,
#     d_model_gaia=2048,
#     num_classes=55,
#     input_dim_spectra=3647,
#     input_dim_gaia=18,
#     n_layers=12,
#     use_cross_attention=True,
#     n_cross_attn_heads=8
# )
# 
# # Generate all visualizations
# visualize_star_classifier_fusion(model)
# ----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import gzip
import io
import os

def add_vertical_line_between_groups(ax, labels):
    """
    Draws a vertical dashed line on the provided axis between the error and flux groups.
    
    :param ax: The matplotlib axes object where the bar chart is plotted.
    :param labels: List of labels for the bars, ordered such that error columns come first and flux columns second.
    """
    # Count the number of error bars (assumes errors come first)
    num_error = sum(1 for label in labels if label.endswith("_error"))
    if num_error and num_error < len(labels):
        # Vertical line placed between the last error and the first flux bar
        separation_index = num_error - 0.5
        ax.axvline(x=separation_index, color='black', linestyle='--', linewidth=5)


def open_fits_file(file_path):
    """
    Opens a FITS file, handling both regular and gzipped formats.
    
    :param file_path: Path to the FITS file
    :return: FITS HDU list or None if there was an error
    """
    try:
        # Check if the file is gzipped
        with open(file_path, 'rb') as f:
            file_start = f.read(2)
            f.seek(0)  # Reset file pointer
            if file_start == b'\x1f\x8b':  # gzip magic number
                # Handle gzipped file
                with gzip.GzipFile(fileobj=f) as gz_f:
                    file_content = gz_f.read()
                print(f"Opening gzipped file: {file_path}")
                return fits.open(io.BytesIO(file_content), ignore_missing_simple=True)
            else:
                # Handle regular file
                print(f"Opening regular file: {file_path}")
                return fits.open(file_path, ignore_missing_simple=True)
    except Exception as e:
        print(f"Error opening file {os.path.basename(file_path)}: {str(e)}")
        return None

def plot_spectrum_with_gaia_and_cmd(source_id, gaia_lamost_merged, df_sample, correct_df, incorrect_df, n,
                                    spectra_folder="lamost_spectra_uniques", save_path=None):
    """
    Plots the LAMOST spectrum, the Gaia parameters with issues, and a Color–Magnitude Diagram (CMD) 
    in a single figure with three subplots.
    
    :param source_id: Gaia Source ID of the incorrectly classified source.
    :param gaia_lamost_merged: DataFrame containing Gaia and LAMOST cross-matched data.
    :param df_sample: DataFrame containing Gaia photometric and parallax data for the CMD.
    :param correct_df: DataFrame containing correctly classified Gaia IDs.
    :param incorrect_df: DataFrame containing incorrectly classified Gaia IDs.
    :param spectra_folder: Path to the folder containing LAMOST FITS spectra.
    :param save_path: If provided, the complete figure is saved to this path.
    """
    try:
        if 'obsid' not in gaia_lamost_merged.columns:
            print("⚠️ 'obsid' column not found in gaia_lamost_merged.")
            return

        match = gaia_lamost_merged.loc[gaia_lamost_merged['source_id'] == source_id]
        if match.empty:
            print(f"⚠️ No LAMOST match found for source_id {source_id}.")
            return

        obsid = int(match.iloc[0]['obsid'])
        print(f"Found match: Source ID {source_id} -> ObsID {obsid}")

        fits_path = f"{spectra_folder}/{int(obsid)}"
        
        # Use the open_fits_file function to handle both regular and gzipped FITS files
        hdul = open_fits_file(fits_path)
        
        if hdul is None:
            print(f"⚠️ Failed to open FITS file for ObsID {obsid}.")
            return
            
        # Process the FITS data
        try:
            # After opening the FITS file, add debugging:
            print(f"FITS file structure for ObsID {obsid}:")
            for i, hdu in enumerate(hdul):
                print(f"  HDU {i}: {hdu.__class__.__name__}, shape={getattr(hdu.data, 'shape', 'No data')}")
            
            # LAMOST DR5 and later uses BinTableHDU in the first extension
            if len(hdul) > 1 and isinstance(hdul[1], fits.BinTableHDU):
                print("Using data from BinTableHDU (extension 1)")
                table_data = hdul[1].data
                
                # Debug table column names
                print(f"  BinTable columns: {table_data.names}")
                
                # For LAMOST spectra, typical column names are 'FLUX', 'WAVELENGTH', 'LOGLAM', etc.
                # Use appropriate column names based on what's available
                if 'FLUX' in table_data.names and 'WAVELENGTH' in table_data.names:
                    flux = table_data['FLUX'][0]  # First row
                    wavelength = table_data['WAVELENGTH'][0]
                    print(f"  Using FLUX and WAVELENGTH columns")
                elif 'FLUX' in table_data.names and 'LOGLAM' in table_data.names:
                    flux = table_data['FLUX'][0]  # First row
                    # Convert log wavelength to linear wavelength
                    log_wavelength = table_data['LOGLAM'][0]
                    wavelength = 10**log_wavelength
                    print(f"  Using FLUX and LOGLAM (converted) columns")
                # Add more conditions for different column naming conventions
                else:
                    # If column names don't match known formats, try first two columns
                    # (often wavelength is first, flux is second)
                    print(f"  Unknown column format, using first two columns")
                    wavelength = table_data[table_data.names[0]][0]
                    flux = table_data[table_data.names[1]][0]
            # Fallback to original method with primary HDU
            elif hdul[0].data is not None and len(hdul[0].data.shape) >= 1:
                print("Using data from PrimaryHDU")
                data = hdul[0].data
                if data.shape[0] < 3:
                    print(f"⚠️ Skipping {obsid}: Primary HDU data has insufficient dimensions: {data.shape}")
                    return
                flux = data[0]
                wavelength = data[2]
            else:
                print(f"⚠️ Skipping {obsid}: No usable data found in FITS file.")
                return
                
            # Check that we have valid data before proceeding
            if flux is None or wavelength is None or len(flux) == 0 or len(wavelength) == 0:
                print(f"⚠️ Skipping {obsid}: Empty flux or wavelength arrays")
                return
                
            print(f"  Data loaded successfully. Wavelength range: {min(wavelength):.2f}-{max(wavelength):.2f} Å")
            print(f"  Flux range: {min(flux):.2e}-{max(flux):.2e}")
            
            # Create a figure with three subplots using GridSpec.
            # Top row: two subplots (spectrum and Gaia issues), Bottom row: CMD spanning full width.
            fig = plt.figure(figsize=(24, 12))
            gs = fig.add_gridspec(1, 3, height_ratios=[1])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[0, 2])
            ax3 = fig.add_subplot(gs[0, 0])
            plt.rcParams.update({'font.size': 16})

            # --- Subplot 1: LAMOST Spectrum ---
            ax1.plot(wavelength, flux, color='blue', alpha=0.7, lw=1, zorder=10)
            ax1.set_xlabel("Wavelength (Å)")
            ax1.set_ylabel("Flux")
            ax1.set_title(f"LAMOST Spectrum for Unlabelled CV")
            ax1.grid(zorder=0)

            # --- Subplot 2: Gaia Parameters with Issues ---
            ax2.grid(zorder=0)
            gaia_info = match.iloc[[0]].drop(columns=["source_id", "obsid"], errors='ignore')
            issues_dict = {}
            issues_text_list = []
            for col in gaia_info.columns:
                value = gaia_info[col].values[0]
                if col.endswith("_error") and value > 1:
                    issues_dict[col] = value
                    issues_text_list.append(f"Large error in {col}")
                elif col.endswith("_flux") and value < -1:
                    issues_dict[col] = value
                    issues_text_list.append(f"Dim object in {col}")

            if issues_dict:
                # Order the labels: errors first, then fluxes.
                error_labels = [l for l in issues_dict.keys() if l.endswith("_error")]
                flux_labels = [l for l in issues_dict.keys() if l.endswith("_flux")]
                ordered_labels = error_labels + flux_labels
                ordered_values = [issues_dict[l] for l in ordered_labels]

                ax2.bar(ordered_labels, ordered_values, color='skyblue', zorder=3)
                ax2.tick_params("x", labelrotation=45)
                ax2.set_title("Gaia Parameters with Issues")
                ax2.set_ylabel("Standard Deviations from Mean")

                # Add a vertical dashed line between error and flux groups.
                add_vertical_line_between_groups(ax2, ordered_labels)
            else:
                ax2.text(0.5, 0.5, "No significant data issues", ha='center', va='center', fontsize=12)
                ax2.axis("off")

            # --- Subplot 3: Color–Magnitude Diagram (CMD) ---
            # Compute additional columns for CMD.
            df_sample['color'] = df_sample['phot_bp_mean_mag'] - df_sample['phot_rp_mean_mag']
            df_sample['distance_pc'] = 1000 / df_sample['parallax']
            df_sample['abs_mag'] = df_sample['phot_g_mean_mag'] - 5 * np.log10(df_sample['distance_pc'] / 10)
            df_sample['is_correct'] = df_sample['source_id'].isin(correct_df['source_id'])
            df_sample['is_incorrect'] = df_sample['source_id'].isin(incorrect_df['source_id'])

            # Plot background stars (those not flagged as correct or incorrect)
            mask_background = ~(df_sample['is_correct'] | df_sample['is_incorrect'])
            ax3.scatter(df_sample.loc[mask_background, 'color'], 
                        df_sample.loc[mask_background, 'abs_mag'],
                        s=3, color='gray', alpha=0.6, label='Nearby Stars')

            # Plot the incorrect in red.
            ax3.scatter(df_sample[df_sample['is_incorrect']]['color'],
                        df_sample[df_sample['is_incorrect']]['abs_mag'],
                        s=100, color='red', label='Incorrectly Classified', alpha=1, 
                        edgecolor='black', marker='H')
            
            # Plot the correct in green.
            ax3.scatter(df_sample[df_sample['is_correct']]['color'],
                        df_sample[df_sample['is_correct']]['abs_mag'],
                        s=100, color='green', label='Correctly Classified', alpha=1, 
                        edgecolor='black', marker='x')
            
            # Plot the target source in blue. FLUX IS NOT THE SAME AS MAGNITUDE, data for both exist in the Gaia table.
            target_color = df_sample.loc[df_sample['source_id'] == source_id, 'color'].values[0]
            target_abs_mag = df_sample.loc[df_sample['source_id'] == source_id, 'abs_mag'].values[0]
            ax3.scatter(target_color, target_abs_mag, s=200, color='blue', label='Target Source', alpha=1, edgecolor='black', marker='o')
            #target_abs_mag = match['phot_g_mean_flux'].values[0] - 5 * np.log10((1/match['parallax'].values[0] )/ 10)
            #ax3.scatter(target_color, target_abs_mag, s=200, color='blue', label='Target Source', alpha=1, edgecolor='black', marker='o')


            # In a CMD, brighter (lower) magnitudes are at the top.
            ax3.invert_yaxis()
            ax3.set_xlim(-0.5, 3.5)
            ax3.set_ylim(14, 0.5)
            ax3.set_xlabel('Colour (BP - RP)')
            ax3.set_ylabel('Absolute G Magnitude')
            ax3.set_title('Color–Magnitude Diagram (CMD)')
            ax3.legend(loc='lower right')

            plt.tight_layout()
            if save_path:
                save_path= save_path.replace(".png", f"_{n}.png")
                plt.savefig(save_path)
            plt.show()
                
        except Exception as e:
            print(f"Error processing FITS data for source_id {source_id}: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Ensure proper cleanup of FITS file
            if hdul is not None:
                try:
                    hdul.close()
                except Exception as e:
                    print(f"Warning: Could not close FITS file: {e}")
                    
    except Exception as e:
        print(f"Error in overall processing for source_id {source_id}: {e}")
        import traceback
        traceback.print_exc()

# Example type conversions (ensure these columns are in the correct type)
gaia_lamost_merged['obsid'] = gaia_lamost_merged['obsid'].astype(int)
gaia_lamost_merged['source_id'] = gaia_lamost_merged['source_id'].astype(int)

# Initialize a counter for the save path
n_=1

# Loop through incorrectly classified sources and plot all spectra with labels if Gaia data is problematic.
for source_id in fn_prime_gaia_ids.astype(int):
    plot_spectrum_with_gaia_and_cmd(source_id, gaia_lamost_merged, save_path=f"Images_and_Plots/CMD_Spectra_Gaia_CV.png", df_sample=df_sample, correct_df=correct_df, incorrect_df=incorrect_df, n=n_)
    n_+=1

# Define the ADQL query to fetch detailed information for the Correctly Classified Gaia IDs
query = """
SELECT source_id, ra, dec, parallax, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_mag, parallax_error
FROM gaiadr3.gaia_source
WHERE source_id IN ({})
"""

# Join the source IDs into a single string
source_ids_str = ",".join([str(id) for id in correct_gaia_ids])
full_query = query.format(source_ids_str)

# Run the query asynchronously
job = Gaia.launch_job_async(full_query)
results = job.get_results()

# Convert to Pandas DataFrame
correct_df = results.to_pandas()

print(f"✅ Retrieved detailed information for {len(correct_df)} correctly classified Gaia IDs.")

# Define the ADQL query to fetch detailed information for the incorrectly Classified Gaia IDs
query = """
SELECT source_id, ra, dec, parallax, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_mag, parallax_error
FROM gaiadr3.gaia_source
WHERE source_id IN ({})
"""

# Join the source IDs into a single string
source_ids_str = ",".join([str(id) for id in incorrect_gaia_ids])
full_query = query.format(source_ids_str)

# Run the query asynchronously
job = Gaia.launch_job_async(full_query)
results = job.get_results()

# Convert to Pandas DataFrame
incorrect_df = results.to_pandas()

print(f"✅ Retrieved detailed information for {len(incorrect_df)} incorrectly classified Gaia IDs.")


# (Re-using and adapting code from the previous response)
# Make sure to have the 'table9.dat' file for rejected IDs.

def process_rejected_ids_file(filepath):
    """Reads a file containing GaiaDR3 source IDs (one per line after header)
    and returns a comma-separated string of these IDs."""
    rejected_ids_list = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line.startswith('---') or not line:
                    continue
                rejected_ids_list.append(line)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath} for rejected IDs.")
        return None
    return ",".join(rejected_ids_list)

def get_missing_rr_lyrae_ids_string():
    """Returns a comma-separated string of the 14 known missing RR Lyrae source_ids."""
    missing_ids_list = [
        '4092009204924599040', '4120414435009794048', '4144246349481643392',
        '5797652730842515968', '5797917193442176640', '5846086424210395520',
        '5917239841741208576', '5991733644318583424', '6017924835910361344',
        '6069336998880602240', '6707009423228603904', '5935214760885709440',
        '4362766825101261952', '5967334102579505664'
    ]
    return ",".join(missing_ids_list)

def get_best_rr_lyrae_ids(rejected_ids_filepath="table9.dat", 
                           use_astroquery=True, 
                           login_credentials=None, 
                           max_retries=3,
                           retry_delay=10):
    """
    Fetches the source_ids for the best RR Lyrae dataset from Gaia DR3
    as defined by Clementini et al. (2022).

    Args:
        rejected_ids_filepath (str): Path to the file containing rejected IDs.
        use_astroquery (bool): If True, attempts to run the query using astroquery.
                               If False, prints the query.
        login_credentials (dict, optional): Gaia login credentials for astroquery.
        max_retries (int): Number of retries for the Gaia query.
        retry_delay (int): Delay in seconds between retries.


    Returns:
        list: A list of Gaia source_ids (as integers), or None if fetching fails.
    """
    print("Fetching best RR Lyrae source_ids...")
    rejected_ids_string = process_rejected_ids_file(rejected_ids_filepath)
    missing_ids_string = get_missing_rr_lyrae_ids_string()

    if rejected_ids_string is None:
        print("Cannot proceed without the list of rejected IDs.")
        return None

    adql_query = f"""
    SELECT source_id FROM (
        SELECT source_id
        FROM gaiadr3.vari_rrlyrae
        WHERE source_id NOT IN ({rejected_ids_string})
        UNION
        SELECT source_id
        FROM gaiadr3.vari_classifier_result
        WHERE source_id IN ({missing_ids_string})
        -- Optional: Add more specific conditions for the 14 if needed
        -- AND classlabel_best = 'RRLYR'
        -- AND (best_classifier_name LIKE '%SOS_CEP_RRL_RRAB%' OR best_classifier_name LIKE '%SOS_CEP_RRL_RRC%' OR best_classifier_name LIKE '%SOS_CEP_RRL_RRD%')
    ) AS combined_rrlyrae_ids
    """

    if not use_astroquery:
        print("\nGenerated ADQL Query for Best RR Lyrae IDs:")
        print(adql_query)
        print("\nCopy the above query and run it on the Gaia Archive website.")
        
        return None # Or raise an exception if you expect IDs

    print("Attempting to execute query with astroquery.gaia...")
    for attempt in range(max_retries):
        try:
            if login_credentials:
                Gaia.login(user=login_credentials['user'], password=login_credentials['password'])
            else:
                Gaia.login_anonymous()

            job = Gaia.launch_job_async(adql_query)
            results_table = job.get_results()
            
            if results_table and 'source_id' in results_table.colnames:
                # Convert to list of integers
                rr_lyrae_ids = results_table['source_id'].tolist()
                print(f"✅ Successfully fetched {len(rr_lyrae_ids)} RR Lyrae source_ids.")
                return rr_lyrae_ids
            else:
                print("Warning: Query executed but no 'source_id' column or no results found.")
                return [] # Return empty list if no IDs

        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__} - {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  Max retries reached. Failed to fetch RR Lyrae IDs.")
                return None
    return None # Should not be reached if max_retries > 0

# ... (all your existing imports and function definitions) ...

# <<< ADD THE HELPER FUNCTIONS FROM ABOVE HERE (process_rejected_ids_file, get_missing_rr_lyrae_ids_string, get_best_rr_lyrae_ids) >>>

# --- Main execution block (modified) ---
if __name__ == "__main__":
    # --- Configuration for your pipeline ---
    lamost_catalogue_path = "lamost/minimal.csv"  # Load LAMOST catalog
    model_path = "Models/model_fusion_mambaoutv3.pth"
    gaia_transformer_path = "Pickles/gaia_normalization.pkl"
    rejected_rr_lyrae_ids_file = "Pickles/Table9_RR*.txt"  # Path to the file with rejected IDs

    # --- Step 0: Fetch the best RR Lyrae Gaia IDs ---
    print("\n🌟 Step 0: Fetching the list of best RR Lyrae Gaia DR3 IDs...")
    
    # Option 1: Use astroquery (recommended for automation)
    # You might need to provide Gaia login credentials if anonymous access is insufficient
    # gaia_login_creds = {'user': 'your_username', 'password': 'your_password'}
    gaia_login_creds = None 
    all_rr_lyrae_gaia_ids = get_best_rr_lyrae_ids(
        rejected_ids_filepath=rejected_rr_lyrae_ids_file,
        use_astroquery=False
    )

    # Option 2: If you want to manually run the ADQL (set use_astroquery=False)
    # get_best_rr_lyrae_ids(rejected_ids_filepath=rejected_rr_lyrae_ids_file, use_astroquery=False)
    # print("After running the query manually, load the IDs, e.g., from a CSV:")
    # loaded_rr_lyrae_df = pd.read_csv("path_to_your_downloaded_rr_lyrae_ids.csv")
    # all_rr_lyrae_gaia_ids = loaded_rr_lyrae_df['source_id'].tolist()


    if all_rr_lyrae_gaia_ids is None or not all_rr_lyrae_gaia_ids:
        print("⚠️ Could not obtain RR Lyrae Gaia IDs. Exiting pipeline.")
    else:
        print(f"Retrieved {len(all_rr_lyrae_gaia_ids)} RR Lyrae source_ids for processing.")
        
        # For testing, you might want to use a smaller subset
        # rr_lyrae_ids_to_process = all_rr_lyrae_gaia_ids[:1000] 
        rr_lyrae_ids_to_process = all_rr_lyrae_gaia_ids # Process all

        print(f"Selected {len(rr_lyrae_ids_to_process)} RR Lyrae IDs for the pipeline.")

        # --- Load LAMOST catalog ---
        try:
            lamost_catalogue = pd.read_csv(lamost_catalogue_path)
        except FileNotFoundError:
            print(f"Error: LAMOST catalogue not found at {lamost_catalogue_path}. Exiting.")
            exit()
        
        # --- Run your prediction pipeline ---
        df_predictions_rrlyrae, gaia_lamost_merged_rrlyrae = predict_star_labels(
            gaia_ids=rr_lyrae_ids_to_process,
            model_path=model_path,
            lamost_catalogue=lamost_catalogue, # Pass the loaded DataFrame
            gaia_transformer_path=gaia_transformer_path
        )

        if df_predictions_rrlyrae is not None and not df_predictions_rrlyrae.empty:
            print("\n✅ RR Lyrae Prediction Pipeline Completed.")
            # Save the predictions
            df_predictions_rrlyrae.to_csv("rr_lyrae_predictions.csv", index=False)
            print("RR Lyrae predictions saved to rr_lyrae_predictions.csv")
            if gaia_lamost_merged_rrlyrae is not None and not gaia_lamost_merged_rrlyrae.empty:
                 gaia_lamost_merged_rrlyrae.to_csv("rr_lyrae_gaia_lamost_merged.csv", index=False)
                 print("RR Lyrae merged data saved to rr_lyrae_gaia_lamost_merged.csv")

            # --- Example: Plotting or further analysis for RR Lyrae ---
            # You can now use df_predictions_rrlyrae and gaia_lamost_merged_rrlyrae
            # for your plotting and analysis specific to RR Lyrae stars.

            # For instance, if you want to see how many were predicted as "RRLyr*":
            if "RRLyr*" in df_predictions_rrlyrae.columns:
                predicted_as_rrlyr_count = df_predictions_rrlyrae["RRLyr*"].sum()
                print(f"Number of stars predicted as 'RRLyr*': {predicted_as_rrlyr_count} out of {len(df_predictions_rrlyrae)}")

            # Example plotting call (you'll need to adapt it for RR Lyrae truth if available)
            # For now, let's just plot the first few if they exist
            # This part needs careful thought on what 'correct_df' and 'incorrect_df' mean for RR Lyrae
            # since you are *predicting* them.
            # For demonstration, let's just pick some from the merged data for plotting.
            
            # if gaia_lamost_merged_rrlyrae is not None and not gaia_lamost_merged_rrlyrae.empty:
            #     sample_ids_for_plotting = gaia_lamost_merged_rrlyrae['source_id'].unique()[:min(5, len(gaia_lamost_merged_rrlyrae['source_id'].unique()))]
            #     
            #     # Create dummy correct/incorrect DFs for plotting function if not otherwise defined for RR Lyrae
            #     # In a real scenario, you'd compare predictions to the known RR Lyrae status.
            #     # For now, this is just to make the plot function runnable.
            #     dummy_correct_df = pd.DataFrame({'source_id': []}) 
            #     dummy_incorrect_df = pd.DataFrame({'source_id': sample_ids_for_plotting}) 
            #
            #     print(f"\nPlotting for a few RR Lyrae samples...")
            #     for n_idx, sid in enumerate(sample_ids_for_plotting):
            #         print(f"Plotting for source_id: {sid}")
            #         # You'd need a df_sample_rrlyrae equivalent for the CMD background
            #         # For now, using the EB one as a placeholder - THIS NEEDS REFINEMENT
            #         df_sample_cmd_background = pd.read_csv("gaia_sample_combined_detailed.csv") # Placeholder
            #
            #         plot_spectrum_with_gaia_and_cmd(
            #             source_id=sid,
            #             gaia_lamost_merged=gaia_lamost_merged_rrlyrae,
            #             df_sample=df_sample_cmd_background, # Placeholder background
            #             correct_df=dummy_correct_df,     # Placeholder
            #             incorrect_df=dummy_incorrect_df, # Placeholder
            #             n=n_idx,
            #             spectra_folder="lamost_spectra_uniques",
            #             save_path="Images_and_Plots/CMD_Spectra_Gaia_RRLyrae.png"
            #         )
        else:
            print("⚠️ RR Lyrae Prediction Pipeline did not produce results.")

# ... (your existing code for Eclipsing Binaries can remain below or be separated)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia # For querying Gaia archive

# --- 0. Configuration ---
GAIA_TABLE_NAME = "gaiadr3.gaia_source" # Gaia Data Release 3
# GAIA_TABLE_NAME = "gaiaedr3.gaia_source" # Gaia Early Data Release 3 (if preferred)
SAVE_FIGURE = True
FIGURE_FILENAME = "cmd_model_rr_stars_queried.png"
# Gaia TAP has limits on the number of IDs in an IN clause.
# Typically around 50,000 but can vary. We'll batch if needed.
MAX_SOURCE_IDS_PER_QUERY = 40000

# --- 1. Load Model Predictions and Class Information ---
try:
    y_pred_full = np.load("y_predictions_ecl.npy")
except FileNotFoundError:
    print("Error: 'y_predictions_ecl.npy' not found. Please check the path.")
    exit()

try:
    classes = pd.read_pickle("Pickles/Updated_List_of_Classes_ubuntu.pkl")
    if not isinstance(classes, list):
        classes = list(classes)
except FileNotFoundError:
    print("Error: 'Pickles/Updated_List_of_Classes_ubuntu.pkl' not found. Please check the path.")
    exit()

print("Classes loaded:", classes)

# Separate predictions from source_ids
y_pred_model = np.array(y_pred_full[:, :-1], dtype=int)
# Get all source_ids from your predictions file; these are the stars we need Gaia data for.
try:
    all_source_ids_from_model = y_pred_full[:, -1].astype(np.int64)
except ValueError:
    print("Warning: Could not convert all source_ids to int64. Attempting to clean...")
    all_source_ids_from_model = pd.to_numeric(y_pred_full[:, -1], errors='coerce')
    all_source_ids_from_model = all_source_ids_from_model[~np.isnan(all_source_ids_from_model)].astype(np.int64)

unique_source_ids_to_query = pd.unique(all_source_ids_from_model)
print(f"Found {len(all_source_ids_from_model)} total predictions, corresponding to {len(unique_source_ids_to_query)} unique source_ids to query from Gaia.")

if len(unique_source_ids_to_query) == 0:
    print("No source_ids found in predictions file. Exiting.")
    exit()

# --- 2. Identify RR* Stars from Model Predictions (using their source_ids) ---
try:
    rr_star_class_index = classes.index("RR*")
    print(f"Index for 'RR*' class: {rr_star_class_index}")
except ValueError:
    print("Error: 'RR*' class not found in the loaded classes list.")
    print("Available classes are:", classes)
    exit()

# Get the column for RR* predictions
rr_star_predictions_column = y_pred_model[:, rr_star_class_index]
# Create a boolean mask for stars predicted as RR*
model_predicted_rr_star_mask = (rr_star_predictions_column == 1)
# Get the source_ids of these predicted RR* stars
predicted_rr_star_source_ids = all_source_ids_from_model[model_predicted_rr_star_mask]
predicted_rr_star_source_ids_unique = pd.unique(predicted_rr_star_source_ids) # Ensure unique

print(f"Found {len(predicted_rr_star_source_ids_unique)} unique sources classified as 'RR*' by the model.")
if len(predicted_rr_star_source_ids_unique) == 0:
    print("No 'RR*' stars were classified by the model by source_id.")

# --- 3. Query Gaia Archive for Photometric Data ---
print("\n--- Querying Gaia Archive ---")

def query_gaia_for_sources_batched(source_ids_list_np_array):
    """
    Queries Gaia archive for a list of source_ids in batches.
    Returns a Pandas DataFrame.
    """
    if not source_ids_list_np_array.size:
        print("No source_ids provided for Gaia query.")
        return pd.DataFrame()

    all_gaia_data_dfs = []
    num_total_ids = len(source_ids_list_np_array)
    num_batches = (num_total_ids + MAX_SOURCE_IDS_PER_QUERY - 1) // MAX_SOURCE_IDS_PER_QUERY

    for i in range(num_batches):
        start_idx = i * MAX_SOURCE_IDS_PER_QUERY
        end_idx = min((i + 1) * MAX_SOURCE_IDS_PER_QUERY, num_total_ids)
        batch_ids_np = source_ids_list_np_array[start_idx:end_idx]
        
        # Convert numpy array of IDs to a comma-separated string for the ADQL query
        id_list_str = ",".join(batch_ids_np.astype(str))
        
        query = f"""
        SELECT source_id, ra, dec, parallax, parallax_error,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
               ruwe -- Renormalised Unit Weight Error, good quality indicator
        FROM {GAIA_TABLE_NAME}
        WHERE source_id IN ({id_list_str})
        """
        print(f"Executing Gaia query for batch {i+1}/{num_batches} (Sources {start_idx+1} to {end_idx} of {num_total_ids})...")
        try:
            # Using synchronous job for simplicity here, can be changed to async for very large datasets
            job = Gaia.launch_job(query) # Synchronous
            # For asynchronous:
            # job = Gaia.launch_job_async(query)
            results_table = job.get_results()
            if len(results_table) > 0:
                gaia_df_batch = results_table.to_pandas()
                all_gaia_data_dfs.append(gaia_df_batch)
                print(f"Batch {i+1} query successful, retrieved {len(gaia_df_batch)} rows from Gaia.")
            else:
                print(f"Batch {i+1} query returned no results from Gaia for the given IDs.")
        except Exception as e:
            print(f"Error querying Gaia for batch {i+1}: {e}")
            # Optionally, add retry logic or decide how to handle partial failures
    
    if not all_gaia_data_dfs:
        print("Gaia query yielded no results across all batches or all batches failed.")
        return pd.DataFrame()
        
    df_gaia_queried = pd.concat(all_gaia_data_dfs, ignore_index=True)
    
    if 'source_id' in df_gaia_queried.columns:
        df_gaia_queried['source_id'] = df_gaia_queried['source_id'].astype(np.int64)
    else:
        print("Critical Error: 'source_id' column not found in Gaia query result. Cannot proceed.")
        return pd.DataFrame() # Return empty DF

    print(f"Total Gaia data queried. Shape: {df_gaia_queried.shape}")
    return df_gaia_queried

# Query Gaia for all unique source_ids identified from your y_pred file
df_gaia_cmd_data = query_gaia_for_sources_batched(unique_source_ids_to_query)

if df_gaia_cmd_data.empty:
    print("Error: Failed to retrieve necessary data from Gaia. Cannot proceed with CMD.")
    exit()

# Essential columns for CMD (check if they exist after query)
required_gaia_cols = ['source_id', 'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax']
missing_cols = [col for col in required_gaia_cols if col not in df_gaia_cmd_data.columns]
if missing_cols:
    print(f"Error: The following required columns are missing from the Gaia query result: {missing_cols}")
    print(f"Available columns in queried data: {df_gaia_cmd_data.columns.tolist()}")
    exit()

# --- 4. Calculate CMD Parameters (Color and Absolute Magnitude) ---
print("\n--- Calculating CMD Parameters ---")
df_cmd = df_gaia_cmd_data.copy() # Use the freshly queried data

# Drop rows with NaN in essential photometric/parallax columns for CMD
df_cmd.dropna(subset=required_gaia_cols, inplace=True)
print(f"Shape after dropping NaNs in essential CMD columns: {df_cmd.shape}")

# Filter for valid parallax values (must be positive for distance calculation)
df_cmd = df_cmd[df_cmd['parallax'] > 0]
# Optional: Add a more stringent parallax quality cut, e.g., parallax_over_error
# if 'parallax_error' in df_cmd.columns and df_cmd['parallax_error'] is not None:
#     df_cmd = df_cmd[df_cmd['parallax'] / df_cmd['parallax_error'] > 5] # Example cut
#if 'ruwe' in df_cmd.columns: # RUWE < 1.4 is often a good quality indicator
#    df_cmd = df_cmd[df_cmd['ruwe'] < 1.4]


print(f"Shape after filtering for positive parallax (and any other quality cuts): {df_cmd.shape}")

if df_cmd.empty:
    print("Error: No valid Gaia data remains after filtering for CMD plotting. Cannot generate CMD.")
    exit()

# Calculate color (G_BP - G_RP)
df_cmd['color'] = df_cmd['phot_bp_mean_mag'] - df_cmd['phot_rp_mean_mag']

# Calculate distance in parsecs (parallax is typically in milliarcseconds (mas))
df_cmd['distance_pc'] = 1000.0 / df_cmd['parallax']

# Calculate absolute G magnitude (M_G)
# M = m - 5 * (log10(d_pc) - 1)
df_cmd['abs_mag_g'] = df_cmd['phot_g_mean_mag'] - 5 * (np.log10(df_cmd['distance_pc']) - 1)

# Flag which stars in the CMD data were predicted as RR* by your model
# Use the unique list of RR* source IDs derived earlier
df_cmd['is_model_rr_star'] = df_cmd['source_id'].isin(predicted_rr_star_source_ids_unique)

# --- 5. Create and Display the CMD Plot ---
print("\n--- Generating CMD Plot ---")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 10))

# Plot all stars from df_cmd (which now contains all queried sources with valid CMD data)
# as background, unless they are model-classified RR*
background_stars_mask = ~df_cmd['is_model_rr_star']
plt.scatter(df_cmd.loc[background_stars_mask, 'color'],
            df_cmd.loc[background_stars_mask, 'abs_mag_g'],
            s=5,  # Small size for background stars
            color='grey',
            alpha=0.4, # Low alpha for density effect
            label='Background Stars (from model input)',
            zorder=1) # Plot background first

# Plot the RR* stars classified by the model that are present in the CMD data
model_rr_stars_in_cmd = df_cmd[df_cmd['is_model_rr_star']]
if not model_rr_stars_in_cmd.empty:
    plt.scatter(model_rr_stars_in_cmd['color'],
                model_rr_stars_in_cmd['abs_mag_g'],
                s=70, # Larger size for highlighted stars
                color='red',
                edgecolor='black',
                marker='*', # Star marker
                label=f'Model Classified RR* ({len(model_rr_stars_in_cmd)})',
                zorder=2) # Plot on top
    print(f"Plotting {len(model_rr_stars_in_cmd)} model-classified RR* stars found with valid CMD data.")
else:
    print("No model-classified RR* stars have valid CMD data to plot, or none were found in the Gaia query results for RR* IDs.")

# Set plot labels and title
plt.xlabel('Color ($G_{BP} - G_{RP}$)', fontsize=14)
plt.ylabel('Absolute G Magnitude ($M_G$)', fontsize=14)
plt.title('Color-Magnitude Diagram (Gaia Data Queried from Model Input)', fontsize=16)

# Invert the y-axis (brighter stars at the top, smaller magnitude values)
plt.gca().invert_yaxis()

# Optional: Set plot limits (adjust as needed for your data's actual range)
# You might want to calculate these dynamically based on the quantiles of your data
# Example:
# color_min, color_max = df_cmd['color'].quantile(0.01), df_cmd['color'].quantile(0.99)
# abs_mag_min, abs_mag_max = df_cmd['abs_mag_g'].quantile(0.01), df_cmd['abs_mag_g'].quantile(0.99)
# plt.xlim(max(-1.0, color_min - 0.5), min(4.0, color_max + 0.5))
# plt.ylim(min(18, abs_mag_max + 2), max(-7, abs_mag_min - 2)) # Remember y-axis is inverted

# Add a legend
plt.legend(fontsize=12, loc='upper right') # 'best' or a specific location

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout() # Adjust layout to prevent labels from overlapping

if SAVE_FIGURE:
    plt.savefig(FIGURE_FILENAME, dpi=300)
    print(f"CMD plot saved as {FIGURE_FILENAME}")

plt.show()

print("\nScript finished.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia

# --- 0. Configuration ---
GAIA_TABLE_NAME = "gaiadr3.gaia_source"
SAVE_FIGURE = True
FIGURE_FILENAME = "cmd_model_rr_stars_with_dense_background.png"
MAX_SOURCE_IDS_PER_MODEL_QUERY_BATCH = 40000 # For querying your model's specific IDs

# Configuration for the DENSE BACKGROUND query
BACKGROUND_QUERY_LIMIT = 200000  # How many background stars to fetch (adjust as needed)
BACKGROUND_PARALLAX_MIN = 1.0   # mas, e.g., > 1 mas means within 1 kpc
BACKGROUND_G_MAG_MAX = 19.0     # Apparent G magnitude limit for background
BACKGROUND_RUWE_MAX = 1.4       # Quality cut for background stars

# --- 1. Load Model Predictions and Class Information ---
try:
    y_pred_full = np.load("y_predictions_ecl.npy")
except FileNotFoundError:
    print("Error: 'y_predictions_ecl.npy' not found. Please check the path.")
    exit()

try:
    classes = pd.read_pickle("Pickles/Updated_List_of_Classes_ubuntu.pkl")
    if not isinstance(classes, list):
        classes = list(classes)
except FileNotFoundError:
    print("Error: 'Pickles/Updated_List_of_Classes_ubuntu.pkl' not found. Please check the path.")
    exit()
print("Classes loaded.")

# Source IDs from your model's input file
y_pred_model_classes = np.array(y_pred_full[:, :-1], dtype=int)
all_source_ids_from_model_input = y_pred_full[:, -1]
try:
    unique_source_ids_for_model_query = pd.unique(all_source_ids_from_model_input.astype(np.int64))
except ValueError:
    temp_ids = pd.to_numeric(all_source_ids_from_model_input, errors='coerce')
    unique_source_ids_for_model_query = pd.unique(temp_ids[~np.isnan(temp_ids)].astype(np.int64))

print(f"Found {len(unique_source_ids_for_model_query)} unique source_ids from model input file.")

# --- 2. Identify RR* Stars from Model Predictions ---
try:
    rr_star_class_index = classes.index("RR*")
except ValueError:
    print("Error: 'RR*' class not found. Available:", classes)
    exit()

rr_star_predictions_column = y_pred_model_classes[:, rr_star_class_index]
model_predicted_rr_star_mask = (rr_star_predictions_column == 1)
predicted_rr_star_source_ids = all_source_ids_from_model_input[model_predicted_rr_star_mask]
try:
    predicted_rr_star_source_ids_unique = pd.unique(predicted_rr_star_source_ids.astype(np.int64))
except ValueError:
    temp_ids_rr = pd.to_numeric(predicted_rr_star_source_ids, errors='coerce')
    predicted_rr_star_source_ids_unique = pd.unique(temp_ids_rr[~np.isnan(temp_ids_rr)].astype(np.int64))
print(f"Identified {len(predicted_rr_star_source_ids_unique)} unique source_ids classified as 'RR*' by the model.")


# --- 3. Gaia Query Functions ---
def query_gaia_for_specific_ids_batched(source_ids_list_np_array):
    """Queries Gaia for a specific list of source_ids in batches."""
    if not source_ids_list_np_array.size: return pd.DataFrame()
    all_gaia_data_dfs = []
    num_total_ids = len(source_ids_list_np_array)
    num_batches = (num_total_ids + MAX_SOURCE_IDS_PER_MODEL_QUERY_BATCH - 1) // MAX_SOURCE_IDS_PER_MODEL_QUERY_BATCH

    for i in range(num_batches):
        start_idx = i * MAX_SOURCE_IDS_PER_MODEL_QUERY_BATCH
        end_idx = min((i + 1) * MAX_SOURCE_IDS_PER_MODEL_QUERY_BATCH, num_total_ids)
        batch_ids_np = source_ids_list_np_array[start_idx:end_idx]
        id_list_str = ",".join(batch_ids_np.astype(str))
        
        query = f"""
        SELECT source_id, parallax, parallax_error,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
        FROM {GAIA_TABLE_NAME}
        WHERE source_id IN ({id_list_str})
        """
        print(f"Querying Gaia for model stars batch {i+1}/{num_batches}...")
        try:
            job = Gaia.launch_job(query)
            results_table = job.get_results()
            if len(results_table) > 0:
                all_gaia_data_dfs.append(results_table.to_pandas())
        except Exception as e:
            print(f"Error querying Gaia for model stars batch {i+1}: {e}")
    
    if not all_gaia_data_dfs: return pd.DataFrame()
    df_gaia_queried = pd.concat(all_gaia_data_dfs, ignore_index=True)
    if 'source_id' in df_gaia_queried.columns:
        df_gaia_queried['source_id'] = df_gaia_queried['source_id'].astype(np.int64)
    return df_gaia_queried

def query_gaia_for_background_stars():
    """Queries Gaia for a general sample of background stars."""
    query = f"""
    SELECT TOP {BACKGROUND_QUERY_LIMIT}
           source_id, parallax, parallax_error,
           phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe
    FROM {GAIA_TABLE_NAME}
    WHERE parallax >= {BACKGROUND_PARALLAX_MIN}
      AND phot_g_mean_mag <= {BACKGROUND_G_MAG_MAX}
      AND ruwe < {BACKGROUND_RUWE_MAX}
      AND phot_bp_mean_mag IS NOT NULL
      AND phot_rp_mean_mag IS NOT NULL
      AND phot_g_mean_mag IS NOT NULL
      AND parallax IS NOT NULL 
    ORDER BY random_index -- Ensures a somewhat random sample if TOP is used with many results
    """
    # For some Gaia archives (like DR2, or if random_index isn't available/performant),
    # you might remove "ORDER BY random_index" or use a different strategy.
    # For DR3, random_index is good.

    print(f"\nQuerying Gaia for {BACKGROUND_QUERY_LIMIT} background stars...")
    try:
        job = Gaia.launch_job(query)
        results_table = job.get_results()
        if len(results_table) > 0:
            df_background = results_table.to_pandas()
            if 'source_id' in df_background.columns:
                 df_background['source_id'] = df_background['source_id'].astype(np.int64)
            print(f"Retrieved {len(df_background)} background stars from Gaia.")
            return df_background
        else:
            print("Gaia query for background stars returned no results.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error querying Gaia for background stars: {e}")
        return pd.DataFrame()

# --- 4. Fetch Data from Gaia ---
# Data for stars in your model input file
print("\n--- Fetching Gaia data for stars from your model input file ---")
df_model_stars_gaia = query_gaia_for_specific_ids_batched(unique_source_ids_for_model_query)
if df_model_stars_gaia.empty:
    print("Warning: Failed to retrieve Gaia data for any stars from your model input file.")
    # Depending on requirements, you might want to exit or continue with only background

# Data for the general background
print("\n--- Fetching Gaia data for dense CMD background ---")
df_background_gaia = query_gaia_for_background_stars()
if df_background_gaia.empty:
    print("Warning: Failed to retrieve Gaia data for the background. CMD will be sparse.")

# --- 5. Prepare DataFrames for CMD Plotting ---
def prepare_for_cmd(df, is_rr_star_col=False, rr_star_ids=None):
    """Calculates color, absolute magnitude, and applies filters."""
    if df.empty:
        return pd.DataFrame()
        
    # Ensure required columns exist
    required_cols = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: DataFrame is missing one or more required columns for CMD: {required_cols}")
        return pd.DataFrame()

    df_cmd = df.copy()
    df_cmd.dropna(subset=required_cols, inplace=True)
    df_cmd = df_cmd[df_cmd['parallax'] > 0] # Valid parallax for distance
    #if 'ruwe' in df_cmd.columns: # Apply RUWE cut if column exists
    #     df_cmd = df_cmd[df_cmd['ruwe'] < BACKGROUND_RUWE_MAX] # Use the same quality for consistency

    if df_cmd.empty: return pd.DataFrame()

    df_cmd['color'] = df_cmd['phot_bp_mean_mag'] - df_cmd['phot_rp_mean_mag']
    df_cmd['distance_pc'] = 1000.0 / df_cmd['parallax']
    df_cmd['abs_mag_g'] = df_cmd['phot_g_mean_mag'] - 5 * (np.log10(df_cmd['distance_pc']) - 1)
    
    if is_rr_star_col and rr_star_ids is not None and 'source_id' in df_cmd.columns:
        df_cmd['is_model_rr_star'] = df_cmd['source_id'].isin(rr_star_ids)
    
    return df_cmd

print("\n--- Preparing CMD data ---")
df_cmd_model_stars = prepare_for_cmd(df_model_stars_gaia, is_rr_star_col=True, rr_star_ids=predicted_rr_star_source_ids_unique)
df_cmd_background = prepare_for_cmd(df_background_gaia)

# --- 6. Create and Display the CMD Plot ---
print("\n--- Generating CMD Plot ---")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 10))

# A. Plot the DENSE BACKGROUND stars first
if not df_cmd_background.empty:
    plt.scatter(df_cmd_background['color'],
                df_cmd_background['abs_mag_g'],
                s=1,  # Very small for dense background
                color='silver', # Light grey or silver
                alpha=0.3,
                label=f'Nearby Stars (Gaia sample, N={len(df_cmd_background)})',
                zorder=1)
    print(f"Plotting {len(df_cmd_background)} background stars.")
else:
    print("No background stars to plot.")

# B. Plot ALL stars from your model's input (that have valid CMD data)
# These will appear over the general background. Some might be faint if not RR*.
if not df_cmd_model_stars.empty:
    # Filter out the ones that will be plotted as RR* to avoid double plotting the points
    model_stars_not_rr = df_cmd_model_stars[~df_cmd_model_stars['is_model_rr_star']]
    if not model_stars_not_rr.empty:
        plt.scatter(model_stars_not_rr['color'],
                    model_stars_not_rr['abs_mag_g'],
                    s=10, # Slightly larger than background, smaller than highlighted RR*
                    color='dimgray', # A darker grey to distinguish from background
                    alpha=0.7,
                    label=f'Other Model Input Stars (N={len(model_stars_not_rr)})',
                    zorder=2)
        print(f"Plotting {len(model_stars_not_rr)} other model input stars.")

    # C. Highlight the RR* stars classified by your model
    model_rr_stars_to_plot = df_cmd_model_stars[df_cmd_model_stars['is_model_rr_star']]
    if not model_rr_stars_to_plot.empty:
        plt.scatter(model_rr_stars_to_plot['color'],
                    model_rr_stars_to_plot['abs_mag_g'],
                    s=70,
                    color='red',
                    edgecolor='black',
                    marker='*',
                    label=f'Model Classified RR* (N={len(model_rr_stars_to_plot)})',
                    zorder=3) # Highest zorder
        print(f"Plotting {len(model_rr_stars_to_plot)} model-classified RR* stars.")
    else:
        print("No model-classified RR* stars with valid CMD data to plot.")
elif df_cmd_model_stars.empty : # If model_stars_gaia was empty to begin with
    print("No model input stars with valid CMD data to plot.")


# Set plot labels and title
plt.xlabel('Color ($G_{BP} - G_{RP}$)', fontsize=14)
plt.ylabel('Absolute G Magnitude ($M_G$)', fontsize=14)
plt.title('Color-Magnitude Diagram with Dense Background', fontsize=16)
plt.gca().invert_yaxis()

# Optional: Set plot limits (adjust based on typical CMDs or your data's range)
# plt.xlim(-0.5, 3.0)  # Typical color range
# plt.ylim(15, -5)    # Typical M_G range (inverted)

# Dynamically set limits based on the plotted data, giving some padding
all_colors = []
all_abs_mags = []
if not df_cmd_background.empty:
    all_colors.extend(df_cmd_background['color'].tolist())
    all_abs_mags.extend(df_cmd_background['abs_mag_g'].tolist())
if not df_cmd_model_stars.empty:
    all_colors.extend(df_cmd_model_stars['color'].tolist())
    all_abs_mags.extend(df_cmd_model_stars['abs_mag_g'].tolist())

if all_colors and all_abs_mags:
    color_min, color_max = np.nanpercentile(all_colors, [1, 99])
    abs_mag_min, abs_mag_max = np.nanpercentile(all_abs_mags, [1, 99]) # Min mag is brighter

    plt.xlim(color_min - 0.2, color_max + 0.2)
    plt.ylim(abs_mag_max + 1, abs_mag_min - 1) # Y-axis is inverted
else: # Fallback if no data plotted
    plt.xlim(-0.5, 3.0)
    plt.ylim(18, -6)


plt.legend(fontsize=10, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

if SAVE_FIGURE:
    plt.savefig(FIGURE_FILENAME, dpi=300)
    print(f"CMD plot saved as {FIGURE_FILENAME}")

plt.show()
print("\nScript finished.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia

# --- 0. Configuration ---
GAIA_TABLE_NAME = "gaiadr3.gaia_source"
SAVE_FIGURE = True
FIGURE_FILENAME = "cmd_model_rr_stars_with_dense_background_v2.png"
MAX_SOURCE_IDS_PER_MODEL_QUERY_BATCH = 40000

BACKGROUND_QUERY_LIMIT = 75000
BACKGROUND_PARALLAX_MIN = 1.0
BACKGROUND_G_MAG_MAX = 19.0
BACKGROUND_RUWE_MAX = 1.4 # RUWE cut for general background

# RUWE cut specifically for your RR* stars - can be made more lenient or disabled
RR_STAR_RUWE_MAX = 2.0 # Example: Allow slightly higher RUWE for RR*
# RR_STAR_RUWE_MAX = None # To disable RUWE cut for RR*

# --- 1. Load Model Predictions and Class Information ---
try:
    y_pred_full = np.load("y_predictions_ecl.npy")
except FileNotFoundError:
    print("Error: 'y_predictions_ecl.npy' not found. Please check the path.")
    exit()

try:
    classes = pd.read_pickle("Pickles/Updated_List_of_Classes_ubuntu.pkl")
    if not isinstance(classes, list):
        classes = list(classes)
except FileNotFoundError:
    print("Error: 'Pickles/Updated_List_of_Classes_ubuntu.pkl' not found. Please check the path.")
    exit()
print("Classes loaded.")

y_pred_model_classes = np.array(y_pred_full[:, :-1], dtype=int)
all_source_ids_from_model_input = y_pred_full[:, -1]
try:
    unique_source_ids_for_model_query = pd.unique(all_source_ids_from_model_input.astype(np.int64))
except ValueError:
    temp_ids = pd.to_numeric(all_source_ids_from_model_input, errors='coerce')
    unique_source_ids_for_model_query = pd.unique(temp_ids[~np.isnan(temp_ids)].astype(np.int64))
print(f"Found {len(unique_source_ids_for_model_query)} unique source_ids from model input file.")

# --- 2. Identify RR* Stars from Model Predictions ---
try:
    rr_star_class_index = classes.index("RR*")
except ValueError:
    print("Error: 'RR*' class not found. Available:", classes); exit()

rr_star_predictions_column = y_pred_model_classes[:, rr_star_class_index]
model_predicted_rr_star_mask = (rr_star_predictions_column == 1)
predicted_rr_star_source_ids_initial = all_source_ids_from_model_input[model_predicted_rr_star_mask]
try:
    # This is the set of IDs your model called RR* BEFORE any Gaia query or filtering
    unique_predicted_rr_star_ids_model = pd.unique(predicted_rr_star_source_ids_initial.astype(np.int64))
except ValueError:
    temp_ids_rr = pd.to_numeric(predicted_rr_star_source_ids_initial, errors='coerce')
    unique_predicted_rr_star_ids_model = pd.unique(temp_ids_rr[~np.isnan(temp_ids_rr)].astype(np.int64))
print(f"Model identified {len(unique_predicted_rr_star_ids_model)} unique source_ids as 'RR*'.")

# --- 3. Gaia Query Functions (same as before) ---
def query_gaia_for_specific_ids_batched(source_ids_list_np_array):
    if not source_ids_list_np_array.size: return pd.DataFrame()
    all_gaia_data_dfs = []
    num_total_ids = len(source_ids_list_np_array)
    num_batches = (num_total_ids + MAX_SOURCE_IDS_PER_MODEL_QUERY_BATCH - 1) // MAX_SOURCE_IDS_PER_MODEL_QUERY_BATCH
    for i in range(num_batches):
        start_idx = i * MAX_SOURCE_IDS_PER_MODEL_QUERY_BATCH
        end_idx = min((i + 1) * MAX_SOURCE_IDS_PER_MODEL_QUERY_BATCH, num_total_ids)
        batch_ids_np = source_ids_list_np_array[start_idx:end_idx]
        id_list_str = ",".join(batch_ids_np.astype(str))
        query = f"""
        SELECT source_id, parallax, parallax_error,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe
        FROM {GAIA_TABLE_NAME}
        WHERE source_id IN ({id_list_str})
        """
        print(f"Querying Gaia for model stars batch {i+1}/{num_batches}...")
        try:
            job = Gaia.launch_job(query); results_table = job.get_results()
            if len(results_table) > 0: all_gaia_data_dfs.append(results_table.to_pandas())
        except Exception as e: print(f"Error Gaia query batch {i+1}: {e}")
    if not all_gaia_data_dfs: return pd.DataFrame()
    df_gaia_queried = pd.concat(all_gaia_data_dfs, ignore_index=True)
    if 'source_id' in df_gaia_queried.columns:
        df_gaia_queried['source_id'] = df_gaia_queried['source_id'].astype(np.int64)
    return df_gaia_queried

def query_gaia_for_background_stars():
    query = f"""
    SELECT TOP {BACKGROUND_QUERY_LIMIT}
           source_id, parallax, parallax_error,
           phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, ruwe
    FROM {GAIA_TABLE_NAME}
    WHERE parallax >= {BACKGROUND_PARALLAX_MIN}
      AND phot_g_mean_mag <= {BACKGROUND_G_MAG_MAX}
      AND ruwe < {BACKGROUND_RUWE_MAX}
      AND phot_bp_mean_mag IS NOT NULL AND phot_rp_mean_mag IS NOT NULL
      AND phot_g_mean_mag IS NOT NULL AND parallax IS NOT NULL 
    ORDER BY random_index
    """
    print(f"\nQuerying Gaia for {BACKGROUND_QUERY_LIMIT} background stars...")
    try:
        job = Gaia.launch_job(query); results_table = job.get_results()
        if len(results_table) > 0:
            df_background = results_table.to_pandas()
            if 'source_id' in df_background.columns:
                 df_background['source_id'] = df_background['source_id'].astype(np.int64)
            print(f"Retrieved {len(df_background)} background stars from Gaia.")
            return df_background
        else: print("Background query returned no results."); return pd.DataFrame()
    except Exception as e: print(f"Error background query: {e}"); return pd.DataFrame()

# ... (Keep all the script before step 4) ...

# --- 4. Fetch Data from Gaia ---
print("\n--- Fetching Gaia data for stars from your model input file ---")
df_all_model_stars_gaia_raw = query_gaia_for_specific_ids_batched(unique_source_ids_for_model_query)

if not df_all_model_stars_gaia_raw.empty:
    # These are the RR* IDs your model identified
    model_rr_ids_set = set(unique_predicted_rr_star_ids_model)
    
    # These are the RR* IDs that were actually found in the main Gaia query
    found_rr_ids_in_main_query_set = set(df_all_model_stars_gaia_raw[
        df_all_model_stars_gaia_raw['source_id'].isin(model_rr_ids_set)
    ]['source_id'])
    
    print(f"Out of {len(model_rr_ids_set)} model-identified RR*, {len(found_rr_ids_in_main_query_set)} were found in the initial Gaia query results.")

    # Identify the RR* IDs that were NOT found
    missing_rr_ids_set = model_rr_ids_set - found_rr_ids_in_main_query_set
    print(f"{len(missing_rr_ids_set)} model-identified RR* IDs were NOT found in the initial Gaia query.")

    if missing_rr_ids_set:
        print("\n--- DEBUG: Checking a sample of MISSING RR* IDs individually in Gaia DR3 ---")
        sample_missing_ids = list(missing_rr_ids_set)[:20] # Check the first 20 missing IDs
        
        if sample_missing_ids:
            missing_ids_found_in_debug_check = []
            query_template_single = "SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE source_id = {}"
            
            for i, missing_id in enumerate(sample_missing_ids):
                print(f"  Debug Query {i+1}/{len(sample_missing_ids)} for missing ID: {missing_id}")
                try:
                    job = Gaia.launch_job(query_template_single.format(missing_id))
                    results = job.get_results()
                    if len(results) > 0:
                        print(f"    SUCCESS: Missing ID {missing_id} WAS FOUND in gaiadr3.gaia_source during debug check.")
                        missing_ids_found_in_debug_check.append(missing_id)
                    else:
                        print(f"    FAILURE: Missing ID {missing_id} was NOT FOUND in gaiadr3.gaia_source during debug check.")
                except Exception as e:
                    print(f"    ERROR during debug query for {missing_id}: {e}")
            
            print(f"Debug Check Summary: Out of {len(sample_missing_ids)} sampled missing RR* IDs, {len(missing_ids_found_in_debug_check)} were found by individual queries.")
            if len(missing_ids_found_in_debug_check) == len(sample_missing_ids) and len(sample_missing_ids) > 0:
                print("    This is unexpected! If individual queries find them, there might be an issue with how the large batch query is handled by Gaia TAP or a very subtle data type/formatting issue for those specific IDs in the batch.")
            elif len(missing_ids_found_in_debug_check) < len(sample_missing_ids) and len(missing_ids_found_in_debug_check) > 0:
                 print("    Some were found individually but not in the batch - this is strange and points to potential TAP service inconsistencies for batch vs single ID queries for these specific IDs.")
            elif not missing_ids_found_in_debug_check:
                 print("    None of the sampled missing IDs were found individually. This strongly suggests these IDs are not in gaiadr3.gaia_source under these identifiers or have no photometry.")

else:
    print("Warning: Gaia query for model input stars returned no results. Cannot perform RR* ID check.")
    # rr_stars_in_gaia_query_results = pd.DataFrame() # Ensure it exists - this line is not strictly needed here with new logic

# ... (Rest of the script: Fetching background, prepare_for_cmd, plotting) ...

# --- 5. Prepare DataFrames for CMD Plotting ---
def prepare_for_cmd(df_input, df_name_for_log, initial_rr_ids_set=None, is_rr_candidate_df=False, ruwe_cut_val=None):
    """Calculates color, absolute magnitude, and applies filters. Logs losses for RR* candidates."""
    if df_input.empty:
        print(f"Log ({df_name_for_log}): Input DataFrame is empty.")
        return pd.DataFrame()
    
    df = df_input.copy()
    original_count = len(df)
    rr_ids_in_df = set()
    if initial_rr_ids_set and 'source_id' in df.columns:
        rr_ids_in_df = initial_rr_ids_set.intersection(set(df['source_id']))
    
    if is_rr_candidate_df:
        print(f"\n--- Processing RR* Candidates from '{df_name_for_log}' (Initial count: {original_count}, containing {len(rr_ids_in_df)} of model's RR* IDs) ---")

    required_cols = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'parallax']
    if not all(col in df.columns for col in required_cols):
        print(f"Log ({df_name_for_log}): Missing one or more required CMD columns: {required_cols}")
        return pd.DataFrame()

    # Step 1: Drop rows with NaN in essential photometric/parallax columns
    df.dropna(subset=required_cols, inplace=True)
    count_after_dropna = len(df)
    if is_rr_candidate_df:
        rr_ids_after_dropna = initial_rr_ids_set.intersection(set(df['source_id']))
        print(f"  After dropping NaNs in essential mags/parallax: {count_after_dropna} rows remain (RR* IDs: {len(rr_ids_after_dropna)}, lost {len(rr_ids_in_df) - len(rr_ids_after_dropna)})")
        rr_ids_in_df = rr_ids_after_dropna # Update current set of RR* IDs

    # Step 2: Filter for valid parallax values
    df = df[df['parallax'] > 0]
    count_after_pos_parallax = len(df)
    if is_rr_candidate_df:
        rr_ids_after_pos_parallax = initial_rr_ids_set.intersection(set(df['source_id']))
        print(f"  After requiring parallax > 0: {count_after_pos_parallax} rows remain (RR* IDs: {len(rr_ids_after_pos_parallax)}, lost {len(rr_ids_in_df) - len(rr_ids_after_pos_parallax)})")
        rr_ids_in_df = rr_ids_after_pos_parallax

    # Step 3: Apply RUWE cut (if specified)
    if ruwe_cut_val is not None and 'ruwe' in df.columns:
        df = df[df['ruwe'] < ruwe_cut_val]
        count_after_ruwe = len(df)
        if is_rr_candidate_df:
            rr_ids_after_ruwe = initial_rr_ids_set.intersection(set(df['source_id']))
            print(f"  After RUWE < {ruwe_cut_val}: {count_after_ruwe} rows remain (RR* IDs: {len(rr_ids_after_ruwe)}, lost {len(rr_ids_in_df) - len(rr_ids_after_ruwe)})")
            rr_ids_in_df = rr_ids_after_ruwe
    elif is_rr_candidate_df and ruwe_cut_val is not None :
         print(f"  RUWE column not found, skipping RUWE cut for '{df_name_for_log}'.")


    if df.empty:
        print(f"Log ({df_name_for_log}): DataFrame became empty after filtering.")
        return pd.DataFrame()

    df['color'] = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']
    df['distance_pc'] = 1000.0 / df['parallax']
    df['abs_mag_g'] = df['phot_g_mean_mag'] - 5 * (np.log10(df['distance_pc']) - 1)
    
    # Final check for NaNs in calculated color/abs_mag (should be rare if inputs are fine)
    df.dropna(subset=['color', 'abs_mag_g'], inplace=True)
    count_after_cmd_calc = len(df)
    if is_rr_candidate_df:
        rr_ids_after_cmd_calc = initial_rr_ids_set.intersection(set(df['source_id']))
        print(f"  After CMD calculations & final NaN drop: {count_after_cmd_calc} rows remain (RR* IDs: {len(rr_ids_after_cmd_calc)}, lost {len(rr_ids_in_df) - len(rr_ids_after_cmd_calc)})")
        print(f"  Total RR* IDs surviving for CMD plotting from '{df_name_for_log}': {len(rr_ids_after_cmd_calc)}")

    # Add the 'is_model_rr_star' flag for the model stars DataFrame
    if is_rr_candidate_df and initial_rr_ids_set and 'source_id' in df.columns:
        df['is_model_rr_star'] = df['source_id'].isin(initial_rr_ids_set)
    
    return df

print("\n--- Preparing CMD data ---")
# Prepare the DataFrame that contains ALL stars from your model input (including RR* candidates)
# The 'is_rr_candidate_df=True' will trigger detailed logging for the RR* subset within this df.
df_cmd_all_model_stars = prepare_for_cmd(
    df_all_model_stars_gaia_raw,
    df_name_for_log="All Model Input Stars (includes RR*)",
    initial_rr_ids_set=set(unique_predicted_rr_star_ids_model), # Pass the original 499 RR* IDs
    is_rr_candidate_df=True, # This flag is to ensure detailed logging for RR* losses
    ruwe_cut_val=RR_STAR_RUWE_MAX # Use the specific RUWE cut for RR*
)

df_cmd_background = prepare_for_cmd(
    df_background_gaia_raw,
    df_name_for_log="Background Gaia Sample",
    ruwe_cut_val=BACKGROUND_RUWE_MAX # Use the general background RUWE cut
)

# --- 6. Create and Display the CMD Plot ---
print("\n--- Generating CMD Plot ---")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 10))

# A. Plot the DENSE BACKGROUND stars first
if not df_cmd_background.empty:
    plt.scatter(df_cmd_background['color'], df_cmd_background['abs_mag_g'],
                s=1, color='silver', alpha=0.3,
                label=f'Nearby Stars (Gaia sample, N={len(df_cmd_background)})', zorder=1)
    print(f"Plotting {len(df_cmd_background)} background stars.")
else:
    print("No background stars to plot.")

# B. Highlight the RR* stars classified by your model (NO "Other Model Input Stars")
if not df_cmd_all_model_stars.empty and 'is_model_rr_star' in df_cmd_all_model_stars.columns:
    model_rr_stars_to_plot = df_cmd_all_model_stars[df_cmd_all_model_stars['is_model_rr_star']] # Filter based on the flag
    if not model_rr_stars_to_plot.empty:
        plt.scatter(model_rr_stars_to_plot['color'], model_rr_stars_to_plot['abs_mag_g'],
                    s=70, color='red', edgecolor='black', marker='*',
                    label=f'Model Classified RR* (N={len(model_rr_stars_to_plot)})', zorder=3)
        print(f"Plotting {len(model_rr_stars_to_plot)} model-classified RR* stars.")
    else:
        print("No model-classified RR* stars with valid CMD data to plot after all filters.")
else:
    print("No model input stars (including RR*) with valid CMD data to plot, or 'is_model_rr_star' column missing.")

plt.xlabel('Color ($G_{BP} - G_{RP}$)', fontsize=14)
plt.ylabel('Absolute G Magnitude ($M_G$)', fontsize=14)
plt.title('Color-Magnitude Diagram: Model RR Lyrae with Dense Background', fontsize=16)
plt.gca().invert_yaxis()

all_colors, all_abs_mags = [], []
if not df_cmd_background.empty:
    all_colors.extend(df_cmd_background['color'].tolist())
    all_abs_mags.extend(df_cmd_background['abs_mag_g'].tolist())
# Include RR* in limits calculation if they exist
if not df_cmd_all_model_stars.empty and 'is_model_rr_star' in df_cmd_all_model_stars.columns:
    model_rr_stars_to_plot_for_limits = df_cmd_all_model_stars[df_cmd_all_model_stars['is_model_rr_star']]
    if not model_rr_stars_to_plot_for_limits.empty:
        all_colors.extend(model_rr_stars_to_plot_for_limits['color'].tolist())
        all_abs_mags.extend(model_rr_stars_to_plot_for_limits['abs_mag_g'].tolist())

if all_colors and all_abs_mags:
    color_min, color_max = np.nanpercentile(all_colors, [1, 99])
    abs_mag_min, abs_mag_max = np.nanpercentile(all_abs_mags, [1, 99])
    plt.xlim(color_min - 0.2, color_max + 0.2)
    plt.ylim(abs_mag_max + 1, abs_mag_min - 1)
else:
    plt.xlim(-0.5, 3.0); plt.ylim(18, -6)

plt.legend(fontsize=10, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tight_layout()

if SAVE_FIGURE:
    plt.savefig(FIGURE_FILENAME, dpi=300)
    print(f"CMD plot saved as {FIGURE_FILENAME}")
plt.show()
print("\nScript finished.")