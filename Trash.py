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