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