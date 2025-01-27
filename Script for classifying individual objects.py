#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import requests
import pyvo as vo
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from astropy.io import fits
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

#--------------------------------------------------------------------------------
# 1) LOAD ALL NECESSARY PRE-FITTED TRANSFORMERS & MODEL
#--------------------------------------------------------------------------------
with open("transforms/gaia_transformers.pkl", "rb") as f:
    gaia_transformers = pickle.load(f)   # Dict of {col_name: fitted PowerTransformer}

#with open("transforms/lamost_spectra_pipeline.pkl", "rb") as f:
   # lamost_spectra_pipeline = pickle.load(f)  # scikit-learn pipeline or similar

#with open("transforms/final_label_columns.pkl", "rb") as f:
  #  final_label_cols = pickle.load(f)  # list of label names (strings)
final_label_cols = pd.read_pickle("Pickles/Updated_list_of_Classes.pkl")

#with open("models/final_multilabel_classifier.pkl", "rb") as f:
    #multi_label_clf = pickle.load(f)  # the trained multi-label classifier


#--------------------------------------------------------------------------------
# 2) HELPER FUNCTIONS
#--------------------------------------------------------------------------------
def split_ids_into_chunks(gaia_id_list, chunk_size=50000):
    """
    Takes a Python list of Gaia IDs (strings or ints),
    returns a list of comma-joined strings, each containing up to `chunk_size` IDs.
    """
    # Convert everything to string for the SQL query
    gaia_id_list = [str(x) for x in gaia_id_list]
    chunks = []
    for i in range(0, len(gaia_id_list), chunk_size):
        chunk = ", ".join(gaia_id_list[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


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
    missing_ids = set(gaia_id_list) - set(all_ids)
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs not found in Gaia DR3.")
        print(f"Missing IDs: {missing_ids}")

    if not all_dfs:
        return pd.DataFrame(columns=desired_cols)
    else:
        return pd.concat(all_dfs, ignore_index=True)


def crossmatch_lamost(gaia_df, lamost_catalog_path, match_radius=1*u.arcsec):
    """
    Cross-match the Gaia DataFrame with a local LAMOST catalogue CSV
    (which must have 'ra' and 'dec' columns).
    Returns a merged DataFrame containing only matched objects, plus LAMOST obsid, etc.
    """
    lamost_df = pd.read_csv(lamost_catalog_path)
    # Basic cleaning
    lamost_df = lamost_df.dropna(subset=['ra','dec'])
    gaia_df = gaia_df.dropna(subset=['ra','dec'])

    # Create astropy SkyCoord objects
    gaia_coords   = SkyCoord(ra=gaia_df['ra'].values*u.deg,
                             dec=gaia_df['dec'].values*u.deg)
    lamost_coords = SkyCoord(ra=lamost_df['ra'].values*u.deg,
                             dec=lamost_df['dec'].values*u.deg)

    # Match to catalog
    idx, sep2d, _ = gaia_coords.match_to_catalog_sky(lamost_coords)
    matches = sep2d < match_radius

    # Subset
    gaia_matched   = gaia_df.iloc[matches].copy().reset_index(drop=True)
    lamost_matched = lamost_df.iloc[idx[matches]].copy().reset_index(drop=True)

    # Merge into single DataFrame
    final = pd.concat([gaia_matched, lamost_matched], axis=1)
    return final
def crossmatch_lamost(gaia_df, lamost_catalog_path, match_radius=3*u.arcsec):  # Increase radius to 3 arcsec
    """
    Cross-match the Gaia DataFrame with a local LAMOST catalogue CSV
    (which must have 'ra' and 'dec' columns).
    Returns a merged DataFrame containing only matched objects, plus LAMOST obsid, etc.
    """
    lamost_df = pd.read_csv(lamost_catalog_path)
    
    # Convert to numeric and remove NaNs
    lamost_df['ra'] = pd.to_numeric(lamost_df['ra'], errors='coerce')
    lamost_df['dec'] = pd.to_numeric(lamost_df['dec'], errors='coerce')
    gaia_df['ra'] = pd.to_numeric(gaia_df['ra'], errors='coerce')
    gaia_df['dec'] = pd.to_numeric(gaia_df['dec'], errors='coerce')

    lamost_df = lamost_df.dropna(subset=['ra', 'dec'])
    gaia_df = gaia_df.dropna(subset=['ra', 'dec'])

    print(f"After NaN removal: Gaia={gaia_df.shape}, LAMOST={lamost_df.shape}")

    # Create SkyCoord objects
    gaia_coords = SkyCoord(ra=gaia_df['ra'].values*u.deg, dec=gaia_df['dec'].values*u.deg)
    lamost_coords = SkyCoord(ra=lamost_df['ra'].values*u.deg, dec=lamost_df['dec'].values*u.deg)

    # Perform crossmatch
    idx, sep2d, _ = gaia_coords.match_to_catalog_sky(lamost_coords)
    matches = sep2d < match_radius

    # Debugging: Print some match distances
    print(f"Match distances (arcsec): {sep2d.to(u.arcsec)[:10]}")

    # Subset
    gaia_matched = gaia_df.iloc[matches].copy().reset_index(drop=True)
    lamost_matched = lamost_df.iloc[idx[matches]].copy().reset_index(drop=True)

    print(f"Matched Gaia Objects: {gaia_matched.shape}")
    print(f"Matched LAMOST Objects: {lamost_matched.shape}")

    if gaia_matched.empty:
        print("⚠️ No matches found! Try increasing `match_radius`.")

    # Merge into single DataFrame
    final = pd.concat([gaia_matched, lamost_matched], axis=1)
    return final

def crossmatch_lamost(gaia_df, lamost_catalog_path, match_radius=3*u.arcsec):
    """
    Cross-matches Gaia sources with a local LAMOST catalogue.
    Returns a merged DataFrame of matched objects.
    """

    # Load LAMOST catalog
    lamost_df = pd.read_csv(lamost_catalog_path)

    # Ensure RA/Dec are numeric
    gaia_df['ra'] = pd.to_numeric(gaia_df['ra'], errors='coerce')
    gaia_df['dec'] = pd.to_numeric(gaia_df['dec'], errors='coerce')
    lamost_df['ra'] = pd.to_numeric(lamost_df['ra'], errors='coerce')
    lamost_df['dec'] = pd.to_numeric(lamost_df['dec'], errors='coerce')

    # Drop NaN values
    gaia_df = gaia_df.dropna(subset=['ra', 'dec'])
    lamost_df = lamost_df.dropna(subset=['ra', 'dec'])

    print(f"After NaN removal: Gaia={gaia_df.shape}, LAMOST={lamost_df.shape}")

    # Check if LAMOST coordinates are in arcseconds (convert if necessary)
    if lamost_df['ra'].max() > 360:  # RA should not exceed 360 degrees
        print("⚠️ LAMOST RA/Dec seem to be in arcseconds. Converting to degrees.")
        lamost_df['ra'] /= 3600
        lamost_df['dec'] /= 3600

    # Convert to SkyCoord objects (ensuring same frame)
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
    print(f"Match distances (arcsec): {d2d.to(u.arcsec).value[matches]}")

    if matches.sum() == 0:
        print("⚠️ No matches found! Try increasing `match_radius`.")
        return pd.DataFrame()

    # Extract matched rows correctly
    gaia_matched = gaia_df.iloc[matches].copy().reset_index(drop=True)
    lamost_matched = lamost_df.iloc[idx[matches]].copy().reset_index(drop=True)

    print(f"Matched Gaia Objects: {gaia_matched.shape}")
    print(f"Matched LAMOST Objects: {lamost_matched.shape}")

    # Merge matches into final DataFrame
    final = pd.concat([gaia_matched, lamost_matched], axis=1)

    return final


def download_lamost_spectra(obsid_list, save_folder="star_spectra"):
    """
    Example function to download LAMOST spectra by obsid.
    Adapt or skip if you already have the spectra or prefer a different approach.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Simple retry session
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429,500,502,503,504])
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=retries))

    for obsid in tqdm(obsid_list, desc="Downloading LAMOST spectra"):
        url = f"https://www.lamost.org/dr7/v2.0/spectrum/fits/{obsid}"
        local_path = os.path.join(save_folder, str(obsid))
        if os.path.exists(local_path):
            continue  # skip if already downloaded
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            print(f"Failed to download obsid={obsid} => {e}")


def process_lamost_spectra(folder_path, lamost_pipeline):
    """
    Use your pre-fitted scikit-learn pipeline (or set of functions) to:
      - read FITS spectra
      - do your row-slicing, interpolation, min-max / power transforms
      - return a DataFrame of final spectral features (one row per spectrum)
    """
    # This function is just conceptual. 
    # In your original code, you built a CSV with flux/freq, 
    # interpolated, normalized, then saved to a pickle, etc.
    # 
    # Here, you'd replicate EXACTLY those steps, but encapsulated so
    # you can re-use them for new obsids. 
    # E.g.:
    #
    #    1) read each FITS file
    #    2) extract flux row
    #    3) pass it to lamost_pipeline.transform(...) or do the same logic
    #    4) assemble into a single DataFrame of shape [N_spectra x M_features]
    #
    # For brevity, we return an empty DataFrame here, but you should
    # replicate your earlier pipeline exactly.
    #
    df_final_spectra = []
    # ...
    # for each file in folder_path:
    #    flux_array = ...
    #    processed  = lamost_pipeline.transform(flux_array)  # e.g. shape(1, M)
    #    store in list/dict
    #
    # df_final_spectra = pd.DataFrame(...)
    #
    return pd.DataFrame(df_final_spectra)

def process_lamost_spectra(folder_path, lamost_pipeline):
    """
    Reads LAMOST FITS spectra, applies interpolation, normalization, and transformation.
    Returns a DataFrame of final spectral features (one row per spectrum).
    """

    processed_spectra = []
    failed_files = []

    # Loop through each FITS file in the folder
    for filename in tqdm(os.listdir(folder_path), desc="Processing LAMOST Spectra"):
        file_path = os.path.join(folder_path, filename)

        try:
            with fits.open(file_path) as hdul:
                # Extract the first 3 rows and first 3748 columns
                data = hdul[0].data[:3, :3748]  

                # Extract flux (first row) and frequency (third row)
                flux = data[0]
                freq = data[2]

                # Handle NaN and zero values
                valid_mask = ~np.isnan(freq) & ~np.isnan(flux) & (freq != 0)
                if valid_mask.sum() < 10:  # Skip if too few valid points
                    failed_files.append(filename)
                    continue

                # Interpolation to handle missing values
                interp_func = interp1d(freq[valid_mask], flux[valid_mask], kind="linear", fill_value="extrapolate")
                new_frequencies = np.linspace(freq[valid_mask].min(), freq[valid_mask].max(), len(flux))
                interpolated_flux = interp_func(new_frequencies)

                # Trim first 100 columns
                interpolated_flux = interpolated_flux[100:]

                # Apply MinMaxScaler
                min_max_scaler = MinMaxScaler()
                normalized_flux = min_max_scaler.fit_transform(interpolated_flux.reshape(1, -1)).flatten()

                # Apply Yeo-Johnson transformation
                power_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
                transformed_flux = power_transformer.fit_transform(normalized_flux.reshape(1, -1)).flatten()

                # Store the processed data
                spectrum_data = {f"flux_{i}": value for i, value in enumerate(transformed_flux)}
                spectrum_data["obsid"] = int(filename.split(".")[0])  # Extract obsid from filename
                processed_spectra.append(spectrum_data)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_files.append(filename)

    # Convert the processed spectra into a DataFrame
    df_spectra = pd.DataFrame(processed_spectra)

    return df_spectra, failed_files



def apply_gaia_transforms(gaia_df, transformers_dict):
    """
    Applies the same Yeo-Johnson (or other) transformations used in training
    to the relevant Gaia columns. 
    """
    # Fill the same NaN values or set the same flags as in training
    # e.g. if you flagged parallax=NaN => set parallax=0, error=10
    # do that here too, to keep consistent with your training pipeline
    #
    # Example based on your code:
    gaia_df['flagnopllx'] = np.where(gaia_df['parallax'].isna(), 1, 0)
    gaia_df['parallax']       = gaia_df['parallax'].fillna(0)
    gaia_df['parallax_error'] = gaia_df['parallax_error'].fillna(10)
    gaia_df['pmra']           = gaia_df['pmra'].fillna(0)
    gaia_df['pmra_error']     = gaia_df['pmra_error'].fillna(10)
    gaia_df['pmdec']          = gaia_df['pmdec'].fillna(0)
    gaia_df['pmdec_error']    = gaia_df['pmdec_error'].fillna(10)

    gaia_df['flagnoflux'] = 0
    # If G or BP or RP is missing
    missing_flux = gaia_df['phot_g_mean_flux'].isna() | gaia_df['phot_bp_mean_flux'].isna() 
    gaia_df.loc[missing_flux, 'flagnoflux'] = 1

    # fill flux with 0 and error with large number
    gaia_df['phot_g_mean_flux']       = gaia_df['phot_g_mean_flux'].fillna(0)
    gaia_df['phot_g_mean_flux_error'] = gaia_df['phot_g_mean_flux_error'].fillna(50000)
    gaia_df['phot_bp_mean_flux']      = gaia_df['phot_bp_mean_flux'].fillna(0)
    gaia_df['phot_bp_mean_flux_error']= gaia_df['phot_bp_mean_flux_error'].fillna(50000)
    gaia_df['phot_rp_mean_flux']      = gaia_df['phot_rp_mean_flux'].fillna(0)
    gaia_df['phot_rp_mean_flux_error']= gaia_df['phot_rp_mean_flux_error'].fillna(50000)

    # Drop any rows that are incomplete, if that was your final approach:
    gaia_df.dropna(axis=0, inplace=True)

    # Now apply the stored transformations:
    for col, transformer in transformers_dict.items():
        if col in gaia_df.columns:
            gaia_df[col] = transformer.transform(gaia_df[[col]])
        else:
            # If the column didn't exist, maybe set to 0 or skip?
            print(f"Warning: column {col} not found in new data, skipping transform.")

    return gaia_df


#--------------------------------------------------------------------------------
# 3) MAIN INFERENCE FUNCTION
#--------------------------------------------------------------------------------
def run_inference(gaia_id_list,
                  lamost_catalog_path="lamost/dr9_v2.0_LRS_catalogue.csv",
                  spectra_folder="star_spectra"):
    """
    Given a list of Gaia DR3 IDs:
      1) Query the GAIA data
      2) Cross-match with LAMOST
      3) Download LAMOST spectra (if desired)
      4) Process the spectra with the stored pipeline
      5) Merge the Gaia + spectral features
      6) Apply final transformations
      7) Predict multi-label membership with the loaded classifier
      8) Return the predicted multi-hot + confidence

    Modify as needed for your actual pipeline steps.
    """

    #---------------------------------------------
    # A) Query Gaia
    #---------------------------------------------
    print("Querying Gaia for your object list...")
    df_gaia = query_gaia_data(gaia_id_list)
    if df_gaia.empty:
        print("No Gaia data returned. Exiting.")
        return pd.DataFrame()

    #---------------------------------------------
    # B) Cross-match with LAMOST
    #---------------------------------------------
    print("Cross-matching with LAMOST catalogue...")
    df_matched = crossmatch_lamost(df_gaia, lamost_catalog_path=lamost_catalog_path)
    if df_matched.empty:
        print("No LAMOST matches found. Exiting.")
        return pd.DataFrame()

    #---------------------------------------------
    # C) Download LAMOST spectra for matched obsids
    #    (Optional if you haven't already cached them.)
    #---------------------------------------------
    print("Downloading LAMOST spectra for matched objects if not already present...")
    obsids = df_matched["obsid"].unique()
    download_lamost_spectra(obsids, save_folder=spectra_folder)

    #---------------------------------------------
    # D) Process the LAMOST spectra => final spectral features
    #---------------------------------------------
    print("Processing LAMOST spectra into final feature set...")
    df_spectra = process_lamost_spectra(spectra_folder, lamost_spectra_pipeline)
    # `df_spectra` should have columns [obsid, spec_feat_1, spec_feat_2, ...]

    # Merge these spectral features back with matched Gaia rows
    # by 'obsid' (or whichever key you use).
    df_all = pd.merge(df_matched, df_spectra, on="obsid", how="inner")

    #---------------------------------------------
    # E) Apply Gaia transformations
    #---------------------------------------------
    print("Applying stored Gaia transformations...")
    df_final = apply_gaia_transforms(df_all, gaia_transformers)

    #---------------------------------------------
    # F) Generate Predictions from Multi-Label Classifier
    #---------------------------------------------
    #  1) Extract just the columns your model expects as `X_infer`.
    #     This must match the training feature order exactly.
    #     (You might load the “train_data_transformed.pkl” columns 
    #      to see the final column order used in training.)
    #
    #     For example, let’s suppose:
    #     train_cols = [...]
    #
    #     Then do:
    #          X_infer = df_final[train_cols].values
    #
    #  2) Call the classifier’s predict_proba(...) or decision_function(...).
    #     Because it’s a multi-label classifier with a sigmoid output,
    #     you typically get probabilities for each label.
    #
    #  3) Combine results into a DataFrame for convenience.
    #

    # Example (adjust column order to match your training):
    # Suppose these columns (not including the label columns themselves):
    example_train_cols = [
        "parallax","pmra","pmdec","phot_g_mean_flux","phot_bp_mean_flux",
        # ... all other columns your model expects ...
        "spec_feat_1","spec_feat_2","spec_feat_3",  # from LAMOST pipeline
        # ...
    ]
    # Make sure they exist in df_final:
    X_infer = df_final[example_train_cols].values

    print("Predicting multi-label probabilities...")
    y_proba = multi_label_clf.predict_proba(X_infer)  # shape: (N, number_of_labels)

    # Combine into a nice DataFrame:
    df_results = pd.DataFrame(y_proba, columns=final_label_cols)
    # Optional thresholding if you want binary predictions:
    # e.g. 
    # df_pred = (df_results >= 0.5).astype(int)

    #---------------------------------------------
    # G) Attach object IDs for clarity
    #---------------------------------------------
    df_results["source_id"] = df_final["source_id"].values
    df_results["obsid"]     = df_final["obsid"].values

    return df_results


#--------------------------------------------------------------------------------
# 4) IF RUN AS A SCRIPT
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage: 
    # Provide some list of Gaia DR3 IDs you want to classify
    example_gaia_ids = [
        4146599265675697536,  # example ID 1
        4113643650790812032,  # example ID 2
        464735664956664320,  # example ID 3
        5520830539439082880, # example ID 4
        
        # ...
    ]
    from astroquery.simbad import Simbad

    # Define the object type
    object_type = "CV*"

    # Get all sources of this type
    Simbad.TIMEOUT = 500  # Increase timeout for large queries
    Simbad.add_votable_fields("ra", "dec", "main_id", "otype")

    query_result = Simbad.query_criteria(otype=object_type)

    # Extract Gaia IDs if available
    gaia_ids = [x for x in query_result["main_id"] if "Gaia DR3" in x]

    print(f"Found {len(gaia_ids)} magnetars in Simbad with Gaia IDs.")

    # Run inference
    results_df = run_inference(gaia_ids)

    # Print the results
    print("Inference Completed!")
    print(results_df.head(10))
    #
    # Each row has the multi-label probabilities in columns named
    # after your final_label_cols, plus the "source_id"/"obsid".
