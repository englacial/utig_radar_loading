"""
GPS file generation for OPR (Open Polar Radar) compatible format.

This module provides functions to generate MATLAB-compatible GPS files
from UTIG raw GPS data streams, matching the format used by CReSIS 
gps_create functions.
"""

import numpy as np
import pandas as pd
import scipy.io
import h5py
import hdf5storage
from pathlib import Path
from datetime import datetime
import warnings
from typing import List, Optional, Union, Dict, Any
from . import stream_util


def load_and_parse_gps_file(gps_path: Union[str, Path], use_ct: bool = True) -> pd.DataFrame:
    """
    Load a single GPS file and parse it to standard format.
    
    Parameters:
    -----------
    gps_path : str or Path
        Path to GPS data file (e.g., GPSnc1/xds.gz)
    use_ct : bool
        Whether to use CT time if available (default: True)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with parsed GPS data including LAT, LON, TIMESTAMP
    """
    gps_path = Path(gps_path)
    
    # Load the GPS data using stream_util
    try:
        df = stream_util.load_xds_stream_file(
            gps_path, 
            debug=False, 
            parse=True, 
            parse_kwargs={'use_ct': use_ct}
        )
    except Exception as e:
        warnings.warn(f"Failed to load {gps_path}: {e}")
        return pd.DataFrame()
    
    # Ensure we have the required columns
    required_cols = ['LAT', 'LON', 'TIMESTAMP']
    if not all(col in df.columns for col in required_cols):
        warnings.warn(f"Missing required columns in {gps_path}")
        return pd.DataFrame()
    
    # Add source file information
    df['source_file'] = str(gps_path)
    
    # Convert TIMESTAMP to Unix epoch if it's a datetime
    df['unix_time'] = ((df['TIMESTAMP'] - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')).values
    df['comp_time'] = df['unix_time'] # For consistency with header files
    df['radar_time'] = df['tim']

    return df


def merge_gps_files(gps_paths: List[Union[str, Path]], use_ct: bool = True) -> pd.DataFrame:
    """
    Load and merge multiple GPS files, sorted by time.
    
    Parameters:
    -----------
    gps_paths : list of str or Path
        List of paths to GPS data files
    use_ct : bool
        Whether to use CT time if available (default: True)
        
    Returns:
    --------
    pd.DataFrame
        Merged and sorted GPS data
    """
    all_dfs = []
    
    for gps_path in gps_paths:
        df = load_and_parse_gps_file(gps_path, use_ct=use_ct)
        if not df.empty:
            # Get first timestamp for sorting files
            first_time = df['unix_time'].iloc[0] if len(df) > 0 else float('inf')
            all_dfs.append((first_time, df))
    
    if not all_dfs:
        raise ValueError("No valid GPS data loaded from provided files")
    
    # Sort by first timestamp of each file
    all_dfs.sort(key=lambda x: x[0])
    
    # Concatenate all dataframes
    merged_df = pd.concat([df for _, df in all_dfs], ignore_index=True)
    
    # Sort by timestamp (in case files have overlapping times)
    merged_df = merged_df.sort_values('unix_time').reset_index(drop=True)

    unix_time_diff = np.diff(merged_df['unix_time'])
    if np.any(unix_time_diff <= 0):
        merged_df = merged_df.drop_duplicates(subset='unix_time', keep='first').reset_index(drop=True)
        warnings.warn("Duplicate or non-increasing unix_time entries found and removed")
    
    return merged_df

def load_and_parse_imu_file(imu_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single IMU file and parse it to standard format.
    
    Parameters:
    -----------
    imu_path : str or Path
        Path to IMU data file (e.g., IMUnc1/xds.gz)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with parsed IMU data including ROLL, PITCH, HEADING, and either TIMESTAMP or gps_time
    """
    imu_path = Path(imu_path)
    
    stream_type = imu_path.parent.name  # e.g., 'AVNnp1

    if stream_type == 'AVNnp1':
        imu_df = stream_util.parse_binary_AVNnp1(imu_path)
    else:
        warnings.warn(f"Unknown IMU stream type '{stream_type}' in {imu_path}")
        return pd.DataFrame()

    # Ensure we have the required columns
    required_cols = ['ROLL', 'PITCH', 'HEADING']
    if not all(col in imu_df.columns for col in required_cols):
        warnings.warn(f"Missing required columns in {imu_path}")
        return pd.DataFrame()

    if 'TIMESTAMP' not in imu_df.columns and 'GPS_TIME' not in imu_df.columns:
        warnings.warn(f"Missing TIMESTAMP or GPS_TIME in {imu_path}")
        return pd.DataFrame()
    
    # Add source file information
    imu_df['source_file'] = str(imu_path)

    return imu_df

def merge_imu_files(imu_paths: List[Union[str, Path]], time_sort_key='GPS_TIME') -> pd.DataFrame:
    """
    Load and merge multiple IMU files, sorted by specified time key.
    
    Parameters:
    -----------
    imu_paths : list of str or Path
        List of paths to IMU data files
    time_sort_key : str
        Column name to sort by ('GPS_TIME' or 'TIMESTAMP')
        
    Returns:
    --------
    pd.DataFrame
        Merged and sorted IMU data
    """
    all_dfs = []
    
    for imu_path in imu_paths:
        df = load_and_parse_imu_file(imu_path)
        if not df.empty:
            # Get first timestamp for sorting files
            if time_sort_key in df.columns:
                first_time = df[time_sort_key].iloc[0]
            else:
                first_time = float('inf')
            all_dfs.append((first_time, df))
    
    if not all_dfs:
        raise ValueError("No valid IMU data loaded from provided files")
    
    # Sort by first timestamp of each file
    all_dfs.sort(key=lambda x: x[0])
    
    # Concatenate all dataframes
    merged_df = pd.concat([df for _, df in all_dfs], ignore_index=True)
    
    # Sort by specified time key (in case files have overlapping times)
    if time_sort_key in merged_df.columns:
        merged_df = merged_df.sort_values(time_sort_key).reset_index(drop=True)
    else:
        warnings.warn(f"Time sort key '{time_sort_key}' not found in IMU data")

    return merged_df



def create_gps_matlab_structure(df: pd.DataFrame, source:str = "UTIG_GPSnc1-field") -> Dict[str, Any]:
    """
    Create a dictionary structure matching MATLAB GPS format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged GPS dataframe with LAT, LON, unix_time columns
        
    Returns:
    --------
    dict
        Dictionary with GPS data in MATLAB-compatible format
    """

    # Extract elevation if available, otherwise use zeros
    if 'ELEV' in df.columns:
        elev = df['ELEV'].values
    elif 'vert_cor' in df.columns:
        elev = df['vert_cor'].values
    else:
        elev = np.zeros(len(df))
    
    # Extract roll, pitch, heading if available
    if 'ROLL' in df.columns:
        roll = df['ROLL'].values
    else:
        roll = np.zeros(len(df))
    
    if 'PITCH' in df.columns:
        pitch = df['PITCH'].values
    else:
        pitch = np.zeros(len(df))
        
    if 'HEADING' in df.columns:
        heading = df['HEADING'].values  
    else:
        heading = np.zeros(len(df))
    
    # Create the GPS structure matching MATLAB format
    gps_struct = {
        'gps_time': df['unix_time'].values.astype(np.float64),  # Unix epoch seconds
        'comp_time': df['comp_time'].values.astype(np.float64),  # Same as gps_time -- for compatibility with header files
        'radar_time': df['radar_time'].values.astype(np.float64), # 10 us ticks
        'lat': df['LAT'].values.astype(np.float64),  # Latitude in degrees
        'lon': df['LON'].values.astype(np.float64),  # Longitude in degrees
        'elev': elev.astype(np.float64),  # Elevation in meters
        'roll': roll.astype(np.float64),  # Roll in radians (or zeros)
        'pitch': pitch.astype(np.float64),  # Pitch in radians (or zeros)
        'heading': heading.astype(np.float64),  # Heading in radians (or zeros)
        'gps_source': source,  # GPS source identifier
        'file_type': 'gps'  # File type identifier
    }
    
    # Ensure all arrays are 1D and same length
    n_records = len(df)
    for key in ['gps_time', 'comp_time', 'radar_time', 'lat', 'lon', 'elev', 'roll', 'pitch', 'heading']:
        if key in gps_struct:
            gps_struct[key] = gps_struct[key].reshape(1, -1)
            assert np.shape(gps_struct[key]) == (1, n_records), f"Shape mismatch for {key}, expected (1, {n_records}), got {np.shape(gps_struct[key])}"
    
    return gps_struct


def save_gps_matlab_file(gps_struct: Dict[str, Any], output_path: Union[str, Path]):
    """
    Save GPS structure to MATLAB-compatible .mat file.
    
    Parameters:
    -----------
    gps_struct : dict
        GPS data structure
    output_path : str or Path
        Output file path (.mat extension recommended)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hdf5storage.savemat(output_path, gps_struct, format='7.3', store_python_metadata=False, matlab_compatible=True)


def generate_gps_file(gps_paths: List[Union[str, Path]], 
                     output_path: Union[str, Path],
                     use_ct: bool = True,
                     imu_paths: Optional[List[Union[str, Path]]] = None,
                     imu_keys_to_merge = ['ROLL', 'PITCH', 'HEADING'],
                     imu_extrapolation_limit = 1,
                     extrapolated_gps_time = 1) -> None:
    """
    Generate a MATLAB-compatible GPS file from UTIG raw GPS data.
    
    This is the main function that orchestrates the entire process:
    1. Load multiple GPS files
    2. Merge them sorted by time
    3. Convert to MATLAB format
    4. Save as .mat file
    
    Parameters:
    -----------
    gps_paths : list of str or Path
        List of paths to raw GPS data files
    output_path : str or Path
        Output file path for MATLAB GPS file
    use_ct : bool
        Whether to use CT time if available (default: True)
        
    Example:
    --------
    >>> gps_paths = [
    ...     '/data/UTIG/ASB/JKB2s/GL0107b/GPSnc1/xds.gz',
    ...     '/data/UTIG/ASB/JKB2s/GL0107c/GPSnc1/xds.gz'
    ... ]
    >>> generate_gps_file(gps_paths, 'output/gps_20180105_01.mat')
    """
    print(f"Processing {len(gps_paths)} GPS files...")
    
    # Load and merge GPS files
    merged_df = merge_gps_files(gps_paths, use_ct=use_ct)
    print(f"Merged {len(merged_df)} GPS records")

    if imu_paths:
        merged_imu_df = merge_imu_files(imu_paths, time_sort_key='GPS_TIME')
        print(f"Merged {len(merged_imu_df)} IMU records")

        # Merge imu_keys_to_merge keys from IMU data into GPS data based on gps_time
        for key in imu_keys_to_merge:
            if key in merged_imu_df.columns:
                # Interpolate IMU data by unix_time to match GPS timestamps
                imu_times = merged_imu_df['GPS_TIME'].values
                imu_values = merged_imu_df[key].values
                gps_times = merged_df['GPS_TIME'].values

                interpolated_values = np.interp(gps_times, imu_times, imu_values)

                # Fill any extrapolation beyond limits with NaN
                mask = (gps_times < imu_times[0] - imu_extrapolation_limit) | (gps_times > imu_times[-1] + imu_extrapolation_limit)
                interpolated_values[mask] = np.nan

                merged_df[key] = interpolated_values
            else:
                warnings.warn(f"Key '{key}' not found in IMU data. Filling with zeros.")
    
    # Add extrapolated values to cover small time gaps between GPS and radar
    if extrapolated_gps_time and extrapolated_gps_time > 0:
        entry_before = merged_df.iloc[0]
        entry_after = merged_df.iloc[-1]
        for t_key in ['unix_time', 'comp_time', 'radar_time']:
            entry_before[t_key] -= extrapolated_gps_time
            entry_after[t_key] += extrapolated_gps_time
        
        merged_df = pd.concat([pd.DataFrame([entry_before]), merged_df, pd.DataFrame([entry_after])], ignore_index=True)
        
    # Convert to MATLAB structure
    gps_struct = create_gps_matlab_structure(merged_df)
    
    # Save to MATLAB file
    save_gps_matlab_file(gps_struct, output_path)
    print(f"Saved GPS file to: {output_path}")

    # Print summary statistics
    print(f"GPS time range: {gps_struct['gps_time'][0,0]:.2f} to {gps_struct['gps_time'][0,-1]:.2f}")
    print(f"Lat range: {gps_struct['lat'].min():.6f} to {gps_struct['lat'].max():.6f}")
    print(f"Lon range: {gps_struct['lon'].min():.6f} to {gps_struct['lon'].max():.6f}")
    print(f"Elev range: {gps_struct['elev'].min():.2f} to {gps_struct['elev'].max():.2f} meters")

def make_segment_gps_file(x, output_base_dir, overwrite=False):
    # Designed to be called by .apply on df_season grouped by segment
    # Example usage: gps_paths = df_season.groupby(['segment_date_str', 'segment_number'])[['segment_date_str', 'segment_number', 'gps_path']].apply(opr_gps_file_generation.make_segment_gps_file, include_groups=False, overwrite=False)
    x = x.reset_index()
    print(f"{x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}")
    gps_paths = list(x['gps_path'].unique())
    if 'imu_path' in x:
        if x['imu_path'].isnull().any():
            warnings.warn(f"IMU paths contain null values for segment {x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}")
            imu_paths = None
        else:
            imu_paths = list(x['imu_path'].unique())
    else:
        imu_paths = None
    output_path = output_base_dir / Path(f"gps_{x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}.mat")

    # Only generate if the file does not exist

    if (not output_path.exists()) or overwrite:
        generate_gps_file(gps_paths, output_path, imu_paths=imu_paths)
    else:
        print(f"File {output_path} already exists. Skipping generation. If you want to regenerate, delete the file or set overwrite=True.")

    return output_path.resolve()