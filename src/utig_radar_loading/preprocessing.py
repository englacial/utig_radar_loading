"""
Preprocessing module for UTIG radar data.

This module replicates key functionality from the MATLAB preprocessing workflow,
specifically for creating parameter spreadsheets and temporary header files
that are needed by the MATLAB records_create step.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import struct
from typing import Dict, List, Optional, Union, Any
import yaml
import warnings
from datetime import datetime
from tqdm import tqdm

# Import existing stream_util functions
from . import stream_util


def extract_headers(bxds_files: List[Union[Path, str]]) -> Dict:
    """
    Extract timing and offset information from RADnh5 bxds files.
    
    Each bxds file is expected to have a corresponding ct.gz file
    in the same directory for timing information.
    
    Parameters:
    -----------
    bxds_files : list of Path
        List of bxds data files to process
        
    Returns:
    --------
    dict
        Dictionary containing:
        - radar_time: array of radar timing values
        - comp_time: array of computer timing values (from CT files)
        - file_idxs: array mapping each record to its source file
        - offsets: array of byte offsets for each record
    """
    headers = {
        'radar_time': [],
        'comp_time': [],
        'file_idxs': [],
        'offsets': []
    }
    
    # Process each bxds file
    for file_idx, bxds_file in enumerate(tqdm(bxds_files)):
        bxds_file = Path(bxds_file)  # Ensure bxds_file is a Path object
        if not bxds_file.exists():
            warnings.warn(f"File not found: {bxds_file}")
            continue
        
        # Load CT timing for this file using stream_util
        ct_data = stream_util.load_ct_file(bxds_file)
        ct_data = stream_util.parse_CT(ct_data)

        headers

        # try:
        #     ct_data = stream_util.load_ct_file(bxds_file)
        #     print(f"Loaded {len(ct_data)} CT records for {bxds_file.name}")
        # except FileNotFoundError:
        #     warnings.warn(f"No CT file found for {bxds_file}")
        # except Exception as e:
        #     warnings.warn(f"Error loading CT file for {bxds_file}: {e}")
            
        # # Read header from bxds file (simplified - just get record count and timing)
        # with open(bxds_file, 'rb') as f:
        #     # Read first XDS header to get structure
        #     nsamp = struct.unpack('>H', f.read(2))[0]  # uint16, big-endian
        #     nchan = struct.unpack('B', f.read(1))[0]   # uint8
            
        #     # Calculate record size
        #     header_size = 36 + 17*2  # XDS header + odd stuff
        #     data_size = 2 * nsamp * nchan
        #     record_size = header_size + data_size
            
        #     # Get file size
        #     f.seek(0, 2)
        #     file_size = f.tell()
        #     num_records = file_size // record_size
            
        #     print(f"nsamp: {nsamp}, nchan: {nchan}, num_records: {num_records}")
            
        #     # Extract timing for each record
        #     for rec_idx in range(num_records):
        #         offset = rec_idx * record_size
        #         f.seek(offset)
                
        #         # Read XDS header fields
        #         f.seek(offset + 28)  # Skip to rseq field
        #         rseq = struct.unpack('>I', f.read(4))[0]  # uint32, big-endian
                
        #         headers['offsets'].append(offset)
        #         headers['file_idxs'].append(file_idx)
        #         headers['radar_time'].append(rseq)  # Using rseq as proxy for time
                
        #         # Add CT timing if available
        #         if ct_data is not None and rec_idx < len(ct_data):
        #             # Convert CT time to Unix epoch using datenum calculation
        #             ct_row = ct_data.iloc[rec_idx]
        #             # Use the 'tim' field as the primary time reference
        #             headers['comp_time'].append(ct_row['tim'])
        #         else:
        #             headers['comp_time'].append(np.nan)
    
    # Convert to numpy arrays
    for key in headers:
        headers[key] = np.array(headers[key])
    
    return headers


def create_segments_from_frames(frames_plan_df: pd.DataFrame, 
                                file_list: List[Path],
                                headers: Optional[Dict] = None) -> List[Dict]:
    """
    Map frame boundaries from frames_plan_df to file indices and record numbers.
    
    Uses exact boundaries from frames_plan_df.
    
    Parameters:
    -----------
    frames_plan_df : DataFrame
        DataFrame with frame definitions including start/stop times
    file_list : list of Path
        List of data files in chronological order
    headers : dict, optional
        Header information from extract_headers
        
    Returns:
    --------
    list of dict
        List of segment dictionaries with file paths and record ranges
    """
    segments = []
    
    # Determine how to map frames to files
    # If we have headers with timing, use that; otherwise use simple division
    if headers and 'comp_time' in headers and not np.all(np.isnan(headers['comp_time'])):
        # Use timing information to map frames to files
        times = headers['comp_time']
        file_idxs = headers['file_idxs']
        
        for enum_idx, (idx, row) in enumerate(frames_plan_df.iterrows()):
            # Handle pandas Series .get() properly
            day_seg = row.get('day_seg')
            if day_seg is None:
                date_str = row.get('date', 'unknown')
                if hasattr(date_str, '__iter__') and not isinstance(date_str, str):
                    date_str = 'unknown'
                day_seg = f"{date_str}_{enum_idx+1:02d}"
            
            segment = {
                'day_seg': day_seg,
                'start_time': row.get('start_time'),
                'stop_time': row.get('stop_time'),
                'file_paths': [],
                'start_record': None,
                'stop_record': None
            }
            
            # Find files that contain data within this time range
            if 'start_time' in row and 'stop_time' in row and not np.all(np.isnan(times)):
                # Find records within time range
                # Note: This assumes times are comparable to frame start/stop times
                # You may need to adjust this based on actual time format
                time_mask = (times >= row['start_time']) & (times <= row['stop_time'])
                if np.any(time_mask):
                    valid_file_idxs = np.unique(file_idxs[time_mask])
                    # Collect actual file paths for this segment
                    segment['file_paths'] = [file_list[int(i)] for i in valid_file_idxs]
                    
                    # Get record ranges
                    record_indices = np.where(time_mask)[0]
                    segment['start_record'] = int(record_indices[0])
                    segment['stop_record'] = int(record_indices[-1])
            
            segments.append(segment)
    else:
        # Simple mapping: divide files equally among frames
        files_per_segment = max(1, len(file_list) // len(frames_plan_df))
        remainder = len(file_list) % len(frames_plan_df)
        
        file_idx = 0
        for enum_idx, (idx, row) in enumerate(frames_plan_df.iterrows()):
            # Handle pandas Series .get() properly
            day_seg = row.get('day_seg')
            if day_seg is None:
                date_str = row.get('date', 'unknown')
                if hasattr(date_str, '__iter__') and not isinstance(date_str, str):
                    date_str = 'unknown'
                day_seg = f"{date_str}_{enum_idx+1:02d}"
            
            segment = {
                'day_seg': day_seg,
                'start_time': row.get('start_time'),
                'stop_time': row.get('stop_time'),
                'file_paths': [],
                'start_record': 0,  # Default to first record
                'stop_record': None
            }
            
            # Calculate how many files for this segment
            segment_files = files_per_segment
            if enum_idx < remainder:
                segment_files += 1
            
            # Get the actual file paths for this segment
            end_idx = min(file_idx + segment_files, len(file_list))
            segment['file_paths'] = file_list[file_idx:end_idx]
            
            # If we have headers, get the actual record count for the last file
            if headers and 'file_idxs' in headers and segment['file_paths']:
                # Find the index of the last file in this segment
                last_file_idx = file_list.index(segment['file_paths'][-1])
                last_file_records = np.sum(headers['file_idxs'] == last_file_idx)
                if last_file_records > 0:
                    segment['stop_record'] = last_file_records - 1
            
            file_idx = end_idx
            segments.append(segment)
    
    return segments


def generate_parameters(segment: Dict, defaults: Dict, 
                       season_name: str, radar_name: str,
                       base_dir: str, board_folder_name: str) -> Dict:
    """
    Generate parameter dictionary for one segment.
    
    Parameters:
    -----------
    segment : dict
        Segment information from create_segments_from_frames
    defaults : dict
        Default parameters loaded from YAML
    season_name : str
        Season name (e.g., '2022_Antarctica_BaslerMKB')
    radar_name : str
        Radar name (e.g., 'rds')
    base_dir : str
        Base directory for data files
    board_folder_name : str
        Board/folder name for data organization
        
    Returns:
    --------
    dict
        Parameters for this segment organized by spreadsheet section
    """
    params = {}
    
    # cmd sheet parameters
    params['cmd'] = defaults.get('cmd', {}).copy()
    params['cmd']['day_seg'] = segment['day_seg']
    
    # records sheet parameters
    params['records'] = defaults.get('records', {}).copy()
    
    # New format: include file paths directly
    if segment.get('file_paths'):
        # Convert Path objects to strings for CSV serialization
        params['records']['file.paths'] = [str(p) for p in segment['file_paths']]
        # Still include start/stop indices for compatibility if needed
        params['records']['file.start_idx'] = 0
        params['records']['file.stop_idx'] = len(segment['file_paths']) - 1
    else:
        params['records']['file.paths'] = []
        params['records']['file.start_idx'] = 0
        params['records']['file.stop_idx'] = 0
    
    params['records']['file.base_dir'] = base_dir
    params['records']['file.board_folder_name'] = board_folder_name
    
    if segment.get('start_record') is not None:
        params['records']['file.start_record'] = segment['start_record']
        params['records']['file.stop_record'] = segment['stop_record']
    
    # qlook sheet parameters
    params['qlook'] = defaults.get('qlook', {}).copy()
    
    # sar sheet parameters
    params['sar'] = defaults.get('sar', {}).copy()
    
    # array sheet parameters
    params['array'] = defaults.get('array', {}).copy()
    
    # radar sheet parameters
    params['radar'] = defaults.get('radar', {}).copy()
    
    # post sheet parameters
    params['post'] = defaults.get('post', {}).copy()
    
    # Add any additional sheets from defaults
    for sheet_name in defaults:
        if sheet_name not in params:
            params[sheet_name] = defaults[sheet_name].copy()
    
    return params


def write_parameter_spreadsheet(all_params: List[Dict], output_file: Union[str, Path],
                               additional_sheets: Optional[Dict[str, Dict]] = None):
    """
    Write parameters to CSV files (one per sheet).
    
    Handles special formatting for list fields like file.paths.
    
    Parameters:
    -----------
    all_params : list of dict
        List of parameter dictionaries, one per segment
    output_file : str or Path
        Base path for output files (will create multiple CSVs)
    additional_sheets : dict, optional
        Additional sheets to write with default values
    """
    output_path = Path(output_file)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_path.stem
    
    # Get all sheet names
    sheet_names = set()
    for params in all_params:
        sheet_names.update(params.keys())
    
    # Add additional sheets
    if additional_sheets:
        sheet_names.update(additional_sheets.keys())
    
    # Write each sheet as a separate CSV
    for sheet_name in sorted(sheet_names):
        sheet_data = []
        
        for params in all_params:
            if sheet_name in params:
                row = params[sheet_name].copy()
                # Add day_seg as first column for identification
                if 'day_seg' not in row and 'cmd' in params:
                    row = {'day_seg': params['cmd'].get('day_seg', '')} | row
                
                # Handle list fields - convert to semicolon-separated strings
                for key, value in row.items():
                    if isinstance(value, list):
                        # Convert list to semicolon-separated string
                        row[key] = ';'.join(str(v) for v in value)
                
                sheet_data.append(row)
            elif additional_sheets and sheet_name in additional_sheets:
                # Use defaults from additional_sheets
                row = additional_sheets[sheet_name].copy()
                if 'cmd' in params:
                    row = {'day_seg': params['cmd'].get('day_seg', '')} | row
                sheet_data.append(row)
        
        if sheet_data:
            df = pd.DataFrame(sheet_data)
            csv_file = output_dir / f"{base_name}_{sheet_name}.csv"
            df.to_csv(csv_file, index=False)
            print(f"Wrote {csv_file}")


def save_temporary_headers(headers: Dict, file_list: List[Path], 
                          output_dir: Path, season_name: str, 
                          board_folder_name: str = ''):
    """
    Save temporary header files for MATLAB records_create.
    
    Parameters:
    -----------
    headers : dict
        Header information from extract_headers
    file_list : list of Path
        List of source data files
    output_dir : Path
        Base directory for temporary files (typically /opr_tmp or similar)
    season_name : str
        Season name for directory structure
    board_folder_name : str
        Board folder name (used in directory structure)
    """
    import scipy.io
    
    # Group headers by file and save
    for file_idx, data_file in enumerate(file_list):
        # Get records for this file
        file_mask = headers['file_idxs'] == file_idx
        if not np.any(file_mask):
            continue
        
        # Create output directory structure matching MATLAB's opr_filename_opr_tmp
        # Pattern: output_dir/headers/season_name/board_folder_name/
        if board_folder_name:
            header_dir = output_dir / 'headers' / season_name / board_folder_name
        else:
            header_dir = output_dir / 'headers' / season_name
        header_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data for this file
        file_headers = {
            'offset': headers['offsets'][file_mask],
            'radar_time': headers['radar_time'][file_mask],
            'comp_time': headers['comp_time'][file_mask]
        }
        
        # Save to .mat file with same name as data file
        output_file = header_dir / f"{data_file.stem}.mat"
        scipy.io.savemat(str(output_file), file_headers, do_compression=True)
        print(f"Saved header file: {output_file}")


def load_defaults(yaml_file: Union[str, Path]) -> Dict:
    """
    Load default parameters from YAML file.
    
    Parameters:
    -----------
    yaml_file : str or Path
        Path to YAML file with default parameters
        
    Returns:
    --------
    dict
        Default parameters organized by sheet name
    """
    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        warnings.warn(f"Defaults file not found: {yaml_path}")
        return {}
    
    with open(yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)
    
    return defaults if defaults else {}