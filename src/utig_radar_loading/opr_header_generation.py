import numpy as np
import pandas as pd
import os
from pathlib import Path
import unfoc
from utig_radar_loading import stream_util


def create_header_df(input_filename, waveform_record_length=6400):
    """
    Create header dataframe for RADnh3 and RADnh5 radar data files.
    Automatically detects file type and number of channels.

    Parameters:
    -----------
    input_filename : str
        Path to input bxds file
    waveform_record_length : int
        Length of waveform records in bytes (default: 6400)

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: tim, COMP_TIME, and ch{N}_offset for each channel
        Index is rseq (radar sequence number)
    """
    # Auto-detect stream type (RADnh3 or RADnh5)
    stream = unfoc.get_radar_stream(input_filename)

    ct_data = stream_util.load_ct_file(input_filename)
    ct_data = stream_util.parse_CT(ct_data)

    # This handles both RADnh3 and RADnh5
    fpos, header_len, header = zip(*unfoc.index_RADnhx_bxds(input_filename, full_header=True))

    # Get rseq based on stream type
    if stream == 'RADnh5':
        rseq = np.array([h.rseq for h in header])
    elif stream == 'RADnh3':
        # For RADnh3, rseq comes from CT file seq field
        rseq = ct_data['seq'].values
    else:
        raise ValueError(f"Unsupported stream type: {stream}")

    choff = np.array([h.choff for h in header])

    # Detect number of channels from unique choff values
    # Note: 0xff is treated as 0x00 (replaced later)
    unique_choff = np.unique(choff[choff != 0xff])
    num_channels = len(unique_choff) * 2  # Each choff represents 2 channels

    # Build dataframe with dynamic channel columns
    df_dict = {
        'rseq': rseq,
        'choff': choff,
        'start_fpos': fpos,
        'header_len': header_len,
    }

    # Add channel offset columns dynamically
    for i in range(num_channels):
        df_dict[f'ch{i}_offset'] = pd.NA

    df = pd.DataFrame(df_dict)
    df = df.join(ct_data[['tim', 'COMP_TIME']])

    # Assign channel offsets dynamically based on unique choff values
    for choff_val in unique_choff:
        choff_idx = np.where(df['choff'] == choff_val)[0]
        base_ch = choff_val  # First channel for this choff value

        df.iloc[choff_idx, df.columns.get_loc(f'ch{base_ch}_offset')] = \
            df.iloc[choff_idx]['start_fpos'] + df.iloc[choff_idx]['header_len']
        df.iloc[choff_idx, df.columns.get_loc(f'ch{base_ch+1}_offset')] = \
            df.iloc[choff_idx]['start_fpos'] + df.iloc[choff_idx]['header_len'] + waveform_record_length

    # Select relevant columns dynamically
    ch_cols = [f'ch{i}_offset' for i in range(num_channels)]
    df = df[['tim', 'COMP_TIME', 'rseq'] + ch_cols]

    # Group by rseq and get first non-NAN value for each channel
    df = df.groupby('rseq').first()

    # Set nan values to -2^31
    with pd.option_context('future.no_silent_downcasting', True):
        df = df.fillna(-2**31)

    # Check for offsets outside the file size
    file_size = os.path.getsize(input_filename)
    max_offset_allowed = file_size - waveform_record_length
    max_offset_df = df[ch_cols].max().max()
    if max_offset_df > max_offset_allowed:
        print(f"[WARNING] Header has offset of {max_offset_df}, which is too large for a file of {file_size} bytes. Violating rows will be dropped.")
        df = df[df[ch_cols].max(axis=1) <= max_offset_allowed]

    return df

def get_header_information(f):
    """
    Get header information from a radar data file.

    Parameters:
    -----------
    f : str
        Path to input bxds file

    Returns:
    --------
    dict
        Dictionary with keys:
        - 'comp_time': Computer time values
        - 'radar_time': Radar time values (from CT file)
        - 'offset': Nb x Nx x Nc array (boards x records x channels) with file offsets
                    Special value -2^31 indicates missing data
    """
    df = create_header_df(f)

    # Get all channel offset columns dynamically
    ch_cols = [col for col in df.columns if col.startswith('ch') and col.endswith('_offset')]

    offsets_array = df[ch_cols].to_numpy()
    # Final shape is Nb x Nx x Nc (boards x records x channels)
    offsets_array = np.expand_dims(offsets_array, axis=0).astype(np.int64) # Add a board axis

    headers = {
        'comp_time': df['COMP_TIME'].values,
        # radar_time is ct_data['tim']
        'radar_time': df['tim'].values,
        # offset is an Nb x Nx x Nc (board x records x channels) array with special value -2^31 for missing data
        'offset': offsets_array,
    }

    return headers

def get_header_file_location(f, base_dir):
    p = Path(f)
    fn_name = p.stem
    board_folder_name = Path(*p.parts[-5:-1])
    board_folder_name_cur = base_dir / board_folder_name
    return str(board_folder_name_cur / (fn_name + '.mat'))