import numpy as np
import pandas as pd
import os
from pathlib import Path
import unfoc
from utig_radar_loading import stream_util


def create_header_df(input_filename, waveform_record_length=6400):
    ct_data = stream_util.load_ct_file(input_filename)
    ct_data = stream_util.parse_CT(ct_data)
    fpos, header_len, header = zip(*unfoc.index_RADnhx_bxds(input_filename, full_header=True))

    rseq = np.array([h.rseq for h in header])
    choff = np.array([h.choff for h in header])

    df = pd.DataFrame({
        'rseq': rseq,
        'choff': choff,
        'start_fpos': fpos,
        'header_len': header_len,
        'ch0_offset': pd.NA,
        'ch1_offset': pd.NA,
        'ch2_offset': pd.NA,
        'ch3_offset': pd.NA
    })

    df = df.join(ct_data[['tim', 'COMP_TIME']])

    choff0 = np.where(df['choff'] == 0)[0]
    choff2 = np.where(df['choff'] == 2)[0]
    df.iloc[choff0, df.columns.get_loc('ch0_offset')] = df.iloc[choff0]['start_fpos'] + df.iloc[choff0]['header_len']
    df.iloc[choff0, df.columns.get_loc('ch1_offset')] = df.iloc[choff0]['start_fpos'] + df.iloc[choff0]['header_len'] + waveform_record_length
    df.iloc[choff2, df.columns.get_loc('ch2_offset')] = df.iloc[choff2]['start_fpos'] + df.iloc[choff2]['header_len']
    df.iloc[choff2, df.columns.get_loc('ch3_offset')] = df.iloc[choff2]['start_fpos'] + df.iloc[choff2]['header_len'] + waveform_record_length

    df = df[['tim', 'COMP_TIME', 'rseq', 'ch0_offset', 'ch1_offset', 'ch2_offset', 'ch3_offset']]
    # Group by rseq and get first non-NAN value for each channel
    df = df.groupby('rseq').first()
    # Set nan values to -2^31
    with pd.option_context('future.no_silent_downcasting', True):
        df = df.fillna(-2**31)

    # Check for offsets outside the file size
    file_size = os.path.getsize(input_filename)
    max_offset_allowed = file_size - waveform_record_length
    ch_cols = ['ch0_offset', 'ch1_offset', 'ch2_offset', 'ch3_offset']
    max_offset_df = df[ch_cols].max().max()
    if max_offset_df > max_offset_allowed:
        print(f"[WARNING] Header has offset of {max_offset_df}, which is too large for a file of {file_size} bytes. Violating rows will be dropped.")
        df = df[df[ch_cols].max(axis=1) <= max_offset_allowed]

    return df

def get_header_information(f):
    df = create_header_df(f)

    offsets_array = df[['ch0_offset', 'ch1_offset', 'ch2_offset', 'ch3_offset']].to_numpy()
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