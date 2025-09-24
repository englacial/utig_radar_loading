import numpy as np
import pandas as pd
from tqdm import tqdm

from utig_radar_loading import stream_util
from utig_radar_loading.geo_util import project_split_and_simplify

def assign_segments(df_season, timestamp_field='tim', timestamp_split_threshold=1000, parse_ct=False):
    """
    Assigns segment paths to each row in df_season based on gaps in the specified timestamp field.
    Parameters:
    - df_season: DataFrame containing the season data with radar paths.
    - timestamp_field: The field in the CT files to use for detecting gaps (default is 'tim').
    - timestamp_split_threshold: The threshold for splitting segments based on the timestamp difference (default is 1000).
    - parse_ct: Whether to parse the CT files (default is False).
    Returns:
    - df_season with additional columns: 'segment_path', 'segment_date_str', 'segment_number'.
    """
    
    last_segment_ct = stream_util.load_ct_file(df_season.iloc[0]['radar_path'])
    if parse_ct:
        last_segment_ct = stream_util.parse_CT(last_segment_ct)

    df_season['segment_path'] = ""
    df_season['segment_date_str'] = ""
    df_season['segment_number'] = -1
    current_segment_datestring = df_season.iloc[0]['start_timestamp'].strftime("%Y%m%d")
    current_segment_idx = 1

    df_season.iloc[0, df_season.columns.get_loc('segment_date_str')] = current_segment_datestring
    df_season.iloc[0, df_season.columns.get_loc('segment_path')] = f"{current_segment_datestring}_{current_segment_idx:02d}"
    df_season.iloc[0, df_season.columns.get_loc('segment_number')] = current_segment_idx


    print(f"Initial segment path is: {df_season.iloc[0]['segment_path']}")

    for row_iloc in tqdm(range(1, len(df_season))):
        try:
            curr_segment_ct = stream_util.load_ct_file(df_season.iloc[row_iloc]['radar_path'])
            if parse_ct:
                curr_segment_ct = stream_util.parse_CT(curr_segment_ct)

            delta_from_last = curr_segment_ct[timestamp_field].iloc[0] - last_segment_ct[timestamp_field].iloc[-1]

            if np.abs(delta_from_last) > timestamp_split_threshold:
                new_datestring = df_season.iloc[row_iloc]['start_timestamp'].strftime("%Y%m%d")
                if new_datestring == current_segment_datestring:
                    current_segment_idx += 1
                else:
                    current_frame_idx = 1
                    current_segment_idx = 1
                    current_segment_datestring = new_datestring

                print(f"Segment path changed to {current_segment_datestring}_{current_segment_idx:02d}. Delta in '{timestamp_field}' was {delta_from_last}")

            df_season.iloc[row_iloc, df_season.columns.get_loc('segment_date_str')] = current_segment_datestring
            df_season.iloc[row_iloc, df_season.columns.get_loc('segment_path')] = f"{current_segment_datestring}_{current_segment_idx:02d}"
            df_season.iloc[row_iloc, df_season.columns.get_loc('segment_number')] = current_segment_idx

            last_segment_ct = curr_segment_ct
        except Exception as e:
            print(f"Could not load index {row_iloc}")
            print(df_season.iloc[row_iloc]['radar_path'])
            print(e)
    
    return df_season