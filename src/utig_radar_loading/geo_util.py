import pyproj
from shapely import LineString
import numpy as np
import pandas as pd
import holoviews as hv
from utig_radar_loading import stream_util, opr_gps_file_generation

def load_gps_data(transects_df, source_type=None):
    segment_dfs = []

    for idx, row in transects_df.iterrows():

        if source_type is None:
            if pd.notna(row['postprocessed_gps_path']):
                tmp_source_type = 'postprocessed'
            else:
                tmp_source_type = 'field'
        else:
            tmp_source_type = source_type

        if tmp_source_type == 'field':
            f = row['gps_path']
            
            df = stream_util.load_xds_stream_file(f, parse=True)
        elif tmp_source_type == 'postprocessed':
            f = row['postprocessed_gps_path']
            df = opr_gps_file_generation.load_and_parse_postprocessed_gps_file(f)
            df['prj'] = idx[0]
            df['set'] = idx[1]
            df['trn'] = idx[2]
        else:
            raise ValueError(f"Unknown source_type {tmp_source_type}")

        #line_length_km = stream_util.calculate_track_distance_km(df)
        #_, _, line_length_m_shapely = project_split_and_simplify(df['LON'].values, df['LAT'].values, calc_length=True, simplify_tolerance=100)

        necessary_keys = ['prj', 'set', 'trn', 'clk_y', 'LAT', 'LON', 'TIMESTAMP']
        for k in necessary_keys:
            if k not in df:
                df[k] = np.nan

        df_sub = df[['prj', 'set', 'trn', 'clk_y', 'LAT', 'LON', 'TIMESTAMP']]

        if 'segment_path' in row:
            df_sub['segment_path'] = row['segment_path']

        segment_dfs.append(df_sub)
    return segment_dfs

def project_split_and_simplify(lon, lat, projection='EPSG:3031', simplify_tolerance=1000,
                                split_dist=2000, calc_length=False):
    # Project
    transformer = pyproj.Transformer.from_crs('EPSG:4326', projection, always_xy=True)
    x_proj, y_proj = transformer.transform(lon, lat)

    # Break into separate segments
    dist_deltas = np.sqrt(np.diff(x_proj)**2 + np.diff(y_proj)**2)
    segment_indices = np.array(np.where(dist_deltas > split_dist)) + 1
    segment_indices = np.insert(segment_indices, 0, 0)
    segment_indices = np.append(segment_indices, len(x_proj))

    x_simplified = []
    y_simplified = []

    length = 0

    for start_idx, end_idx in zip(segment_indices[:-1], segment_indices[1:]):
        if end_idx - start_idx < 5:
            continue

        x_segment = x_proj[start_idx:end_idx]
        y_segment = y_proj[start_idx:end_idx]

        if np.isnan(x_segment).any() or np.isnan(y_segment).any():
            print(f"Warning: NaN values found in segment {start_idx}:{end_idx}")
            continue

        # Use shapely to simplify paths to 1km tolerance
        line = LineString(zip(x_segment, y_segment))
        if calc_length:
            length += line.length
        
        if simplify_tolerance:
            line = line.simplify(tolerance=simplify_tolerance)
        coords = list(line.coords)

        x_simplified.extend([c[0] for c in coords])
        y_simplified.extend([c[1] for c in coords])
        x_simplified.append(np.nan)
        y_simplified.append(np.nan)

    if calc_length:
        return x_simplified, y_simplified, length
    else:
        return x_simplified, y_simplified

def create_path(segment_dfs, path_opts_kwargs={}):
    dfs = []

    for idx, df_sub in enumerate(segment_dfs):
        df_tmp = df_sub.copy()
        df_tmp = df_tmp[df_tmp['LAT'] <= -50]

        if len(df_tmp) < 3:
            continue

        try:
            x_proj, y_proj = project_split_and_simplify(df_tmp['LON'].values, df_tmp['LAT'].values)
        except Exception as e:
            print(f"Error processing segment {idx}: {e}")
            #print(df_tmp)
            continue

        # Finish with a nan to divide from next
        x_proj.append(np.nan)
        y_proj.append(np.nan)

        # Add projected coordinates to dataframe
        df_simplified = pd.DataFrame({
            'x': x_proj,
            'y': y_proj
        })

        required_fields = ['prj', 'set', 'trn']
        optional_fields = ['segment_path', 'radar_stream_type']
        display_fields = required_fields.copy()
        for k in optional_fields:
            if k in df_tmp:
                display_fields.append(k)

        for k in display_fields:
            df_simplified[k] = df_tmp[k].iloc[0]
            if len(df_tmp[k].unique()) > 1:
                print(f"segment_dfs[{idx}]['{k}'].unique(): {df_tmp[k].unique()}")

        dfs.append(df_simplified)

    df_combined = pd.concat(dfs, ignore_index=True)
    # Create hv.Path with already projected coordinates
    path = hv.Path(df_combined,
                ['x', 'y'],
                display_fields,
                ).opts(
                    tools=['hover'],
                    line_width=0.5,
                    show_legend=True,
                    **path_opts_kwargs
                )
    return dfs, path