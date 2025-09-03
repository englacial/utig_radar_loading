import pandas as pd
import glob
import os
from pathlib import Path

def load_file_index_df(base_path : str, cache_file : str, read_cache : bool = True) -> pd.DataFrame:
    cache_path = Path(cache_file)
    if read_cache and cache_path.exists():
        # Read from cache
        print(f"Reading from cache file {cache_path}")
        df_files = pd.read_csv(cache_path, index_col=0)
        df_files.columns = df_files.columns.astype(int)
    else:
        # Load and process files to create DataFrame
        print(f"Generating file index")
        print(f"Looking for xds.gz")
        xds_files = glob.glob(f"{base_path}/**/xds.gz", recursive=True)
        print(f"Looking for bxds")
        bxds_files = glob.glob(f"{base_path}/**/bxds*", recursive=True)
        df_files = pd.DataFrame([Path(f).parts for f in (xds_files + bxds_files)])
        
        if cache_file is not None:
            print(f"Saving new cache to {cache_file}")
            df_files.to_csv(cache_path)
    
    return df_files

def create_artifacts_df(file_index_df : pd.DataFrame, datasets=['UTIG1', 'UTIG2']) -> pd.DataFrame:
    column_mapping = {
        5: "dataset",
        6: "processing_level",
        7: "processing_type",
        8: "prj",
        9: "set",
        10: "trn",
        11: "stream",
        12: "file_name",
        "full_path": "full_path"
        }

    df_tmp = file_index_df.copy()

    df_tmp = df_tmp.dropna(axis='columns')

    df_tmp["full_path"] = df_tmp.apply(lambda row: Path(*row).as_posix(), axis=1)

    df_tmp = df_tmp[list(column_mapping.keys())]
    df_artifacts = df_tmp.rename(columns=column_mapping)

    if datasets is not None:
        df_artifacts = df_artifacts[df_artifacts['dataset'].isin(datasets)]

    #df_artifacts = df_tmp.drop(columns=[0,1,2,3,4,12,13])

    df_artifacts['artifact'] = df_artifacts.apply(lambda row: tuple(row[['processing_level', 'processing_type', 'stream']]), axis='columns')

    return df_artifacts