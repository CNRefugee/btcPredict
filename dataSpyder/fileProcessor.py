import os
import pandas as pd

directory = '../data/rawData'
def save_dataframe_to_pickle(dataframe, filepath):
    if os.path.exists(filepath):
        print(f"Warning: The file {filepath} already exists and will be overwritten.")

    try:
        dataframe.to_pickle(filepath)
        print(f"DataFrame saved successfully to {filepath}")
    except Exception as e:
        print(f"Failed to save DataFrame to pickle. Error: {e}")



def append_dataframe_to_pickle(dataframe, filepath):
    try:
        # Read the existing data if file exists
        if pd.io.common.file_exists(filepath):
            existing_df = pd.read_pickle(filepath)
            dataframe = pd.concat([existing_df, dataframe], ignore_index=False)

        # Save the combined DataFrame
        dataframe.to_pickle(filepath)
        print(f"DataFrame appended successfully to {filepath}")
    except Exception as e:
        print(f"Failed to append DataFrame to pickle. Error: {e}")





def save_dataframe_to_csv(dataframe, filepath, index=False):
    if os.path.exists(filepath):
        print(f"Warning: The file {filepath} already exists and will be overwritten.")

    try:
        dataframe.to_csv(filepath, index=index)
        print(f"DataFrame saved successfully to {filepath}")
    except Exception as e:
        print(f"Failed to save DataFrame to CSV. Error: {e}")

def save_dataframe_to_csv_append(dataframe, filepath, index=False):
    try:
        # Check if file exists to decide header write
        header = not pd.io.common.file_exists(filepath)
        dataframe.to_csv(filepath, mode='a', index=index, header=header)
        print(f"DataFrame appended successfully to {filepath}")
    except Exception as e:
        print(f"Failed to append DataFrame to CSV. Error: {e}")


def read_data_file(filepath):
    """
    Reads a data file into a Pandas DataFrame. Supports .pkl and .csv files.

    Parameters:
    - filepath: str, the path to the file to be read.

    Returns:
    - A Pandas DataFrame containing the data from the file.
    """
    if filepath.endswith('.pkl'):
        # Load a pickle file
        df = pd.read_pickle(filepath)
    elif filepath.endswith('.csv'):
        # Load a CSV file
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a .pkl or .csv file.")

    return df
