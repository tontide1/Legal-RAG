import pandas as pd

def read_csv(file_path):
    """Reads a CSV file and returns a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

def extract_relevant_data(df, columns):
    """Extracts relevant data from the DataFrame based on specified columns."""
    if columns:
        return df[columns]
    return df

def save_to_chroma(df, vector_store):
    """Saves the DataFrame content to the Chroma vector store."""
    for index, row in df.iterrows():
        # Assuming vector_store has a method to add data
        vector_store.add_data(row.to_dict())