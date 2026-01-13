import pandas as pd

def load_local_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)