import os
import yaml
import logging
import pandas as pd

from pathlib import Path
from typing import Tuple
from pandas import DataFrame
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)
logger = logging.getLogger(__name__)

"""
load_config -> load_data -> preprocess_data -> save_data
"""

def load_config(
    config_path = "config/paths.yaml"
) -> dict:
    """
    Loads the configuration from a YAML file with error handling.
    
    Args:
        config_path: Configuration path
    
    Returns:
        dict: A dictionary containing configuration data.
        
    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: Error related with parsing YAML file.
        Exceptions: For all other unrelated exceptions.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logging.info(f"Successfully loaded configuration from: {config_path}")
            return config         
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise 
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading config: {e}")
        raise
    
def load_data(
    data_path: str
) -> DataFrame:
    """
    Load raw data from a given data source.
    If performs basic validation on the input source, checks if the data is empty, and logs the outcome.
    
    Args:
        data_url (str): The URL where data resides.
        
    Returns:
        DataFrame: Loaded data in the form of pandas DataFrame
    
    Raises:
        ValueError: If the loaded data is empty.
        FileNotFoundError: If the specified file path does not exist.
        pd.errors.ParseError: If there is an issue with parsing CSV data.
        Exception: Any other errors during loading the data and raised for further handling.
    """
    try:
        if data_path.startswith(("http://", "https://")):
            df = pd.read_csv(data_path)
        else:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at: {data_path}")
            df = pd.read_csv(data_path)

        if df.empty:
            raise ValueError("Loaded data is empty")
        logger.info(f"""
                    Successfully loaded data from: {data_path}
                    DataFrame: {len(df)} rows and {len(df.columns)} columns.
                    """)
        return df
    except pd.errors.ParserError as e:
        logger.info(f"Unexpected error occured while loading the data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error error occurred: {e}", exc_info=True)
        raise

def split_data(
    df: DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataFrame, DataFrame]:
    """
    Split the data into train and test datasets
    
    Args:
        df: Raw DataFrame to split.
        test_size: Proportion of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 0.42)
    
    Returns:
        Tuple [DataFrame, DataFrame]
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    try:
        train_data, test_data = train_test_split(
            df, 
            test_size=test_size,
            random_state=random_state
        )
        logger.info(
            f"Data split complete: {len(train_data)} train, {len(test_data)} test"
        )
        return train_data, test_data
     
    except Exception as e:
        logger.error(f"Unexpected error occured: {e}", exc_info=True)
        raise

def save_data(
    train_data: DataFrame,
    test_data: DataFrame,
    data_path: Path
) -> None:
    """
    Save train and test datasets to CSV files.
    
    Args:
        train_data: Training dataset
        test_data: Testing dataset
        data_path: Directory path to save the data
       
    Raises:
        OSError: If directory creation or file writing fails
    """
    try:
        data_path.mkdir(parents=True, exist_ok=True)
        ext = ".csv"
        
        train_path = data_path / f"train{ext}"
        test_path = data_path / f"test{ext}"
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        logger.info(
            f"Data saved successfully:\n"
            f"  Train: {train_path} ({len(train_data)} rows)\n"
            f"  Test: {test_path} ({len(test_data)} rows)"
        )
    except Exception as e:
        logger.error(f"Unexpected error saving data: {e}", exc_info=True)
        raise

def main():
    """
    Main entry point for data ingestion
    """
    # Step 0: Load the configuration
    config = load_config(
        config_path="config/paths.yaml"
    )
    
    # Step 1: Load the data
    data_path = config["data"]["raw_data"]
    df = load_data(data_path=data_path)
    
    # Step 2: Split the loaded data into training and testing sets
    train_data, test_data = split_data(df)
    
    # Get the directory where the data will be saved
    data_dir = config["data"]["save_path"]
    if not data_dir.exists():
        os.makedirs(data_dir)
    
    # Save the split data (train and test datasets) to project directory
    save_data(train_data, test_data, data_dir)
    
if __name__ == "__main__":
    main()