import tempfile
import pytest
import pandas as pd

from pandas import DataFrame
from pathlib import Path
from data import data_ingestion


@pytest.fixture
def sample_csv(tmp_path):
    """
    Create a temporary CSV file for testing
    """
    df = DataFrame({
       "text": ["Fun day sunday", "Watch until the end", "I am pissed about being late"],
        "sentiment": [1, 0, -1]
    })
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index = False)
    return str(csv_path)

def test_load_data(sample_csv):
    """
    Test the load_data if it correctly reads the CSV file
    """
    df = data_ingestion.load_data(sample_csv)
    
    assert isinstance(df, DataFrame)
    assert not df.empty
    assert "text" in df.columns
    assert "sentiment" in df.columns
    assert len(df) == 3
 
def test_split_data():
    """
    Test split_data function if it properly splits the data
    """
    df = DataFrame({
        "feature": range(10),
        "target": [0, 1] * 5
    })
    
    train_df, test_df = data_ingestion.split_data(df, test_size=0.3, random_state=42)
    
    assert len(train_df) == 7
    assert len(test_df) == 3
    
    merged_df = pd.merge(train_df, test_df, how="inner")
    assert merged_df.empty
    
def test_save_data():
    """
    Test save_data function
    """
    train_df = DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 1]
    })
    test_df = DataFrame({
        "feature": [4, 5],
        "target": [1, 0]
    })
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)

        # Call save data
        data_ingestion.save_data(
            train_data=train_df, 
            test_data=test_df,
            data_path=tmp_path
        )
        
        train_file = tmp_path / "train.csv"
        test_file = tmp_path / "test.csv"
        
        assert train_file.exists() and train_file.is_file()
        assert test_file.exists() and test_file.is_file()
        
        # Reload back the train and test data
        loaded_train = pd.read_csv(train_file)
        loaded_test = pd.read_csv(test_file)
        
        pd.testing.assert_frame_equal(train_df, loaded_train)
        pd.testing.assert_frame_equal(test_df, loaded_test)