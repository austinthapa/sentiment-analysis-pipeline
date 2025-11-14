import os
import sys
import tempfile
import pytest
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from pandas import DataFrame
from src.data import text_data_preprocessing
from unittest.mock import patch
from pathlib import Path

def test_preprocess_text():
    """
    Test preprocess_text with stopwords removal and lemmatization
    """
    input_txt = "I was running @ to 3 cars!!!"
    result = text_data_preprocessing.preprocess_text(
        text=input_txt,
        remove_stopwords=True,
        lemmatize=True
    )
    result_tokens = set(result.split())
    
    expected_words = {"running", "car"}
    
    assert expected_words.issubset(result_tokens)

def test_normalize_text_column():
    """
    Test normalize_text_column
    """
    df = DataFrame({
        "clean_comment": ["hello", None, "hello", "world"]
    })
    with patch("src.data.text_data_preprocessing.preprocess_text", lambda x: x):
        result = text_data_preprocessing.normalize_text_column(df)

    # Rows after cleaning: "hello", "world"
    assert len(result) == 2
    assert set(result["clean_comment"]) == {"hello", "world"}
    
    
def test_save_preprocessed_data():
    """
    Create a sample data
    """
    train_df = DataFrame({
        "feature": range(5),
        "target": [1] * 5
    })
    test_df = DataFrame({
        "feature": range(2),
        "target": [0, 1]
    })
    
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)
    
        text_data_preprocessing.save_preprocessed_data(
            train_data=train_df, 
            test_data=test_df,
            data_path=tmp_path
        )

        train_file = tmp_path / "train.csv"
        test_file = tmp_path / "test.csv"
        
        assert train_file.exists() and train_file.is_file()
        assert test_file.exists() and test_file.is_file()
        
        loaded_train = pd.read_csv(train_file)
        loaded_test = pd.read_csv(test_file)
        
        pd.testing.assert_frame_equal(train_df, loaded_train)
        pd.testing.assert_frame_equal(test_df, loaded_test)