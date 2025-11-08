# Imports
import os
import re
import logging
import pandas as pd

from pathlib import Path
from pandas import DataFrame
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Configure logging
logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True
) -> str:
    """
    Preprocess a single text string through multiple cleaning steps.
    
    Args:
        text: Input text to process
        remove_stopwords: Whether to remove stopwords (default: True)
        lemmatize: Whether to apply lemmatize (default: True)
    
    Returns:
        Preprocessed text string
    """
    
    # Step 1: Removing leading/trailing whitespace
    text = text.strip()
    
    # Return early if empty.
    if not text:
        return ""

    # Step 2: Replace new lines with space
    text = text.replace("\n", " ")

    # Step 3: Remove all the digits
    text = re.sub(r"\d+", " ", text)

    # Step 4: Remove punctuations and emojis
    text = re.sub(r"[^A-Za-z\s]", " ", text)

    # Step 5: Convert text into lower case
    text = text.lower()

    # Step 6: Collapse multiple whitespaces into one
    text = re.sub(r"\s+", " ", text)

    # Step 7: Tokenize
    tokens = text.split()
    
    # Step 7: Remove all the stop words
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

    # Step 9: Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Step 10: Join back word into string
    return " ".join(tokens)


def normalize_text_column(
    df: DataFrame,
    text_column: str = "clean_comment"
) -> DataFrame:
    """
    Apply text normalization to a DataFrame column.
    
    Args:
        df: Input DataFrame
        text_column: Name of the column containing text.
    
    Returns:
        DataFrame with normalized text columns
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    try:
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        logger.info(f"Starting text normalization...")
        df["clean_comment"] = df["clean_comment"].apply(preprocess_text)
        logger.info("Text normalization complete...")
        return df
    except Exception as e:
        logger.error(f"Unexpected error occured during the text normalization: {e}", exc_info=True)
        raise
    
def save_preprocessed_data(
    train_data: DataFrame,
    test_data: DataFrame,
    data_path: Path
) -> None:
    """
    Save preprocessed train and test datasets.
    
    Args:
        train_data: Preprocessed training data
        test_data: Preprocessed testing data
        data_path: Directory path to save the data
    
    Raises:
        OSError: If directory creation or file writing fails.
    """
    try:
        data_path.mkdir(parents=True, exist_ok=True)
        ext = ".csv"
        train_path = data_path / f"train{ext}"
        test_path = data_path / f"test{ext}"
        
        logger.info(f"Saving preprocessed train data to: {train_path}")
        train_data.to_csv(train_path, index=False)
        
        logger.info(f"Saving preprocessed test data to: {test_path}")
        test_data.to_csv(test_path, index=False)
        
        logger.info(
            f"Preprocessed data saved successfully:\n"
            f"  Train: {train_path} ({len(train_data)} rows)\n"
            f"  Test: {test_path} ({len(test_data)} rows)"
        )
    except Exception as e:
        logger.error(f"Unexpected error occured during saving the preprocessed data: {e}", exc_info=True)
        raise
    
def main():
    """
    Main entry point for text data preprocessing.
    """
    try:
        # Get project root directory
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        
        # Define paths
        train_raw_data_path = project_root / "data" / "raw" / "train.csv"
        test_raw_data_path = project_root / "data" / "raw" / "test.csv"
        
        train_data = pd.read_csv(train_raw_data_path)
        test_data = pd.read_csv(test_raw_data_path)
        
        train_data = normalize_text_column(train_data)
        test_data = normalize_text_column(test_data)
        
        
        # Get the directory where the data will be saved
        data_dir = project_root / "data" / "clean"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        save_preprocessed_data(
            train_data=train_data,
            test_data=test_data,
            data_path=data_dir
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()