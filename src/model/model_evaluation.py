import logging
import joblib
import yaml

import pandas as pd

from pathlib import Path
from pandas import DataFrame, Series
from typing import Any, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)
logger = logging.getLogger(__name__)

# Configure Paths
def load_config(
    config_path = "config.yaml"
):
    """
    Function to load path configuration from a YAML file
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


"""
load_data -> load_model -> load_params -> evaluate_model
"""

def load_data(
    data_path: str
) -> DataFrame:
    """
    Load the data from a specified CSV path.
    
    Args:
        data_path (str): Location to the test dataset (CSV file).
        
    Returns:
        DataFrame: The dataframe containing test data.
        
    Raises:
        FileNotFoundError: If the file specified at certain location does not exist.
        ValueError: If the file is successfully read, but results in an empty DataFrame.
        Exception: For other unexpected exceptions.
    """
    path = Path(data_path)
    try:
        logger.info(f"Reading the test data from: {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"File not found at the specified location: {path}")
        
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        
        if df.empty:
            raise ValueError(f"Read DataFrame is empty")
        
        logger.info("Data successfully read..."
                   f"Dimensions: {len(df)} rows and {len(df.columns)} columns.")
        return df
    except FileNotFoundError as e:
        logger.error(f"File Not Found Error: {e}", exc_info=False)
        raise
    except ValueError as e:
        logger.error(f"Data Validation Error: {e}", exc_info=False)
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Pandas Parser Error while loading the data: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected Error: {e}", exc_info=True)
        raise
    
def load_model(
    model_path
) -> Any:
    """
    Loads the trained model from the given path using joblib.
    
    Args:
        model_path (str): Location to where the model resides.
        
    Returns:
        Any: The loaded model object.
        
    Raises:
        FileNotFoundError: If the model file is not found.
        Exception: For other unrelated exceptions.
    """
    path = Path(model_path)
    try:
        logger.info("Loading the model has started...")
        if not path.exists():
            raise FileNotFoundError(f"File Not found at: {path}")
        
        model = joblib.load(path)
    
        logger.info("Loading the model has complete...")
        return model
    except FileNotFoundError as e:
        logger.error(f"File Not Found Error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error("Unexpected error: {e}", exc_info=True)
        raise
    
def load_vectorizer(
    vectorizer_path: str
) -> TfidfVectorizer:
    """
    Loads the Tfidf vectorized from given path using joblib.
    
    Args:
        vectorizer_path (str): Location to where the vectorized resides.
    
    Returns:
        TfidfVectorizer:
    
    Raises:

        Exception: For other unrelated exceptions.
    """
    path = Path(vectorizer_path)
    try:
        logger.info(f"Loading vectorizer from: {path}")
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {path}")
        vectorizer = joblib.load(path)
        logger.info(f"Loading vectorizer complete.")
        return vectorizer
    except FileNotFoundError as e:
        logger.error(f"File Not Found Error: {e}", exc_info=False)
        raise
    except Exception as e:
        logger.error("Unexpected error: {e}", exc_info=True)
        raise

def evaluate_model(
    model: Any, 
    X_test: DataFrame,
    y_test: Series
) -> Dict[str, float]:
    """
    Evaluate the model on test data and return a dictionary of common metrics.
    
    Args:
        model (Any): The trained machine learning model (e.g., a scikit-learn estimator).
        X_test (DataFrame): The feature data for evaluation.
        y_test (Series): The true target values (labels) for the test data.
        
    Returns:
        Dict[str, float]: A dictionary containing the accuracy, precision, recall and f1_score.
        
    Raises:
        Exception: For error occured during metric calculation.
    """
    try:
        logger.info(f"Starting model evaluation...")
        
        y_predict = model.predict(X_test)
        
        metrics = {
            "accuracy_score": accuracy_score(y_test, y_predict),
            "precision_score": precision_score(y_test, y_predict, average='weighted', zero_division=0),
            "recall_score": recall_score(y_test, y_predict, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_predict, average='weighted', zero_division=0)
        }
        
        logger.info("Model Evaluation complete.")
        logger.info(f"{metrics}")
        
        clf_report = classification_report(y_test, y_predict, zero_division=0)
        logger.info(f"\n--- Classication report ---\n{clf_report}")
        
        cm = confusion_matrix(y_test, y_predict)
        logger.info(f"\n--- Confusion Matrix ---\n{cm}")
                
        return metrics
    except Exception as e:
        logger.error(f"Unexpected Error occured: {e}", exc_info=True)
        raise
    
def main():
    
    # 0. Load configuration
    config = load_config()
    
    data_path = config['test_data_path']
    model_path = config['model_path']
    vectorizer_path = config['vectorizer_path']
    FEATURE_COL = config['FEATURE_COL']
    LABEL_COL = config['LABEL_COL']
    
    # 1. Load the data
    test_df  = load_data(data_path=data_path)
    X_test = test_df[FEATURE_COL]
    y_test = test_df[LABEL_COL] + 1

    # 2. Load the model
    model = load_model(model_path=model_path)
    
    # 3. Load the vectorizer
    vectorizer = load_vectorizer(vectorizer_path=vectorizer_path)
    X_test_vec = vectorizer.transform(X_test)
    
    # 4. Evaluate model
    scores = evaluate_model(
        model= model, 
        X_test=X_test_vec, 
        y_test=y_test
    )
    
if __name__ == "__main__":
    main()