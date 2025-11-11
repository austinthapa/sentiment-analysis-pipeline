import os
import joblib
import logging
import lightgbm
import yaml
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Any, Dict
from pandas import DataFrame, Series
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

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
load_data -> apply_vectorization -> load_params -> train_model -> save_artifacts
"""

def load_data(
    data_path: str
) -> DataFrame:
    """
    Load the preprocessed data from  a CSV file.
    
    Args:
        data_path (str): The file path to the CSV data.
        
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
        
    Raises:
        FileNotFoundError: If the file is not found at given location.
        ValueError: If the CSV file is empty after reading.
        pd.errors.ParserError: If parsing error occurs while loading the data.
        Exception: For other related exceptions.
    """
    logger.info(f"Reading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found at given location: {data_path}")
    try:
        df = pd.read_csv(data_path)
        df.dropna(inplace=True)  # Test
        
        # Check for empty DataFrame
        if df.empty:
            raise ValueError("Read CSV file is empty.")
        logger.info("Data successfully read..."
                    f"DataFrame: {len(df)} rows & {len(df.columns)} columns")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error while loading the data: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error occured: {e}", exc_info=True)
        raise

def apply_vectorization(
    X_train: DataFrame,
    max_features: int = 1_000,
    ngram_range: Tuple[int, int] = (1, 1)
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Apply TF-IDF vectorization to the input training text data
    and returns the transformed data and fitted vectorizer.
    
    Args:
        X_train (DataFrame): The input training dataframe.
        max_features (int): The number of max features to use for vectorization.
        ngram_range (Tuple[int, int]): The lower and upper boundary of the ngram range.
    
    Returns:
        Tuple[csr_matrix, TfidfVectorizer]:
            - X_train_vec: Transformed training data in a sparse matrix format.
            - vectorizer: The fitted TfidfVectorizer.
    
    Raises:
        ValueError: If the input dataframe is empty.
        Exception: For other related exception.
    """
    if X_train.empty:
        raise ValueError("Training input dataframe is empty")
    try:
        logger.info("Performing TF-IDF Vectorization...")
        vectorier = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )
        X_train_vec = vectorier.fit_transform(X_train).toarray()
        logger.info("Vectorization successfully complete..."
                    f"\nOutput shape: {X_train.shape}")

        return X_train_vec, vectorier
    except Exception as e:
        logger.error(f"Unexpected Error during vectorization: {e}", exc_info=True)
        raise

def load_params(
    file_name: str = "params.yaml"
) -> Dict[str, Any]:
    """
    Load the configuration parameters from a YAML file relative to current directory.
    
    Args:
        file_name (str): The name of YAML file.
    
    Returns:
        Dict[str, Any]: A dictionary containing the loaded parameters.
    
    Raises:
        FileNotFoundError: If the parameters file cannot be found at expected path.
        yaml.YAMLError: If the file is not found but contains invalid YAML syntax.
        Exception: For other related exceptions.
    """
    try:
        current_dir = Path(__file__).resolve().parent
        root_dir = current_dir.parent.parent
        param_path = root_dir / "params.yaml"
    
        if not param_path.exists():
            raise FileNotFoundError(f"Parameters file not found at: {param_path}")
        logger.info(f"Loading parameters from: {param_path}")
        
        with open(param_path, "r") as file:
            params = yaml.safe_load(file)["model"]
        
        if not params:
            logger.warning("Loaded parameters dictionary is empty or invalid")
            params = {}
        logger.info("Parameters successfully loaded.")
        return params
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error while loading the params: {e}", exc_info=True)
        raise

def train_model(
    X_train: csr_matrix,
    y_train: Series,
    **params: Any
)-> lightgbm.LGBMClassifier:
    """
    Trains the LightGBM model with updated parameters.
    
    This function intialized the LightGBM classifier with the provided parameters,
    fits it to the sparse training data, returns the fitted model object.
    
    Args:
        X_train (csr_matrix): The sparse input matrix.
        y_train (Series): The input labels.
        **params: Arbitrary keyword arguments passed directly to the LGBM Classifier constructor.
        
    Returns:
        lightgbm.LGBMClassifier: The fully trained and fitted LightGBM model object.
        
    Raises:
        Exception: If any errors occurs during model initialization or training. 
    """
    try:
        logger.info(f"Training the model with params: {params}")
                
        # Train the model
        logger.info("Starting model training")
        
        mlflow.set_experiment(experiment_name="Light_GBM_Experiment")
        mlflow.set_tracking_uri("/Users/anilthapa/sentiment-analysis-pipeline/mlruns")
        
        LGBMClassifier = lightgbm.LGBMClassifier(
            **params
        )
        
        with mlflow.start_run(run_name="Light_GBM_Model"):
            mlflow.log_params(params)
            
            LGBMClassifier.fit(X_train, y_train)
            
            y_predict = LGBMClassifier.predict(X_train)
            
            mlflow.lightgbm.log_model(
                lgb_model=LGBMClassifier, 
                input_example=X_train[:10],
                registered_model_name="LGBM_Classifier"
            )
            
            mlflow.log_metrics({
                "accuracy_score_train": accuracy_score(y_train, y_predict),
                "precision_score_train": precision_score(y_train, y_predict, average = 'macro'),
                "recall_score_train": recall_score(y_train, y_predict, average = 'macro'),
                "f1_score_train": f1_score(y_train, y_predict, average = 'macro')
            })
            
            c_matrix = confusion_matrix(y_train, y_predict)
            c_matrix_disp = ConfusionMatrixDisplay(c_matrix, display_labels=LGBMClassifier.classes_)
            c_matrix_disp.plot(cmap="Blues")
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()
            
            clf_report = classification_report(y_train, y_predict)
            with open("classification_report.txt", "w") as f:
                f.write(clf_report)
            mlflow.log_artifact("classification_report.txt")
            logger.info("Training model completed and logged successfully to MLflow.")
            
        return LGBMClassifier
    except Exception as e:
        logger.error(f"Unexpected while training the model: {e}", exc_info=True)
        raise

def save_artifacts(
    model: Any,
    vectorizer: Any,
    model_name: str,
    vectorizer_name: str
) -> Path:
    """
    Saves a trained machine learning model object using joblib.
    
    The model is saved to a file named f"{model_name}.joblib" inside the specified directory.
    
    Args:
        model (Any): The trained model object (e.g., lightgbm.LightGBMClassifier).
        model_name (str): The base name for the output file (e.g., 'final_lightgbm')
        outout_dir (str): The directory where the model will be saved.
    
    Returns:
        Path: the absolute path of the saved model file.
    
    Raises:
        Exception: If any error occurs while saving the model.
    """
    try:
        current_dir = Path(__file__).resolve().parent
        root_dir = current_dir.parent.parent
        
        output_dir_path = root_dir / "artifacts"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # --- Save the model ---
        model_file_name = f"{model_name}.joblib"
        model_full_path = output_dir_path / model_file_name
        
        logger.info(f"Saving model to: {model_file_name}")
        joblib.dump(model, model_full_path)
        
        # --- Save the vectorzer ---
        vectorizer_file_name = f"{vectorizer_name}.joblib"
        vectorizer_full_path = output_dir_path / vectorizer_file_name
        
        logger.info(f"Saving artifact to: {vectorizer_full_path}")
        joblib.dump(vectorizer, vectorizer_full_path)
    
        return model_full_path
    except Exception as e:
        logger.error(f"Unexpected error while saving model: {e}", exc_info=True)
        raise
    
def main():
    
    # 0. Load the configuration
    config = load_config()
    data_path = config["train_data_path"]
    FEATURE_COL = config['FEATURE_COL']
    LABEL_COL = config['LABEL_COL']
    
    # 1. Load the data
    df = load_data(data_path=data_path)
    
    # Extract the features and labels.
    X_train = df[FEATURE_COL]
    y_train = df[LABEL_COL] + 1
    
    # 2. Perform vectorization
    X_train_vec, vectorizer = apply_vectorization(
        X_train=X_train,
        max_features=1_000,
        ngram_range=(1, 1)
    )
    
    # 3. Load the params from params.yaml
    params = load_params()
    
    # 4. Train the model.
    LGBMClassifier = train_model(
        X_train=X_train_vec,
        y_train=y_train,
        **params
    )
    
    # 5. Dump the vectorizer and model in a pickle file.
    save_artifacts(
        model=LGBMClassifier,
        vectorizer=vectorizer,
        model_name="light_gbm_clf",
        vectorizer_name="Tfidf_vectorizer"
    )
    logger.info("Training pipeline successfully executed. Artifacts saved successfully.")
    
    
if __name__ == "__main__":
    main()