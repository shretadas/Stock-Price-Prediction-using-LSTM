import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
from src.utils.technical_indicators import TechnicalIndicators

class DataProcessor:
    def __init__(self, sequence_length: int = 60):
        """
        Initialize the DataProcessor class.
        
        Args:
            sequence_length (int): Number of time steps to use for sequence prediction
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.technical_indicators = TechnicalIndicators()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess the CSV data.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        # Load the data with the date index
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # Sort index to ensure chronological order
        df.sort_index(inplace=True)
        # Forward fill any missing values
        df.fillna(method='ffill', inplace=True)
        # Backward fill any remaining missing values
        df.fillna(method='bfill', inplace=True)
        return df
        

    
    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str], 
                     test_split: float = 0.2
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            feature_columns (List[str]): List of feature column names
            test_split (float): Proportion of data to use for testing
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training and testing data
        """
        if len(df) <= self.sequence_length:
            raise ValueError(f"Not enough data points. Need more than {self.sequence_length} rows.")
            
                # Add technical indicators
        df_with_indicators = TechnicalIndicators.add_all_indicators(df, feature_columns)
        
        # Get all feature columns (original + indicators)
        all_feature_columns = [col for col in df_with_indicators.columns 
                             if any(asset in col for asset in feature_columns)]
        
        # Prepare features and targets
        feature_data = df_with_indicators[all_feature_columns].values
        target_data = df[feature_columns].values  # Original asset prices as targets
        
        # Scale the data
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        scaled_targets = self.target_scaler.fit_transform(target_data)
        
        # Create sequences for X and y
        X, y = [], []
        for i in range(len(scaled_features) - self.sequence_length):
            feature_seq = scaled_features[i:(i + self.sequence_length)]
            target_val = scaled_targets[i + self.sequence_length]
            X.append(feature_seq)
            y.append(target_val)
            
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        train_size = int(len(X) * (1 - test_split))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled target values.
        
        Args:
            data (np.ndarray): Scaled data
            
        Returns:
            np.ndarray: Original scale data
        """
        return self.target_scaler.inverse_transform(data)