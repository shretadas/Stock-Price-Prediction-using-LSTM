import tensorflow as tf
import numpy as np
from typing import Tuple, List

# Import Keras components
from keras.models import Sequential, Model
from keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                         Input, Concatenate, Add, LayerNormalization,
                         GlobalAveragePooling1D, Bidirectional)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import custom callback
from src.utils.callbacks import LivePlotCallback

class LSTMModel:
    def __init__(self, sequence_length: int, n_features: int, units: List[int] = [50, 50],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize the LSTM model.
        
        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of features
            units (List[int]): List of units in each LSTM layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """
        Build and compile an advanced LSTM model with residual connections
        and bidirectional layers.
        
        Returns:
            Model: Compiled Keras model
        """
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Normalize inputs
        x = LayerNormalization()(inputs)
        
        # First Bidirectional LSTM layer
        lstm1 = Bidirectional(LSTM(
            units=self.units[0],
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)
        ))(x)
        lstm1 = LayerNormalization()(lstm1)
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        
        # Second Bidirectional LSTM layer
        lstm2 = Bidirectional(LSTM(
            units=self.units[0],
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)
        ))(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        lstm2 = Dropout(self.dropout_rate)(lstm2)
        
        # Residual connection
        lstm2_with_residual = Add()([lstm1, lstm2])
        
        # Final LSTM layer
        final_lstm = Bidirectional(LSTM(
            units=self.units[1],
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)
        ))(lstm2_with_residual)
        final_lstm = LayerNormalization()(final_lstm)
        final_lstm = Dropout(self.dropout_rate)(final_lstm)
        
        # Dense layers for prediction
        dense1 = Dense(self.units[1] * 2, activation='relu')(final_lstm)
        dense1 = LayerNormalization()(dense1)
        dense1 = Dropout(self.dropout_rate/2)(dense1)
        
        # Output layer for asset price predictions
        outputs = Dense(units=4, name='price_predictions')(dense1)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> tf.keras.callbacks.History:
        """
        Train the LSTM model with advanced callbacks.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        self.n_features = X_train.shape[2]  # Update n_features based on actual input
        self.model = self._build_model()  # Rebuild model with correct input shape
        # Early stopping with patience
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate reduction on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Model checkpointing
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='auto'
        )
        
        # Live plotting callback
        live_plot = LivePlotCallback(asset_name="Portfolio")
        
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, checkpoint, live_plot],
            verbose=1,
            shuffle=True
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath, save_format='keras')
    
    @classmethod
    def load_model(cls, filepath: str) -> Sequential:
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Sequential: Loaded Keras model
        """
        return tf.keras.models.load_model(filepath)