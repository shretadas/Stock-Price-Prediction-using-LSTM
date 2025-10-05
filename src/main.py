import os
import pandas as pd
import numpy as np
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor
from src.model import LSTMModel
from src.utils.visualization import Visualizer, Evaluator

def main():
    # Configuration
    SEQUENCE_LENGTH = 60  # Increased lookback period for better context
    TEST_SPLIT = 0.2     # Proportion of data for testing
    EPOCHS = 100         # Increased epochs for better convergence
    BATCH_SIZE = 32
    LSTM_UNITS = [128, 64]  # Increased model capacity
    DROPOUT_RATE = 0.2   # Added dropout for regularization
    TARGET_ASSET = 'AMZN'  # Target asset to predict
    
    # Initialize data processor
    data_processor = DataProcessor(sequence_length=SEQUENCE_LENGTH)
    
    # Load and preprocess data
    df = data_processor.load_data('data/portfolio_data.csv')
    
    # Define features (all will be used as targets too)
    feature_columns = ['AMZN', 'DPZ', 'BTC', 'NFLX']  # All available assets
    # Create 'plots' directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot historical data
    print(f"\nVisualizing historical data for portfolio assets...")
    Visualizer.plot_stock_history(df[feature_columns], title="Portfolio Assets Historical Prices")
    Visualizer.show_plot('plots/historical_prices', block=True)
    
    # Prepare data for training
    X_train, X_test, y_train, y_test = data_processor.prepare_data(
        df, feature_columns=feature_columns, test_split=TEST_SPLIT
    )
    
    # Initialize and train model
    model = LSTMModel(
        sequence_length=SEQUENCE_LENGTH,
        n_features=len(feature_columns),
        units=LSTM_UNITS
    )
    
    # Train the model
    history = model.train(
        X_train, y_train,
        X_test, y_test,  # Using test set as validation
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Inverse transform predictions
    train_predictions = data_processor.inverse_transform_target(train_predictions)
    test_predictions = data_processor.inverse_transform_target(test_predictions)
    y_train_orig = data_processor.inverse_transform_target(y_train)
    y_test_orig = data_processor.inverse_transform_target(y_test)
    
    # Evaluate model
    train_metrics = Evaluator.calculate_metrics(y_train_orig, train_predictions)
    test_metrics = Evaluator.calculate_metrics(y_test_orig, test_predictions)
    
    print("\nTraining Set Metrics:")
    Evaluator.print_metrics(train_metrics)
    print("\nTest Set Metrics:")
    Evaluator.print_metrics(test_metrics)
    
    # Calculate directional accuracy
    train_da = Evaluator.calculate_directional_accuracy(y_train_orig, train_predictions)
    test_da = Evaluator.calculate_directional_accuracy(y_test_orig, test_predictions)
    print(f"\nDirectional Accuracy:")
    print(f"Training Set: {train_da:.2%}")
    print(f"Test Set: {test_da:.2%}")
    
    # Visualize results
    print(f"\nPlotting training history...")
    Visualizer.plot_training_history(history.history, asset_name=TARGET_ASSET)
    
    # Plot predictions
    print(f"\nPlotting predictions...")
    # Calculate correct date ranges for predictions
    train_dates = df.index[SEQUENCE_LENGTH:SEQUENCE_LENGTH+len(train_predictions)]
    test_dates = df.index[-len(test_predictions):]
    
    # Plot training predictions
    Visualizer.plot_predictions(
        y_train_orig, train_predictions,
        dates=train_dates,
        title="Stock Price Prediction - Training Set",
        asset_name=TARGET_ASSET
    )
    
    # Plot test predictions
    Visualizer.plot_predictions(
        y_test_orig, test_predictions,
        dates=test_dates,
        title="Stock Price Prediction - Test Set",
        asset_name=TARGET_ASSET
    )
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save_model('models/lstm_model.h5')

if __name__ == "__main__":
    main()