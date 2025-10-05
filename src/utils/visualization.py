import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up plotting configuration
sns.set_theme()  # Apply seaborn defaults
plt.rcParams.update({
    'figure.figsize': [12, 6],
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': '#f0f0f0',
    'figure.facecolor': 'white'
})

# Configure plot style and display settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.autolayout'] = True

# Enable real-time display in Jupyter/IPython environments
try:
    from IPython import display
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

def show_plot(name: str = None, block: bool = True) -> None:
    """
    Display and save the current plot with real-time updates.
    
    Args:
        name (str): Name to use for saving the plot file. If None, uses 'latest_plot'.
        block (bool): If True, block execution until plot window is closed.
                     If False, continue execution immediately.
    """
    try:
        # Save the plot
        filename = f"{name if name else 'latest_plot'}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
        # Clear output and display in Jupyter/IPython environments
        if JUPYTER_AVAILABLE:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        
        # Show plot in other environments
        plt.show()
        if not block:
            plt.draw()
            plt.pause(0.1)
        
        print(f"Plot saved to '{filename}'")
        
    except Exception as e:
        print(f"Note: Plot saved but could not be displayed: {e}")

# Set custom color palette
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f']
sns.set_palette(colors)

# Configure global matplotlib settings
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.grid'] = True

class Visualizer:
    @staticmethod
    def clear_current_figure():
        """Clear the current figure to prevent plot overlapping"""
        plt.close('all')  # Close all figures
        plt.clf()         # Clear current figure
        
    @staticmethod
    def plot_stock_history(df: pd.DataFrame, title: str = "Stock Price History"):
        """
        Plot the historical stock prices with volume.
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            title (str): Plot title
        """
        # Clear any existing figures
        Visualizer.clear_current_figure()
        
        # Create figure and axis with secondary y-axis
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
        
        # Main price plot
        ax1 = plt.subplot(gs[0])
        
        # Volume subplot
        ax2 = plt.subplot(gs[1], sharex=ax1)
        
        # Plot stock prices with dynamic color palette
        for i, column in enumerate(df.columns):
            if column != 'Volume':
                ax1.plot(df.index, df[column], label=column, color=colors[i % len(colors)])
        
        # Plot volume as bar chart with gradient color
        if 'Volume' in df.columns:
            volume_colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(df)))
            ax2.bar(df.index, df['Volume'], alpha=0.6, color=volume_colors, label='Volume')
        
        # Customize the plots
        ax1.set_title(title, fontsize=16, pad=20)
        ax2.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        
        # Add grid with custom style
        ax1.grid(True, alpha=0.2, linestyle='--')
        ax2.grid(True, alpha=0.2, linestyle='--')
        
        # Enhance spines
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Add legend with custom style
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
                  fancybox=True, shadow=True)
        
        # Show plot in real-time
        show_plot()

    @staticmethod
    def plot_predictions(actual: np.ndarray, predicted: np.ndarray, 
                        dates: Optional[List] = None, title: str = "Stock Price Predictions",
                        asset_name: str = "Portfolio"):
        """
        Plot actual vs predicted values for multiple assets with improved styling.
        
        Args:
            actual (np.ndarray): Actual values for all assets
            predicted (np.ndarray): Predicted values for all assets
            dates (Optional[List]): List of dates for x-axis
            title (str): Plot title
            asset_name (str): Name of the portfolio
        """
        # Clear any existing figures
        Visualizer.clear_current_figure()
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        feature_names = ['AMZN', 'DPZ', 'BTC', 'NFLX']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f']
        
        for idx, (ax, feature, color) in enumerate(zip(axes.flat, feature_names, colors)):
            actual_feature = actual[:, idx]
            predicted_feature = predicted[:, idx]
            
            if dates is not None:
                ax.plot(dates, actual_feature, label=f'Actual {feature}', color=color, linewidth=2)
                ax.plot(dates, predicted_feature, label=f'Predicted {feature}', 
                       linestyle='--', color=color, alpha=0.7, linewidth=2)
                
                # Add error bands
                ax.fill_between(dates, 
                              predicted_feature,
                              actual_feature,
                              color=color,
                              alpha=0.1)
            else:
                ax.plot(actual_feature, label=f'Actual {feature}', color=color, linewidth=2)
                ax.plot(predicted_feature, label=f'Predicted {feature}', 
                       linestyle='--', color=color, alpha=0.7, linewidth=2)
            
            # Calculate metrics for this feature
            mse = mean_squared_error(actual_feature, predicted_feature)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_feature, predicted_feature)
            r2 = r2_score(actual_feature, predicted_feature)
            
            metrics_text = f'RMSE: ${rmse:.2f}\nMAE: ${mae:.2f}\nR²: {r2:.4f}'
            ax.text(0.02, 0.95, metrics_text,
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{feature} Price Prediction', fontsize=12)
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Price', fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{title}\n{asset_name}', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Show plot in real-time
        show_plot()
    
    @staticmethod
    def plot_training_history(history: dict, asset_name: str = "Stock"):
        """
        Plot training history with improved styling.
        
        Args:
            history (dict): Training history dictionary
            asset_name (str): Name of the asset being predicted
        """
        # Clear any existing figures
        Visualizer.clear_current_figure()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Loss plot
        ax1 = plt.subplot(gs[0])
        
        # Learning rate plot
        ax2 = plt.subplot(gs[1])
        
        # Plot training and validation loss
        ax1.plot(history['loss'], label='Training Loss', color='#3498db')
        ax1.plot(history['val_loss'], label='Validation Loss', color='#e74c3c')
        
        ax1.set_title('LSTM Model Training History', fontsize=14, pad=20)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(fancybox=True, shadow=True)
        
        # Add final loss values as text
        final_train_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        best_val_loss = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_val_loss)
        
        loss_text = (f'Final Training Loss: {final_train_loss:.4f}\n'
                    f'Final Validation Loss: {final_val_loss:.4f}\n'
                    f'Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch+1})')
        
        ax1.text(0.02, 0.98, loss_text,
                transform=ax1.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5))
        
        # Plot learning rate if available
        if 'learning_rate' in history:
            ax2.plot(history['learning_rate'], color='#2ecc71', label='Learning Rate')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_yscale('log')  # Use log scale for learning rate
            ax2.legend(fancybox=True, shadow=True)
            
        # Enhance spines
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2, linestyle='--')
        
        # Show plot in real-time
        show_plot()
    
    @staticmethod
    def plot_feature_importance(feature_names: List[str], importances: np.ndarray):
        """
        Plot feature importance.
        
        Args:
            feature_names (List[str]): List of feature names
            importances (np.ndarray): Feature importance scores
        """
        # Clear any existing figures
        Visualizer.clear_current_figure()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=feature_names)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)  # Small pause to ensure plot is shown

class Evaluator:
    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
        """
        Calculate regression metrics.
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    @staticmethod
    def print_metrics(metrics: dict):
        """
        Print evaluation metrics.
        
        Args:
            metrics (dict): Dictionary containing evaluation metrics
        """
        print("\nModel Performance Metrics:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
    @staticmethod
    def show_plot(name: str = None, block: bool = True) -> None:
        """
        Display and save the current plot.
        
        Args:
            name (str): Name to use for saving the plot file. If None, uses 'latest_plot'.
            block (bool): If True, block execution until plot window is closed.
                         If False, continue execution immediately.
        """
        try:
            # First save the plot
            filename = f"{name if name else 'latest_plot'}.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Plot saved to '{filename}'")
            
            # Then try to display it
            plt.show(block=block)
            if not block:
                plt.pause(1)  # Longer pause to ensure plot is visible
        except Exception as e:
            print(f"Note: Plot saved but could not be displayed: {e}")
        finally:
            plt.close('all')  # Clean up to prevent memory leaks

    @staticmethod
    def calculate_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate directional accuracy of predictions.
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            
        Returns:
            float: Directional accuracy score
        """
        actual_direction = np.diff(actual.flatten())
        predicted_direction = np.diff(predicted.flatten())
        
        correct_direction = np.sum((actual_direction > 0) == (predicted_direction > 0))
        total_predictions = len(actual_direction)
        
        return correct_direction / total_predictions