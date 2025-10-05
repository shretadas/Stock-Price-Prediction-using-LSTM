import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

class LivePlotCallback(tf.keras.callbacks.Callback):
    """Callback to update training history plot in real-time during training."""
    
    def __init__(self, asset_name: str = "Stock"):
        super().__init__()
        self.asset_name = asset_name
        self.history = {'loss': [], 'val_loss': [], 'learning_rate': []}
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Update history
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['learning_rate'].append(
            float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        )
        
        # Clear the output and create a new figure
        clear_output(wait=True)
        plt.figure(figsize=(12, 4))
        
        # Plot training history
        epochs_range = range(1, len(self.history['loss']) + 1)
        plt.plot(epochs_range, self.history['loss'], 'b-', label='Training Loss')
        plt.plot(epochs_range, self.history['val_loss'], 'r-', label='Validation Loss')
        
        # Customize the plot
        plt.title(f'{self.asset_name} Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Display the plot
        plt.show()
        plt.close()