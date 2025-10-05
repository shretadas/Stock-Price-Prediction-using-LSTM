# 📈 Stock Market Predictor with LSTM

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://tensorflow.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/shretadas/Stock-Price-Prediction-using-LSTM/graphs/commit-activity)

> 🚀 A deep learning approach to predict stock market prices using LSTM (Long Short-Term Memory) networks.

## � Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements an LSTM-based deep learning model to predict stock market prices. We analyze historical data from multiple assets (AMZN, DPZ, BTC, NFLX) to forecast future price movements, helping investors make data-driven decisions.

### ✨ Key Features

- 🔮 Multi-asset price prediction with LSTM networks
- 📊 Real-time visualization of training progress
- 📈 Comprehensive performance metrics and analysis
- 🧪 Interactive Jupyter notebook implementation
- 🔍 Robust data preprocessing pipeline
- 🤖 Advanced LSTM architecture with dropout layers

## 📈 Visualizations

### Historical Price Data
![Historical Prices](images/historical_prices.png)
*Historical price trends for all assets in the portfolio*

### Training Progress
![Training Progress](images/training_progress.png)
*Model training progress showing loss reduction over epochs*

### Prediction Results
![Predictions](images/predictions.png)
*Actual vs Predicted prices for target asset*

## 🛠️ Technical Architecture

### Data Processing Pipeline
```
Load Data → Preprocess → Create Sequences → Scale → Train/Test Split
```

### LSTM Model Architecture
- Input Layer: Sequence length of 60 timesteps
- LSTM Layer 1: 128 units with dropout
- LSTM Layer 2: 64 units with dropout
- Dense Output Layer: Multi-asset prediction

## 📝 Implementation Details

### Data Preprocessing
- Handles missing values automatically
- Implements MinMax scaling
- Creates sliding window sequences
- Performs train-test splitting

### Model Configuration
- Sequence Length: 60 days
- Test Split: 20%
- Batch Size: 32
- Learning Rate: Adaptive (with ReduceLROnPlateau)
- Early Stopping: Patience of 15 epochs

## 🚀 Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shretadas/Stock-Price-Prediction-using-LSTM.git
   cd Stock-Price-Prediction-using-LSTM
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook Stock_Price_Prediction.ipynb
   ```

## 📊 Results & Visualizations

### Portfolio Historical Data
![Portfolio History](images/portfolio_history.png)
*Historical price trends across our portfolio of assets*

### Training Performance
![Training History](images/training_history.png)
*Model training history showing loss convergence over epochs*

### Prediction Analysis
![Predictions](images/predictions.png)
*Comparison between predicted and actual stock prices*

### Latest Market Analysis
![Latest Analysis](images/latest_plot.png)
*Most recent market predictions and analysis*

Our LSTM model demonstrates:
- Low Mean Squared Error (MSE) across predictions
- Accurate trend predictions for multiple assets
- Robust performance across different market conditions
- Reliable pattern recognition in volatile markets

## 🔧 Technologies Used

- TensorFlow 2.x
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 📁 Project Structure

```
Stock-Price-Prediction-using-LSTM/
│
├── data/
│   └── portfolio_data.csv
│
├── images/
│   ├── historical_prices.png
│   ├── training_progress.png
│   └── predictions.png
│
├── models/
│   └── best_model.h5
│
├── Stock_Price_Prediction.ipynb
├── requirements.txt
└── README.md
```

## ✨ Features

- [x] Multi-asset price prediction
- [x] Real-time training visualization
- [x] Automatic data preprocessing
- [x] Model checkpoint saving
- [x] Performance visualization
- [x] Error analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/shretadas">Shreta Das</a>
  <br>
  <br>
  <p>If you find this project helpful, please give it a ⭐!</p>
</div>