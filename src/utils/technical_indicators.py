import numpy as np
import pandas as pd
from typing import List, Optional

class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data (pd.Series): Price series
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9
                     ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data (pd.Series): Price series
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            
        Returns:
            tuple[pd.Series, pd.Series, pd.Series]: MACD line, signal line, and histogram
        """
        exp1 = data.ewm(span=fast_period, adjust=False).mean()
        exp2 = data.ewm(span=slow_period, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, 
                                std_dev: float = 2.0
                               ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data (pd.Series): Price series
            period (int): Moving average period
            std_dev (float): Number of standard deviations
            
        Returns:
            tuple[pd.Series, pd.Series, pd.Series]: Upper band, middle band, lower band
        """
        middle_band = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()
        
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_ema(data: pd.Series, periods: List[int] = [5, 10, 20]
                    ) -> List[pd.Series]:
        """
        Calculate Multiple Exponential Moving Averages.
        
        Args:
            data (pd.Series): Price series
            periods (List[int]): List of periods for EMAs
            
        Returns:
            List[pd.Series]: List of EMA series
        """
        return [data.ewm(span=period, adjust=False).mean() for period in periods]
    
    @staticmethod
    def calculate_volume_indicators(price: pd.Series, volume: pd.Series,
                                 period: int = 20) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Volume-based indicators (OBV and Volume MA).
        
        Args:
            price (pd.Series): Price series
            volume (pd.Series): Volume series
            period (int): Period for moving averages
            
        Returns:
            tuple[pd.Series, pd.Series]: On-Balance Volume (OBV) and Volume MA
        """
        # Calculate On-Balance Volume (OBV)
        price_change = price.diff()
        obv = pd.Series(index=price.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(price)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        # Calculate Volume Moving Average
        volume_ma = volume.rolling(window=period).mean()
        
        return obv, volume_ma
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with price and volume data
            feature_columns (List[str]): List of asset columns
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        df_with_indicators = df.copy()
        
        for asset in feature_columns:
            # RSI
            df_with_indicators[f'{asset}_RSI'] = TechnicalIndicators.calculate_rsi(df[asset])
            
            # MACD
            macd, signal, hist = TechnicalIndicators.calculate_macd(df[asset])
            df_with_indicators[f'{asset}_MACD'] = macd
            df_with_indicators[f'{asset}_MACD_Signal'] = signal
            df_with_indicators[f'{asset}_MACD_Hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(df[asset])
            df_with_indicators[f'{asset}_BB_Upper'] = upper
            df_with_indicators[f'{asset}_BB_Middle'] = middle
            df_with_indicators[f'{asset}_BB_Lower'] = lower
            
            # EMAs
            emas = TechnicalIndicators.calculate_ema(df[asset])
            for period, ema in zip([5, 10, 20], emas):
                df_with_indicators[f'{asset}_EMA_{period}'] = ema
            
            # Volume Indicators (if volume is available)
            if f'{asset}_Volume' in df.columns:
                obv, vol_ma = TechnicalIndicators.calculate_volume_indicators(
                    df[asset], df[f'{asset}_Volume']
                )
                df_with_indicators[f'{asset}_OBV'] = obv
                df_with_indicators[f'{asset}_Volume_MA'] = vol_ma
        
        return df_with_indicators.fillna(method='bfill').fillna(method='ffill')