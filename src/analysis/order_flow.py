import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class OrderFlowImbalance:
    """Represents an order flow imbalance."""
    price: float
    volume: float
    side: str  # 'bid' or 'ask'
    strength: float  # 0.0 to 1.0
    time: pd.Timestamp

@dataclass
class LiquidityZone:
    """Represents a significant liquidity zone."""
    price_start: float
    price_end: float
    volume: float
    type: str  # 'bid_wall', 'ask_wall', 'absorption', 'distribution'
    confidence: float

class OrderFlowAnalyzer:
    """Analyzes order flow patterns in both L1 and L2 data."""
    
    def __init__(self, use_level2: bool = False):
        """Initialize analyzer.
        
        Args:
            use_level2: Whether level 2 data is available
        """
        self.use_level2 = use_level2
        
    def _analyze_level1_footprint(self, df: pd.DataFrame) -> Dict:
        """Analyze institutional footprints using only L1 data."""
        # Base calculations
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        
        # === Volume Analysis ===
        vol_mean = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        
        # Institutional threshold (dynamic based on recent volatility)
        vol_volatility = vol_std / vol_mean
        threshold_multiplier = 2 + vol_volatility  # Adaptive threshold
        large_volume_threshold = vol_mean + (threshold_multiplier * vol_std)
        df['institutional_volume'] = df['volume'] > large_volume_threshold
        
        # === Smart Money Index ===
        # First hour typically retail, last hour typically institutional
        df['hour'] = pd.to_datetime(df.index).hour
        first_hour = df[df['hour'] == 9]['volume'].mean()  # Assuming market opens at 9
        last_hour = df[df['hour'] == 15]['volume'].mean()  # Assuming market closes at 16
        smart_money_ratio = last_hour / first_hour if first_hour > 0 else 1.0
        
        # === Volume Profile ===
        # Price levels where most volume occurs
        price_bins = pd.qcut(df['typical_price'], q=10)
        volume_profile = df.groupby(price_bins)['volume'].sum()
        high_volume_nodes = volume_profile[volume_profile > volume_profile.mean() + volume_profile.std()]
        
        # === Delta & Imbalance Analysis ===
        # Enhanced delta calculation considering price movement
        price_move = df['close'] - df['open']
        volume_intensity = df['volume'] / vol_mean
        df['weighted_delta'] = np.where(
            df['close'] > df['open'],
            df['volume'] * (1 + abs(price_move/df['open'])),  # Stronger signal for larger moves
            -df['volume'] * (1 + abs(price_move/df['open']))
        )
        df['cum_delta'] = df['weighted_delta'].rolling(20).sum()
        
        # === Price Action Analysis ===
        df['spread'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # === Pattern Detection ===
        # Absorption (high volume with small range)
        avg_spread = df['spread'].rolling(20).mean()
        df['absorption'] = (
            (df['volume'] > large_volume_threshold) & 
            (df['spread'] < avg_spread * 0.7)
        )
        
        # Stopping Volume (price rejection with high volume)
        df['stopping_volume'] = (
            (df['volume'] > large_volume_threshold) &
            (
                ((df['close'] > df['open']) & (df['lower_wick'] > df['body'] * 1.5)) |
                ((df['close'] < df['open']) & (df['upper_wick'] > df['body'] * 1.5))
            )
        )
        
        # Distribution/Accumulation
        df['distribution'] = (
            (df['volume'] > vol_mean * 1.2) &
            (df['high'] < df['high'].shift()) &
            (df['close'] < df['open']) &
            (df['upper_wick'] > df['body'])
        )
        
        df['accumulation'] = (
            (df['volume'] > vol_mean * 1.2) &
            (df['low'] > df['low'].shift()) &
            (df['close'] > df['open']) &
            (df['lower_wick'] > df['body'])
        )
        
        # === Recent Analysis ===
        recent = df.tail(20)
        inst_bars = recent[recent['institutional_volume']]
        
        # Directional bias with volume weighting
        bullish_volume = inst_bars[inst_bars['close'] > inst_bars['open']]['volume'].sum()
        bearish_volume = inst_bars[inst_bars['close'] < inst_bars['open']]['volume'].sum()
        
        # Detect climax volumes (potential exhaustion)
        is_climax = recent['volume'].iloc[-1] > recent['volume'].max() * 0.8
        
        # Detect stealth accumulation/distribution
        stealth_mode = (
            (recent['volume'] > vol_mean * 1.2).sum() > 10 and  # Consistent high volume
            abs(recent['close'].iloc[-1] - recent['close'].iloc[0]) < recent['std'].mean()  # Small price range
        )
        
        return {
            "institutional_bias": "bullish" if bullish_volume > bearish_volume else "bearish",
            "bias_strength": abs(bullish_volume - bearish_volume) / (bullish_volume + bearish_volume),
            "smart_money_ratio": smart_money_ratio,
            "absorption_detected": recent['absorption'].any(),
            "stopping_volume": recent['stopping_volume'].any(),
            "climax_volume": is_climax,
            "stealth_mode": stealth_mode,
            "accumulation": recent['accumulation'].sum() > recent['distribution'].sum(),
            "delta_trend": {
                "direction": np.sign(recent['cum_delta'].iloc[-1]),
                "strength": abs(recent['cum_delta'].iloc[-1]) / (recent['volume'].sum())
            },
            "volume_profile": {
                "high_nodes": [
                    {"price": interval.mid, "volume": vol}
                    for interval, vol in high_volume_nodes.items()
                ],
                "concentration": volume_profile.std() / volume_profile.mean()
            },
            "large_player_activity": recent['institutional_volume'].sum() / len(recent),
            "price_rejection": {
                "up": (recent['upper_wick'] > recent['body'] * 2).any(),
                "down": (recent['lower_wick'] > recent['body'] * 2).any()
            }
        }
        
    def _analyze_level2_footprint(self, 
                                l2_data: pd.DataFrame,
                                trades: pd.DataFrame) -> Dict:
        """Analyze institutional footprints using L2 data.
        
        Visual representation of what we look for:
        
        Order Book Imbalance:
        ```
        Asks    │  Bids
               │
        100    │  150
        200    │  500 ← Bid wall
        150    │  450
        300    │  200
        ```
        
        Iceberg Detection:
        ```
        Time    Visible    Executed
        t1      100       100
        t2      100       500  ← Hidden size
        t3      100       100
        ```
        
        Args:
            l2_data: DataFrame with columns [timestamp, price, size, side]
            trades: DataFrame with columns [timestamp, price, size, aggressor]
        """
        # Implementation for when we have L2 data
        # This is a placeholder for now
        return {
            "institutional_bias": "neutral",
            "bias_strength": 0.0,
            "absorption_detected": False,
            "stopping_volume": False,
            "delta_trend": 0,
            "large_player_activity": 0.0,
            "price_rejection": {"up": False, "down": False}
        }
        
    def analyze_order_flow(self, 
                          df: pd.DataFrame,
                          l2_data: Optional[pd.DataFrame] = None,
                          trades: Optional[pd.DataFrame] = None) -> Dict:
        """Analyze order flow using available data."""
        if self.use_level2 and l2_data is not None and trades is not None:
            return self._analyze_level2_footprint(l2_data, trades)
        else:
            return self._analyze_level1_footprint(df) 

def _mock_test():
    """Run a mock test of the OrderFlowAnalyzer."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create mock OHLCV data
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2024-01-01 09:30:00', end='2024-01-01 16:00:00', freq='1min')
    
    # Generate realistic price movement
    base_price = 100
    returns = np.random.normal(0, 0.0002, len(dates))  # Small random returns
    prices = base_price * np.exp(np.cumsum(returns))  # Log-normal price movement
    
    # Generate volume with some spikes
    base_volume = np.random.lognormal(6, 0.5, len(dates))
    # Add some institutional volume spikes
    spike_indices = np.random.choice(len(dates), 10, replace=False)
    base_volume[spike_indices] *= 5
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0001, len(dates))),
        'high': prices * (1 + abs(np.random.normal(0, 0.0002, len(dates)))),
        'low': prices * (1 - abs(np.random.normal(0, 0.0002, len(dates)))),
        'close': prices * (1 + np.random.normal(0, 0.0001, len(dates))),
        'volume': base_volume
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Create analyzer instance
    analyzer = OrderFlowAnalyzer(use_level2=False)
    
    # Run analysis
    try:
        result = analyzer._analyze_level1_footprint(df)
        print("\n=== Order Flow Analysis Results ===")
        for key, value in result.items():
            print(f"\n{key}:")
            print(f"  {value}")
        print("\nMock test completed successfully!")
        return True
    except Exception as e:
        print(f"\nError during mock test: {str(e)}")
        return False

if __name__ == "__main__":
    _mock_test() 