import pandas as pd
import numpy as np
from enum import Enum
from typing import Tuple, Dict, List
from dataclasses import dataclass

class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING_LOW_VOL = "ranging_low_vol"
    RANGING_HIGH_VOL = "ranging_high_vol"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    ACCUMULATION = "accumulation"  # New: Sideways with increasing volume
    DISTRIBUTION = "distribution"  # New: Sideways with decreasing volume
    MOMENTUM = "momentum"  # New: Strong trend with increasing momentum
    EXHAUSTION = "exhaustion"  # New: Strong trend with decreasing momentum
    UNKNOWN = "unknown"

@dataclass
class RegimeTransition:
    """Represents a transition between market regimes."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    confidence: float
    timestamp: pd.Timestamp
    metrics: Dict

class MarketRegimeDetector:
    """Detects market regime using multiple indicators and timeframes."""
    
    def __init__(self, 
                 trend_window: int = 20,
                 vol_window: int = 20,
                 breakout_std: float = 2.0,
                 trend_threshold: float = 0.6,
                 momentum_lookback: int = 10,
                 transition_memory: int = 5):
        """Initialize detector with parameters.
        
        Args:
            trend_window: Window for trend calculations
            vol_window: Window for volatility calculations
            breakout_std: Standard deviations for breakout detection
            trend_threshold: Threshold for trend strength (0.0 to 1.0)
            momentum_lookback: Periods to look back for momentum calculation
            transition_memory: Number of regime transitions to remember
        """
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.breakout_std = breakout_std
        self.trend_threshold = trend_threshold
        self.momentum_lookback = momentum_lookback
        self.transition_memory = transition_memory
        self.regime_history: List[RegimeTransition] = []
        
    def _calculate_trend_strength(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate trend strength using multiple indicators.
        
        Returns:
            Tuple[float, float]: (trend_strength, trend_direction)
            trend_strength: 0.0 to 1.0 (stronger trend)
            trend_direction: -1.0 to 1.0 (down to up)
        """
        # Calculate EMAs
        ema20 = df['close'].ewm(span=20).mean()
        ema50 = df['close'].ewm(span=50).mean()
        
        # ADX for trend strength
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        
        # Combine indicators for trend strength and direction
        trend_strength = min(1.0, adx.iloc[-1] / 100)
        
        # Trend direction from EMAs and DI
        ema_direction = 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1
        di_direction = 1 if plus_di.iloc[-1] > minus_di.iloc[-1] else -1
        trend_direction = ema_direction if ema_direction == di_direction else 0
        
        return trend_strength, trend_direction
        
    def _detect_volatility_regime(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Detect if we're in a high volatility regime.
        
        Returns:
            Tuple[bool, float]: (is_high_vol, vol_percentile)
        """
        # Calculate historical volatility
        returns = np.log(df['close'] / df['close'].shift(1))
        rolling_std = returns.rolling(self.vol_window).std() * np.sqrt(252)
        
        # Get current volatility percentile
        vol_percentile = (rolling_std.iloc[-1] - rolling_std.min()) / (rolling_std.max() - rolling_std.min())
        is_high_vol = vol_percentile > 0.7
        
        return is_high_vol, vol_percentile
        
    def _detect_breakout(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Detect if we're in a breakout.
        
        Returns:
            Tuple[bool, float]: (is_breakout, breakout_strength)
        """
        # Calculate Bollinger Bands
        rolling_mean = df['close'].rolling(self.trend_window).mean()
        rolling_std = df['close'].rolling(self.trend_window).std()
        
        upper_band = rolling_mean + (rolling_std * self.breakout_std)
        lower_band = rolling_mean - (rolling_std * self.breakout_std)
        
        # Check if price is outside bands
        current_price = df['close'].iloc[-1]
        is_breakout = current_price > upper_band.iloc[-1] or current_price < lower_band.iloc[-1]
        
        # Calculate breakout strength
        if is_breakout:
            if current_price > upper_band.iloc[-1]:
                strength = (current_price - upper_band.iloc[-1]) / rolling_std.iloc[-1]
            else:
                strength = (lower_band.iloc[-1] - current_price) / rolling_std.iloc[-1]
        else:
            strength = 0.0
            
        return is_breakout, min(1.0, strength / 2)
        
    def _calculate_momentum(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate momentum using rate of change and volume.
        
        Returns:
            Tuple[float, float]: (momentum_strength, momentum_direction)
            momentum_strength: 0.0 to 1.0 (stronger momentum)
            momentum_direction: -1.0 to 1.0 (down to up)
        """
        # Rate of change momentum
        roc = df['close'].pct_change(self.momentum_lookback)
        
        # Volume-weighted momentum
        vol_roc = df['volume'].pct_change(self.momentum_lookback)
        rel_vol = df['volume'] / df['volume'].rolling(50).mean()
        
        # RSI for overbought/oversold
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Combine indicators
        mom_direction = np.sign(roc.iloc[-1])
        mom_strength = min(1.0, abs(roc.iloc[-1]) * (1 + rel_vol.iloc[-1]) / 2)
        
        # Adjust for overbought/oversold
        if rsi.iloc[-1] > 70:
            mom_strength *= 0.7  # Reduce momentum in overbought
        elif rsi.iloc[-1] < 30:
            mom_strength *= 0.7  # Reduce momentum in oversold
            
        return mom_strength, mom_direction
        
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume profile for institutional activity patterns."""
        # Calculate base metrics
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['money_flow'] = df['typical_price'] * df['volume']
        
        # Detect large volume clusters
        vol_mean = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        df['large_volume'] = df['volume'] > (vol_mean + 2 * vol_std)
        
        # Analyze price action around large volume
        large_vol_bars = df[df['large_volume']]
        bullish_volume = large_vol_bars[large_vol_bars['close'] > large_vol_bars['open']]['volume'].sum()
        bearish_volume = large_vol_bars[large_vol_bars['close'] < large_vol_bars['open']]['volume'].sum()
        
        # Delta volume analysis
        df['delta'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        cumulative_delta = df['delta'].rolling(20).sum()
        
        # Detect absorption
        df['spread'] = df['high'] - df['low']
        df['absorption'] = (df['volume'] > vol_mean * 1.5) & (df['spread'] < df['spread'].rolling(20).mean() * 0.7)
        
        # Price rejection analysis
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body'] = abs(df['close'] - df['open'])
        
        recent = df.tail(20)
        
        return {
            "vol_trend_strength": abs(cumulative_delta.iloc[-1] / df['volume'].mean()),
            "vol_trend_direction": np.sign(cumulative_delta.iloc[-1]),
            "institutional_bias": "bullish" if bullish_volume > bearish_volume else "bearish",
            "absorption_detected": recent['absorption'].any(),
            "price_rejection": {
                "up": (recent['upper_wick'] > recent['body'] * 2).any(),
                "down": (recent['lower_wick'] > recent['body'] * 2).any()
            },
            "relative_volume": df['volume'].iloc[-1] / vol_mean.iloc[-1],
            "large_player_activity": (df['large_volume'].rolling(5).sum().iloc[-1] > 2)
        }
        
    def _detect_liquidity_levels(self, df: pd.DataFrame) -> Dict:
        """Detect potential liquidity levels using volume and price action.
        
        Returns:
            Dict with liquidity analysis
        """
        # Find high volume nodes
        vol_profile = pd.DataFrame({
            'price': df['close'],
            'volume': df['volume']
        })
        
        # Create price bins
        bins = pd.qcut(vol_profile['price'], q=20)
        vol_by_price = vol_profile.groupby(bins)['volume'].sum()
        
        # Find high volume nodes
        high_vol_threshold = vol_by_price.mean() + vol_by_price.std()
        liquidity_levels = vol_by_price[vol_by_price > high_vol_threshold]
        
        # Calculate distance to nearest liquidity level
        current_price = df['close'].iloc[-1]
        
        # Get bin intervals for liquidity levels
        distances = []
        for bin_interval, level in liquidity_levels.items():
            mid_price = (bin_interval.left + bin_interval.right) / 2
            distances.append((level, abs(current_price - mid_price)))
        
        distances.sort(key=lambda x: x[1])
        
        return {
            "nearest_liquidity": distances[0][0] if distances else 0,
            "distance_to_liquidity": distances[0][1] if distances else float('inf'),
            "liquidity_above": any((bin_interval.left + bin_interval.right) / 2 > current_price 
                                 for bin_interval, _ in liquidity_levels.items()),
            "liquidity_below": any((bin_interval.left + bin_interval.right) / 2 < current_price 
                                 for bin_interval, _ in liquidity_levels.items())
        }
        
    def _detect_regime_transition(self, current_regime: MarketRegime, 
                                confidence: float, metrics: Dict,
                                timestamp: pd.Timestamp) -> None:
        """Detect and record regime transitions."""
        if not self.regime_history:
            self.regime_history.append(RegimeTransition(
                MarketRegime.UNKNOWN, current_regime, confidence, timestamp, metrics
            ))
            return
            
        last_regime = self.regime_history[-1].to_regime
        if last_regime != current_regime:
            transition = RegimeTransition(
                last_regime, current_regime, confidence, timestamp, metrics
            )
            self.regime_history.append(transition)
            
            # Keep only recent transitions
            if len(self.regime_history) > self.transition_memory:
                self.regime_history.pop(0)
                
    def detect_regime(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame) -> Dict:
        """Detect current market regime using multiple timeframes.
        
        Returns:
            Dict with regime info:
            - regime: MarketRegime enum
            - confidence: 0.0 to 1.0
            - details: Dict with supporting metrics
            - transitions: List of recent regime transitions
        """
        # Get trend info from both timeframes
        h_trend_str, h_trend_dir = self._calculate_trend_strength(hourly_df)
        m15_trend_str, m15_trend_dir = self._calculate_trend_strength(min15_df)
        
        # Get momentum info
        h_mom_str, h_mom_dir = self._calculate_momentum(hourly_df)
        m15_mom_str, m15_mom_dir = self._calculate_momentum(min15_df)
        
        # Get volatility info
        h_high_vol, h_vol_pct = self._detect_volatility_regime(hourly_df)
        m15_high_vol, m15_vol_pct = self._detect_volatility_regime(min15_df)
        
        # Get breakout info
        h_breakout, h_break_str = self._detect_breakout(hourly_df)
        m15_breakout, m15_break_str = self._detect_breakout(min15_df)
        
        # Get volume profile analysis
        h_vol_profile = self._analyze_volume_profile(hourly_df)
        m15_vol_profile = self._analyze_volume_profile(min15_df)
        
        # Get liquidity analysis
        h_liquidity = self._detect_liquidity_levels(hourly_df)
        m15_liquidity = self._detect_liquidity_levels(min15_df)
        
        # Combine metrics
        trend_strength = (h_trend_str * 0.7 + m15_trend_str * 0.3)
        trend_direction = h_trend_dir if abs(h_trend_dir) > 0 else m15_trend_dir
        mom_strength = (h_mom_str * 0.7 + m15_mom_str * 0.3)
        mom_direction = h_mom_dir if abs(h_mom_dir) > 0 else m15_mom_dir
        is_high_vol = h_high_vol or m15_high_vol
        vol_percentile = max(h_vol_pct, m15_vol_pct)
        is_breakout = h_breakout or m15_breakout
        breakout_strength = max(h_break_str, m15_break_str)
        
        # Enhanced regime detection
        regime = MarketRegime.UNKNOWN
        confidence = 0.0
        
        # Check for momentum regime
        if trend_strength > 0.6 and mom_strength > 0.7 and trend_direction == mom_direction:
            regime = MarketRegime.MOMENTUM
            confidence = min(trend_strength, mom_strength)
            
        # Check for exhaustion
        elif trend_strength > 0.6 and mom_strength < 0.3:
            regime = MarketRegime.EXHAUSTION
            confidence = trend_strength * (1 - mom_strength)
            
        # Check for accumulation/distribution
        elif trend_strength < 0.4 and not is_high_vol:
            vol_trend = h_vol_profile['vol_trend_direction']
            if vol_trend > 0 and h_vol_profile['high_vol_at_lows'] > 1.2:
                regime = MarketRegime.ACCUMULATION
                confidence = h_vol_profile['high_vol_at_lows'] / 2
            elif vol_trend < 0 and h_vol_profile['high_vol_at_highs'] > 1.2:
                regime = MarketRegime.DISTRIBUTION
                confidence = h_vol_profile['high_vol_at_highs'] / 2
                
        # Check other regimes
        elif is_breakout and breakout_strength > 0.5:
            regime = MarketRegime.BREAKOUT
            confidence = breakout_strength
        elif trend_strength > self.trend_threshold:
            if trend_direction > 0:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
            confidence = trend_strength
        else:
            if is_high_vol:
                regime = MarketRegime.RANGING_HIGH_VOL
            else:
                regime = MarketRegime.RANGING_LOW_VOL
            confidence = 1 - trend_strength
            
        # Check for reversal
        if trend_strength > 0.4 and breakout_strength > 0.3 and trend_direction * h_trend_dir < 0:
            regime = MarketRegime.REVERSAL
            confidence = min(trend_strength, breakout_strength)
            
        # Record regime transition
        self._detect_regime_transition(
            regime, confidence,
            {
                "trend_strength": trend_strength,
                "trend_direction": trend_direction,
                "momentum_strength": mom_strength,
                "momentum_direction": mom_direction,
                "volatility_percentile": vol_percentile,
                "breakout_strength": breakout_strength
            },
            hourly_df.index[-1]
        )
            
        return {
            "regime": regime,
            "confidence": confidence,
            "details": {
                "trend_strength": trend_strength,
                "trend_direction": trend_direction,
                "momentum_strength": mom_strength,
                "momentum_direction": mom_direction,
                "volatility_percentile": vol_percentile,
                "is_high_volatility": is_high_vol,
                "breakout_strength": breakout_strength,
                "volume_profile": h_vol_profile,
                "liquidity_levels": h_liquidity
            },
            "transitions": [
                {
                    "from": t.from_regime.value,
                    "to": t.to_regime.value,
                    "confidence": t.confidence,
                    "timestamp": t.timestamp,
                    "metrics": t.metrics
                }
                for t in self.regime_history
            ]
        } 