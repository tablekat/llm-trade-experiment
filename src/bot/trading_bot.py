from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd
import asyncio
import logging
from src.llm.base import LLMProvider
from src.data.market_data import MarketDataFetcher, MarketDataProvider
from src.analysis.market_regime import MarketRegimeDetector, MarketRegime

class TradingBot:
    """Main trading bot class that coordinates LLM decisions with market data."""
    
    def __init__(self, symbol: str, data_fetcher, llm, 
                 max_position_size: float = 1.0,
                 min_confidence: float = 0.6,
                 min_risk_reward: float = 1.5):
        """Initialize the trading bot.
        
        Args:
            symbol: Trading symbol (e.g. SPY, QQQ)
            data_fetcher: Market data fetcher instance
            llm: LLM provider instance
            max_position_size: Maximum position size (1.0 = 100%)
            min_confidence: Minimum confidence required to take a trade
            min_risk_reward: Minimum risk/reward ratio required
        """
        self.symbol = symbol
        self.data_fetcher = data_fetcher
        self.llm = llm
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        self.regime_detector = MarketRegimeDetector()
        self.logger = logging.getLogger(__name__)
        
    def _adjust_for_regime(self, decision: dict, regime_info: dict) -> dict:
        """Adjust trading decision based on market regime.
        
        Args:
            decision: Original trading decision
            regime_info: Market regime information
            
        Returns:
            dict: Adjusted trading decision
        """
        regime = regime_info['regime']
        regime_conf = regime_info['confidence']
        details = regime_info['details']
        
        # Adjust confidence based on regime alignment
        position = decision.get('position', 0)
        confidence = decision.get('confidence', 0)
        
        # In ranging markets, reduce position size and tighten stops
        if regime in [MarketRegime.RANGING_LOW_VOL, MarketRegime.RANGING_HIGH_VOL]:
            decision['position'] = position * 0.7  # Reduce position size
            
            # Tighten stops in high volatility
            if regime == MarketRegime.RANGING_HIGH_VOL:
                current_price = decision.get('current_price', 0)
                if position > 0:  # Long
                    new_sl = max(
                        decision.get('stop_loss', 0),
                        current_price - (current_price - decision.get('stop_loss', 0)) * 0.7
                    )
                else:  # Short
                    new_sl = min(
                        decision.get('stop_loss', 0),
                        current_price + (decision.get('stop_loss', 0) - current_price) * 0.7
                    )
                decision['stop_loss'] = new_sl
                
        # In trending markets, align with trend and potentially increase position
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            trend_alignment = (
                (regime == MarketRegime.TRENDING_UP and position > 0) or
                (regime == MarketRegime.TRENDING_DOWN and position < 0)
            )
            if trend_alignment:
                decision['confidence'] = min(1.0, confidence * (1 + regime_conf * 0.3))
            else:
                decision['confidence'] = confidence * 0.5
                
        # In breakout regimes, increase size if aligned with breakout
        elif regime == MarketRegime.BREAKOUT:
            breakout_alignment = (
                (details['trend_direction'] > 0 and position > 0) or
                (details['trend_direction'] < 0 and position < 0)
            )
            if breakout_alignment:
                decision['confidence'] = min(1.0, confidence * (1 + regime_conf * 0.5))
                
        # In reversal regimes, increase confidence if trading against old trend
        elif regime == MarketRegime.REVERSAL:
            reversal_alignment = (
                (details['trend_direction'] < 0 and position > 0) or
                (details['trend_direction'] > 0 and position < 0)
            )
            if reversal_alignment:
                decision['confidence'] = min(1.0, confidence * (1 + regime_conf * 0.4))
                
        # Add regime info to reasoning
        decision['reasoning'] = f"Market Regime: {regime.value} (conf: {regime_conf:.2f})\n" + decision.get('reasoning', '')
        
        return decision
        
    def _calculate_position_size(self, decision: dict) -> float:
        """Calculate position size based on confidence and risk/reward.
        
        Args:
            decision: Trading decision dict with position, confidence, take_profit, stop_loss
            
        Returns:
            float: Position size (0.0 to max_position_size)
        """
        if abs(decision.get('position', 0)) < 0.1 or decision.get('confidence', 0) < self.min_confidence:
            return 0.0
            
        # Calculate risk/reward ratio
        current_price = float(decision.get('current_price', 0))
        take_profit = float(decision.get('take_profit', 0))
        stop_loss = float(decision.get('stop_loss', 0))
        
        if not all([current_price, take_profit, stop_loss]):
            return 0.0
            
        # Calculate potential profit and loss
        if decision['position'] > 0:  # Long
            potential_profit = take_profit - current_price
            potential_loss = current_price - stop_loss
        else:  # Short
            potential_profit = current_price - take_profit
            potential_loss = stop_loss - current_price
            
        if potential_loss <= 0:
            return 0.0
            
        risk_reward = potential_profit / potential_loss
        
        if risk_reward < self.min_risk_reward:
            self.logger.info(f"Risk/reward {risk_reward:.2f} below minimum {self.min_risk_reward}")
            return 0.0
            
        # Scale position size by confidence and risk/reward
        confidence_factor = min(1.0, decision.get('confidence', 0))
        risk_reward_factor = min(1.0, risk_reward / (2 * self.min_risk_reward))
        
        position_size = self.max_position_size * confidence_factor * risk_reward_factor
        return round(position_size, 2)
        
    def _adjust_position_for_regime(self, position: float, confidence: float, regime_info: Dict) -> float:
        """Adjust position size based on market regime."""
        regime = regime_info['regime']
        regime_conf = regime_info['confidence']
        
        # Base position scaling factors for different regimes
        regime_factors = {
            MarketRegime.TRENDING_UP: 1.0,
            MarketRegime.TRENDING_DOWN: 1.0,
            MarketRegime.MOMENTUM: 1.2,  # Increase size in strong momentum
            MarketRegime.BREAKOUT: 1.1,  # Slightly increase for breakouts
            MarketRegime.RANGING_LOW_VOL: 0.8,  # Reduce size in ranging markets
            MarketRegime.RANGING_HIGH_VOL: 0.6,  # Further reduce in high volatility
            MarketRegime.REVERSAL: 0.7,  # Conservative on reversals
            MarketRegime.ACCUMULATION: 0.9,  # Moderate in accumulation
            MarketRegime.DISTRIBUTION: 0.7,  # Conservative in distribution
            MarketRegime.EXHAUSTION: 0.5,  # Very conservative in exhaustion
        }
        
        # Get base regime factor
        regime_factor = regime_factors.get(regime, 0.5)
        
        # Adjust factor based on regime confidence
        regime_factor *= regime_conf
        
        # Calculate final position size
        adjusted_position = position * regime_factor * confidence
        
        # Ensure position is within bounds
        return max(min(adjusted_position, 1.0), -1.0)

    def _adjust_stops_for_regime(self, take_profit: float, stop_loss: float, 
                               current_price: float, regime_info: Dict) -> Tuple[float, float]:
        """Adjust stop levels based on market regime."""
        regime = regime_info['regime']
        
        # Base risk multipliers for different regimes
        risk_multipliers = {
            MarketRegime.TRENDING_UP: 1.0,
            MarketRegime.TRENDING_DOWN: 1.0,
            MarketRegime.MOMENTUM: 0.8,  # Tighter stops in momentum
            MarketRegime.BREAKOUT: 1.2,  # Wider stops for breakouts
            MarketRegime.RANGING_LOW_VOL: 0.9,
            MarketRegime.RANGING_HIGH_VOL: 1.3,  # Wider stops in high vol
            MarketRegime.REVERSAL: 1.1,
            MarketRegime.ACCUMULATION: 0.9,
            MarketRegime.DISTRIBUTION: 1.1,
            MarketRegime.EXHAUSTION: 1.2,
        }
        
        multiplier = risk_multipliers.get(regime, 1.0)
        
        # Calculate base distances
        if take_profit is not None:
            tp_distance = abs(take_profit - current_price)
            adjusted_tp = current_price + (tp_distance * multiplier * (1 if take_profit > current_price else -1))
        else:
            adjusted_tp = take_profit
            
        if stop_loss is not None:
            sl_distance = abs(stop_loss - current_price)
            adjusted_sl = current_price + (sl_distance * multiplier * (1 if stop_loss > current_price else -1))
        else:
            adjusted_sl = stop_loss
            
        return adjusted_tp, adjusted_sl

    def _analyze_order_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze order flow patterns to detect institutional activity."""
        # Calculate imbalances
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Detect sweeps (large aggressive orders)
        vol_mean = df['volume'].rolling(20).mean()
        df['sweep'] = (df['volume'] > vol_mean * 2) & (df['body_size'] > df['body_size'].rolling(20).mean())
        
        # Analyze recent sweeps
        recent = df.tail(20)
        bullish_sweeps = recent[
            (recent['sweep']) & 
            (recent['close'] > recent['open'])
        ]['volume'].sum()
        bearish_sweeps = recent[
            (recent['sweep']) & 
            (recent['close'] < recent['open'])
        ]['volume'].sum()
        
        # Detect stopping volume
        df['stopping_volume'] = (
            (df['volume'] > vol_mean * 1.5) &
            (
                ((df['close'] > df['open']) & (df['lower_wick'] > df['body_size'])) |
                ((df['close'] < df['open']) & (df['upper_wick'] > df['body_size']))
            )
        )
        
        return {
            "sweep_bias": "bullish" if bullish_sweeps > bearish_sweeps else "bearish",
            "sweep_strength": max(bullish_sweeps, bearish_sweeps) / vol_mean.mean(),
            "stopping_volume_detected": recent['stopping_volume'].any(),
            "aggressive_flow": (bullish_sweeps + bearish_sweeps) > (vol_mean.mean() * 10)
        }

    async def get_trading_decision(self, timestamp=None):
        """Get trading decision with enhanced order flow analysis."""
        try:
            # Fetch data
            hourly_df, min15_df, min5_df, min1_df = await self.data_fetcher.fetch_multi_timeframe_data(end_time=timestamp)
            
            if min1_df.empty:
                return {
                    "position": 0.0,
                    "confidence": 0.0,
                    "reasoning": "No market data available"
                }
                
            current_price = min1_df.iloc[-1]['close']
            
            # Analyze order flow
            flow_analysis = self._analyze_order_flow(min1_df)
            
            # Detect market regime
            regime_info = self.regime_detector.detect_regime(hourly_df, min15_df)
            
            # Get base trading decision
            decision = await self.llm.get_trading_decision(
                hourly_df=hourly_df,
                min15_df=min15_df,
                min5_df=min5_df,
                min1_df=min1_df,
                additional_context={
                    "market_regime": regime_info,
                    "order_flow": flow_analysis
                }
            )
            
            # Add current price
            decision['current_price'] = current_price
            
            # Adjust based on order flow
            if flow_analysis['aggressive_flow']:
                if flow_analysis['sweep_bias'] == "bullish" and decision['position'] > 0:
                    decision['confidence'] = min(1.0, decision['confidence'] * (1 + flow_analysis['sweep_strength']))
                elif flow_analysis['sweep_bias'] == "bearish" and decision['position'] < 0:
                    decision['confidence'] = min(1.0, decision['confidence'] * (1 + flow_analysis['sweep_strength']))
                    
            if flow_analysis['stopping_volume_detected']:
                decision['confidence'] *= 0.7  # Reduce confidence when stopping volume is detected
            
            # Adjust for regime
            decision = self._adjust_for_regime(decision, regime_info)
            
            # Calculate final position size
            raw_position = decision.get('position', 0)
            adjusted_position = self._adjust_position_for_regime(
                raw_position, 
                decision['confidence'],
                regime_info
            )
            
            # Adjust stops
            adjusted_tp, adjusted_sl = self._adjust_stops_for_regime(
                decision['take_profit'],
                decision['stop_loss'],
                current_price,
                regime_info
            )
            
            # Update decision
            decision.update({
                'position': adjusted_position,
                'take_profit': adjusted_tp,
                'stop_loss': adjusted_sl
            })
            
            # Log enhanced decision details
            self.logger.info(
                f"Decision: pos={decision['position']:.2f} (raw={raw_position:.2f}), "
                f"conf={decision.get('confidence', 0):.2f}, "
                f"tp={decision.get('take_profit', 0):.2f}, "
                f"sl={decision.get('stop_loss', 0):.2f}, "
                f"regime={regime_info['regime'].value}, "
                f"flow_bias={flow_analysis['sweep_bias']}"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error getting trading decision: {str(e)}")
            return {
                "position": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }
            
    async def get_minute_data(self, start_time, end_time):
        """Get 1-minute candle data between specified timestamps.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            pandas.DataFrame: 1-minute OHLCV data
        """
        try:
            return await self.data_fetcher.get_candles(
                interval="1m",
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            self.logger.error(f"Error fetching minute data: {str(e)}")
            return pd.DataFrame()
            
    async def run_live(self):
        """Run the bot in live trading mode."""
        self.logger.info(f"Starting live trading for {self.symbol}")
        
        while True:
            try:
                decision = await self.get_trading_decision()
                
                # Log the decision
                self.logger.info(
                    f"Decision: pos={decision['position']:.2f}, "
                    f"conf={decision['confidence']:.2f}, "
                    f"reason={decision['reasoning']}"
                )
                
                # Check if we should take action
                if (abs(decision["position"]) >= self.position_threshold and
                    decision["confidence"] >= self.min_confidence):
                    
                    # Here you would implement actual trade execution
                    self.logger.info(
                        f"Would execute trade: "
                        f"{'LONG' if decision['position'] > 0 else 'SHORT'} "
                        f"with size {abs(decision['position']):.2f}"
                    )
                    self.current_position = decision["position"]
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in live trading loop: {str(e)}")
                await asyncio.sleep(self.update_interval)
                
    def run(self):
        """Run the bot (blocking)."""
        asyncio.run(self.run_live()) 