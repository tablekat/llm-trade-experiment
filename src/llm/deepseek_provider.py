import json
import logging
import aiohttp
import pandas as pd
import platform
import asyncio
from typing import Dict, Optional
from .base import LLMProvider
from src.prompts.generators import BasePromptGenerator, PromptV0, PromptFVG

# Configure Windows-specific event loop policy
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging to ignore DEBUG from other libraries
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)

class DeepSeekProvider(LLMProvider):
    """DeepSeek implementation of the LLM provider."""
    
    def __init__(self, api_key: str, dry_run: bool = False, prompt_generator: Optional[BasePromptGenerator] = None):
        """Initialize provider with API key.
        
        Args:
            api_key: DeepSeek API key
            dry_run: If True, only log the prompt without making API calls
            prompt_generator: Optional prompt generator to use. Defaults to PromptFVG
        """
        self.api_key = api_key
        self.dry_run = dry_run
        self.prompt_generator = prompt_generator or PromptFVG()
        self.logger = logging.getLogger(__name__)

    def _generate_prompt(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame, min5_df: pd.DataFrame, min1_df: pd.DataFrame, additional_context: dict = None) -> str:
        """Generate analysis prompt from market data."""
        # Get key price levels
        def get_key_levels(df: pd.DataFrame, periods: int = 20) -> tuple:
            highs = df['high'].rolling(periods, center=True).max()
            lows = df['low'].rolling(periods, center=True).min()
            current_price = df['close'].iloc[-1]
            
            resistance_levels = sorted([p for p in highs.unique() if p > current_price])[:3]
            support_levels = sorted([p for p in lows.unique() if p < current_price], reverse=True)[:3]
            return support_levels, resistance_levels

        # Calculate technical indicators
        def calculate_indicators(df: pd.DataFrame) -> dict:
            # EMAs
            df['ema20'] = df['close'].ewm(span=20).mean()
            df['ema50'] = df['close'].ewm(span=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Volume analysis
            vol_sma = df['volume'].rolling(20).mean()
            rel_vol = df['volume'] / vol_sma
            
            return {
                'ema_trend': 'Bullish' if df['ema20'].iloc[-1] > df['ema50'].iloc[-1] else 'Bearish',
                'rsi': rsi.iloc[-1],
                'rel_volume': rel_vol.iloc[-1]
            }

        # Generate timeframe analysis
        summaries = []
        for name, df, weight in [
            ("Hourly", hourly_df, "40%"),
            ("15-minute", min15_df, "30%"),
            ("5-minute", min5_df, "20%"),
            ("1-minute", min1_df, "10%")
        ]:
            if df.empty:
                continue
                
            current = df.iloc[-1]
            recent = df.tail(10)
            
            # Get key levels
            supports, resistances = get_key_levels(df)
            indicators = calculate_indicators(df)
            
            price_change = ((current['close'] - recent.iloc[0]['close']) / recent.iloc[0]['close']) * 100
            vol_change = ((current['volume'] - recent['volume'].mean()) / recent['volume'].mean()) * 100
            
            summary = f"\n{name} Analysis (Weight: {weight}):\n"
            summary += f"Price Action:\n"
            summary += f"- Current: {current['close']:.2f} ({price_change:+.2f}% last 10 periods)\n"
            summary += f"- Key Resistances: {', '.join(f'{r:.2f}' for r in resistances)}\n"
            summary += f"- Key Supports: {', '.join(f'{s:.2f}' for s in supports)}\n"
            summary += f"\nTechnical Indicators:\n"
            summary += f"- Trend: {indicators['ema_trend']} (EMA20 vs EMA50)\n"
            summary += f"- RSI: {indicators['rsi']:.1f}\n"
            summary += f"- Volume: {vol_change:+.2f}% vs average (Relative: {indicators['rel_volume']:.2f}x)\n"
            summaries.append(summary)
            
        prompt = """You are an expert futures trader specializing in market structure analysis and risk management.
Analyze the following market data and provide a detailed trading decision.

Key Requirements:
1. Position size must reflect both directional conviction AND current market regime
2. Stop-loss must be placed beyond the nearest significant structure (support/resistance, fair value gap)
3. Take-profit must target the next major structure level with good risk:reward (minimum 1:1.5)
4. Confidence should consider:
   - Alignment of trends across timeframes
   - Volume confirmation
   - Market structure (support/resistance, fair value gaps)
   - Current market regime and volatility

Market Analysis:
"""
        prompt += "\n".join(summaries)
        
        if additional_context:
            prompt += f"\nMarket Context:\n{additional_context}\n"
            
        prompt += """\nBased on this analysis, provide a trading decision with:
1. Position (-1.0 for full short to 1.0 for full long)
2. Confidence level (0.0 to 1.0)
3. Take-profit price (must be at significant structure level)
4. Stop-loss price (must be beyond nearest structure)
5. Detailed reasoning including:
   - Primary market structure levels being used
   - Multi-timeframe trend alignment
   - Volume confirmation/concerns
   - Risk:reward ratio justification

Format response as JSON with keys: position, confidence, take_profit, stop_loss, reasoning
Note: reasoning should be a dictionary with keys: primary_levels, multi_timeframe_alignment, volume_confirmation, risk_reward"""
        
        return prompt

    def _format_reasoning(self, reasoning_dict: dict) -> str:
        """Format the reasoning dictionary into a readable string."""
        if not isinstance(reasoning_dict, dict):
            return str(reasoning_dict)
            
        sections = []
        for key, value in reasoning_dict.items():
            # Convert key from snake_case to Title Case
            title = key.replace('_', ' ').title()
            sections.append(f"{title}: {value}")
            
        return "\n".join(sections)

    async def get_trading_decision(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame, min5_df: pd.DataFrame, min1_df: pd.DataFrame, additional_context: Optional[Dict] = None) -> Dict:
        """Get a trading decision from the model."""
        # Generate prompt using the configured generator
        prompt = self.prompt_generator.generate(
            hourly_df=hourly_df,
            min15_df=min15_df,
            min5_df=min5_df,
            min1_df=min1_df,
            additional_context=additional_context
        )
        
        self.logger.info("Generated prompt:")
        self.logger.info(prompt)
        
        if self.dry_run:
            self.logger.info("Dry run mode - skipping API call")
            return {
                "position": 0.0,
                "confidence": 0.0,
                "take_profit": None,
                "stop_loss": None,
                "reasoning": "Dry run mode - no API call made"
            }
        
        # Prepare API request
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional futures trader. You will analyze market data and provide trading decisions in JSON format with position, confidence, and reasoning fields."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    self.logger.info(f"API response status: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"API error response: {error_text}")
                        raise ValueError(f"API request failed with status {response.status}: {error_text}")
                    
                    raw_response = await response.text()
                    self.logger.info(f"Raw API response: {raw_response}")
                    
                    try:
                        response_json = json.loads(raw_response)
                        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Clean up content (remove markdown code blocks if present)
                        content = content.replace("```json", "").replace("```", "").strip()
                        
                        # Find the JSON object in the content
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start == -1 or end == 0:
                            raise ValueError("No JSON object found in response")
                        
                        json_str = content[start:end]
                        self.logger.info(f"Cleaned content for parsing: {json_str}")
                        
                        # Parse the response
                        decision = json.loads(json_str)
                        
                        # Format the reasoning if it's a dictionary
                        if isinstance(decision.get('reasoning'), dict):
                            decision['reasoning'] = self._format_reasoning(decision['reasoning'])
                        
                        # Validate decision format
                        required_keys = ["position", "confidence", "take_profit", "stop_loss", "reasoning"]
                        if not all(key in decision for key in required_keys):
                            raise ValueError(f"Missing required keys in decision: {required_keys}")
                        
                        # Validate value ranges
                        if not -1.0 <= float(decision["position"]) <= 1.0:
                            raise ValueError(f"Position value out of range: {decision['position']}")
                        if not 0.0 <= float(decision["confidence"]) <= 1.0:
                            raise ValueError(f"Confidence value out of range: {decision['confidence']}")
                        
                        # Validate take-profit and stop-loss are numeric
                        try:
                            decision["take_profit"] = float(decision["take_profit"])
                            decision["stop_loss"] = float(decision["stop_loss"])
                        except (ValueError, TypeError):
                            raise ValueError("take_profit and stop_loss must be numeric values")
                        
                        # Validate take-profit and stop-loss make sense for the position
                        current_price = float(min1_df.iloc[-1]['close'])
                        if decision["position"] > 0:  # Long position
                            if decision["take_profit"] <= current_price:
                                raise ValueError("take_profit must be above current price for long positions")
                            if decision["stop_loss"] >= current_price:
                                raise ValueError("stop_loss must be below current price for long positions")
                        elif decision["position"] < 0:  # Short position
                            if decision["take_profit"] >= current_price:
                                raise ValueError("take_profit must be below current price for short positions")
                            if decision["stop_loss"] <= current_price:
                                raise ValueError("stop_loss must be above current price for short positions")
                        
                        self.logger.info(f"Final decision: {decision}")
                        return decision
                        
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse JSON response: {e}")
                        self.logger.error(f"Raw response that failed to parse: {raw_response}")
                        raise
                        
        except Exception as e:
            self.logger.error(f"Error getting trading decision: {str(e)}")
            return {
                "position": 0.0,
                "confidence": 0.0,
                "take_profit": None,
                "stop_loss": None,
                "reasoning": f"Error getting trading decision: {str(e)}"
            }

    async def test_api_connection(self):
        """Test the API connection with a simple request."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"API test failed: {error_text}")
                        return False
                    return True
        except Exception as e:
            self.logger.error(f"API test failed: {str(e)}")
            return False 