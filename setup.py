from setuptools import setup, find_packages

setup(
    name="llm_trading_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "openai>=1.12.0",
        "anthropic>=0.8.1",
        "pandas>=2.2.0",
        "numpy>=1.26.3",
        "polygon-api-client>=1.14.4",
        "plotly>=5.18.0",
        "dash>=2.14.2",
        "dash-bootstrap-components>=1.5.0",
        "ccxt>=4.2.15",
        "pytest>=8.0.0",
        "python-binance>=1.0.19"
    ],
) 