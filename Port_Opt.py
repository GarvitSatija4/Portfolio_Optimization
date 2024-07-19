import yfinance as yf
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from datetime import datetime

start_date = "2017-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

stocks = [
    "COALINDIA.NS", "TATASTEEL.NS", "BPCL.NS", "TATAMOTORS.NS", "HCLTECH.NS", "TCS.NS",
    "SBIN.NS", "ADANIENT.NS", "TITAN.NS", "GRASIM.NS", "LTIM.NS", "HINDALCO.NS", "LT.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "INFY.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "KOTAKBANK.NS",
    "DRREDDY.NS", "HDFCLIFE.NS", "AXISBANK.NS", "WIPRO.NS", "JSWSTEEL.NS", "ICICIBANK.NS",
    "BAJAJFINSV.NS", "APOLLOHOSP.NS", "ADANIPORTS.NS", "SBILIFE.NS", "ITC.NS", "INDUSINDBK.NS",
    "HINDUNILVR.NS", "SHRIRAMFIN.NS", "BRITANNIA.NS", "TECHM.NS", "BAJFINANCE.NS", "RELIANCE.NS",
    "NTPC.NS", "ONGC.NS", "BHARTIARTL.NS", "TATACONSUM.NS", "M&M.NS", "ULTRACEMCO.NS", "POWERGRID.NS",
    "BAJAJ-AUTO.NS", "DIVISLAB.NS", "CIPLA.NS", "HDFCBANK.NS", "NESTLEIND.NS", "SUNPHARMA.NS"
]

df = pd.DataFrame()

for stock in stocks:
    try:
        data = yf.download(stock, start=start_date, end=end_date)
        df[stock] = data["Adj Close"]
    except Exception as e:
        print(f"Could not download {stock}: {e}")
        continue

df = df.replace("", np.nan, regex=True)
df = df[df.columns[df.isnull().mean() < 0.5]]
df = df.dropna()

# Calculation of Returns, Standard Deviation, CoVariance, etc.
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimizing Sharpe Ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print(cleaned_weights)

ef.portfolio_performance(verbose=True)

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)

da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=100000)
allocation, leftover = da.lp_portfolio()

print("Discrete allocation:", allocation)
print("Funds remaining: Rs{:.2f}".format(leftover))
# Export results to Excel
with pd.ExcelWriter("portfolio_analysis.xlsx", engine='xlsxwriter') as writer:
    # Export cleaned weights
    pd.Series(cleaned_weights).to_excel(writer, sheet_name="Weights")
    
    # Export portfolio performance
    performance_df = pd.DataFrame({
        "Metric": ["Expected Annual Return", "Annual Volatility", "Sharpe Ratio"],
        "Value": [expected_return, volatility, sharpe_ratio]
    })
    performance_df.to_excel(writer, sheet_name="Performance", index=False)
    
    # Export discrete allocation
    allocation_df = pd.DataFrame(list(allocation.items()), columns=["Ticker", "Shares"])
    allocation_df.to_excel(writer, sheet_name="Discrete Allocation", index=False)
    
    # Export remaining funds
    pd.DataFrame({"Funds Remaining": [f"Rs{leftover:.2f}"]}).to_excel(writer, sheet_name="Discrete Allocation", startrow=len(allocation_df) + 2, index=False)
