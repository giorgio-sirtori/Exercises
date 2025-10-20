import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timedelta

def get_user_input():
    """
    Prompts the user to enter their monthly GBP cash holdings and a fixed bond amount.
    """
    print("--- Currency Exposure & VaR Calculator ---")
    print("Please enter your total GBP cash holdings for the last 12 months.")
    
    monthly_exposures = {}
    today = datetime.today()
    for i in range(12, 0, -1):
        # Go back month by month
        month_date = today - timedelta(days=i*30)
        month_name = month_date.strftime("%Y-%m")
        while True:
            try:
                exposure = float(input(f"Enter GBP exposure for {month_name}: £"))
                # Use the end of the month for data fetching
                date_key = pd.to_datetime(month_name + '-01').to_period('M').end_time.date()
                monthly_exposures[date_key] = exposure
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    while True:
        try:
            bond_amount = float(input("\nEnter your fixed bond/guarantee amount in GBP: £"))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    print("\nInputs received. Starting analysis...")
    return monthly_exposures, bond_amount

def get_exchange_rate_data(start_date, end_date, daily=False):
    """
    Fetches historical GBP/EUR exchange rate data from Yahoo Finance.
    Can return either daily data or end-of-month resampled data.
    """
    print(f"Fetching {'daily' if daily else 'monthly'} GBP/EUR exchange rate data...")
    try:
        gbp_eur_data = yf.download('GBPEUR=X', start=start_date, end=end_date, progress=False)
        if gbp_eur_data.empty:
            raise ValueError("No data fetched. Check ticker or date range.")
        
        close_prices = gbp_eur_data['Close'].squeeze()
        
        if daily:
            return close_prices
        else:
            return close_prices.resample('M').last()
    except Exception as e:
        print(f"Could not fetch exchange rate data: {e}")
        return None

def calculate_monte_carlo_var(calculation_date, historical_daily_rates, confidence_level=0.95, simulations=10000, time_horizon_days=21):
    """
    Calculates monthly VaR using a Monte Carlo simulation for a specific date.
    """
    # Define the lookback period for calculating historical drift and volatility (1 year)
    lookback_end = calculation_date
    lookback_start = lookback_end - timedelta(days=365)
    
    # Filter the historical data for the lookback period
    historical_window = historical_daily_rates.loc[lookback_start:lookback_end]
    
    # Calculate daily log returns
    log_returns = np.log(1 + historical_window.pct_change())
    
    # Calculate drift (mean return) and volatility (standard deviation)
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    # Get the starting exchange rate for the simulation
    start_price = historical_window.iloc[-1]
    
    # --- Run Simulation ---
    # Create an array to hold the simulation results
    final_prices = np.zeros(simulations)
    
    for i in range(simulations):
        # Generate daily price path for the time horizon
        daily_returns = np.exp(mu - 0.5 * sigma**2 + sigma * np.random.normal(0, 1, time_horizon_days))
        price_path = np.zeros_like(daily_returns)
        price_path[0] = start_price
        for t in range(1, time_horizon_days):
            price_path[t] = price_path[t-1] * daily_returns[t]
        
        final_prices[i] = price_path[-1]

    # Calculate simulated monthly returns
    simulated_returns = (final_prices / start_price) - 1
    
    # Calculate VaR from the simulated returns
    var_percentile = 1 - confidence_level
    worst_loss_percentile = np.percentile(simulated_returns, var_percentile * 100)
    
    return abs(worst_loss_percentile)

def run_analysis(monthly_exposures, bond_amount):
    """
    Runs the main analysis and plotting.
    """
    exposure_df = pd.DataFrame(
        list(monthly_exposures.items()),
        columns=['Date', 'GBP_Exposure']
    ).set_index('Date').sort_index()

    # --- Data Fetching ---
    end_date_user = exposure_df.index.max()
    # Fetch a long history of DAILY data for Monte Carlo calculations (5 years before the first exposure date)
    start_date_mc = exposure_df.index.min() - timedelta(days=5*365)
    
    daily_rates_history = get_exchange_rate_data(start_date_mc, end_date_user, daily=True)
    monthly_rates_user = daily_rates_history.resample('M').last()

    if daily_rates_history is None:
        return

    # Map the end-of-month rates to the exposure DataFrame
    exposure_df['GBP_EUR_Rate'] = pd.to_datetime(exposure_df.index).to_period('M').map(monthly_rates_user.to_period('M'))
    exposure_df['GBP_EUR_Rate'].ffill(inplace=True)
    exposure_df['GBP_EUR_Rate'].bfill(inplace=True)

    # --- Calculations ---
    print("\nCalculating monthly VaR using Monte Carlo simulations...")
    # This can take a moment as it runs a simulation for each month
    exposure_df['VaR_Percentage'] = [
        calculate_monte_carlo_var(date, daily_rates_history) for date in exposure_df.index
    ]
    print("Monte Carlo simulations complete.")
    
    exposure_df['Bond_Amount_GBP'] = bond_amount
    exposure_df['Uncovered_Exposure_GBP'] = exposure_df['GBP_Exposure'] - exposure_df['Bond_Amount_GBP']
    
    exposure_df['EUR_Exposure'] = exposure_df['GBP_Exposure'] * exposure_df['GBP_EUR_Rate']
    exposure_df['Uncovered_Exposure_EUR'] = exposure_df['Uncovered_Exposure_GBP'] * exposure_df['GBP_EUR_Rate']
    
    # Calculate Monthly VaR in EUR only on the positive (uncovered) portion of the exposure
    exposure_df['Monthly_VaR_EUR'] = exposure_df['Uncovered_Exposure_EUR'].clip(lower=0) * exposure_df['VaR_Percentage']
    
    print("\n--- Analysis Results (Last 5 Months) ---")
    print(exposure_df[['GBP_Exposure', 'Uncovered_Exposure_GBP', 'GBP_EUR_Rate', 'VaR_Percentage', 'Monthly_VaR_EUR']].tail())

    # --- Plotting ---
    plot_results(exposure_df)


def plot_results(df):
    """
    Generates and displays plots for the analysis.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Monthly GBP Currency Exposure Analysis (Monte Carlo VaR)', fontsize=20, y=0.95)
    
    formatter_gbp = mticker.FuncFormatter(lambda x, p: f'£{x:,.0f}')
    formatter_eur = mticker.FuncFormatter(lambda x, p: f'€{x:,.0f}')
    
    ax1.bar(df.index, df['GBP_Exposure'], label='Monthly GBP Exposure', color='skyblue', width=20)
    ax1.axhline(y=df['Bond_Amount_GBP'].iloc[0], color='r', linestyle='--', label=f'Fixed Bond (£{df["Bond_Amount_GBP"].iloc[0]:,.0f})')
    ax1.set_title('Booking Curve: Monthly GBP Exposure vs. Fixed Bond', fontsize=14)
    ax1.set_ylabel('Amount (GBP)')
    ax1.yaxis.set_major_formatter(formatter_gbp)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    uncovered_colors = ['lightcoral' if x > 0 else 'lightgreen' for x in df['Uncovered_Exposure_GBP']]
    ax2.bar(df.index, df['Uncovered_Exposure_GBP'], color=uncovered_colors, width=20)
    ax2.set_title('Uncovered Exposure (Exposure - Bond)', fontsize=14)
    ax2.set_ylabel('Amount (GBP)')
    ax2.yaxis.set_major_formatter(formatter_gbp)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.legend(handles=[
        plt.Rectangle((0,0),1,1, color='lightcoral', label='Exposure > Bond (Uncovered)'),
        plt.Rectangle((0,0),1,1, color='lightgreen', label='Exposure < Bond (Covered)')
    ])
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    ax3.plot(df.index, df['Monthly_VaR_EUR'], marker='o', linestyle='-', color='purple', label='Monthly VaR (95%)')
    ax3.set_title('Potential Monthly Risk (Value at Risk in EUR)', fontsize=14)
    ax3.set_ylabel('Potential Loss (EUR)')
    ax3.yaxis.set_major_formatter(formatter_eur)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(df.index, [d.strftime('%Y-%m') for d in df.index], rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    # monthly_exposures, bond_amount = get_user_input()

    print("--- Using Sample Data for Demonstration ---")
    today = datetime.today()
    sample_exposures = {
        pd.to_datetime(today - timedelta(days=12*30)).to_period('M').end_time.date(): 33_800_000,
        pd.to_datetime(today - timedelta(days=11*30)).to_period('M').end_time.date(): 33_900_000,
        pd.to_datetime(today - timedelta(days=10*30)).to_period('M').end_time.date(): 27_300_000,
        pd.to_datetime(today - timedelta(days=9*30)).to_period('M').end_time.date(): 43_800_000, # Peak Season
        pd.to_datetime(today - timedelta(days=8*30)).to_period('M').end_time.date(): 48_500_000, # Peak Season
        pd.to_datetime(today - timedelta(days=7*30)).to_period('M').end_time.date(): 58_600_000,
        pd.to_datetime(today - timedelta(days=6*30)).to_period('M').end_time.date(): 56_800_000,
        pd.to_datetime(today - timedelta(days=5*30)).to_period('M').end_time.date(): 58_500_000,
        pd.to_datetime(today - timedelta(days=4*30)).to_period('M').end_time.date(): 60_500_000, # Off Season
        pd.to_datetime(today - timedelta(days=3*30)).to_period('M').end_time.date(): 60_200_000, # Off Season
        pd.to_datetime(today - timedelta(days=2*30)).to_period('M').end_time.date(): 48_800_000,
        pd.to_datetime(today - timedelta(days=1*30)).to_period('M').end_time.date(): 45_000_000,
    }
    bond_amount = 15_000_000
    print(f"Sample Bond Amount: £{bond_amount:,.0f}")
    
    run_analysis(sample_exposures, bond_amount)

