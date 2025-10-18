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
        month_date = today - timedelta(days=i*30)
        month_name = month_date.strftime("%Y-%m")
        while True:
            try:
                exposure = float(input(f"Enter GBP exposure for {month_name}: £"))
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

def calculate_monte_carlo_var(calculation_date, historical_daily_rates, confidence_level=0.95, 
                               simulations=100000, time_horizon_days=21, lookback_days=252):
    """
    Calculates VaR using Monte Carlo simulation with proper Geometric Brownian Motion.
    """
    # Define the lookback window
    lookback_start = calculation_date - timedelta(days=lookback_days)
    
    # Filter historical data for the lookback window
    historical_window = historical_daily_rates.loc[lookback_start:calculation_date]
    
    if len(historical_window) < 60:  # Minimum 60 days of data
        print(f"Warning: Insufficient data for {calculation_date.strftime('%Y-%m')}")
        return 0.05  # Return conservative 5% VaR as fallback
    
    # Calculate daily log returns properly
    log_returns = np.log(historical_window / historical_window.shift(1)).dropna()
    
    # Remove outliers (beyond 3 standard deviations)
    returns_clean = log_returns[np.abs(log_returns - log_returns.mean()) <= 3 * log_returns.std()]
    
    if len(returns_clean) < 30:
        returns_clean = log_returns  # Fallback to original if too many outliers removed
    
    # Calculate parameters for GBM
    mu_daily = returns_clean.mean()
    sigma_daily = returns_clean.std()
    
    if sigma_daily == 0 or np.isnan(sigma_daily):
        return 0.03  # Return 3% VaR for zero volatility periods
    
    # Get the starting exchange rate
    start_price = historical_window.iloc[-1]
    
    # --- Run Simulation using proper GBM ---
    final_prices = np.zeros(simulations)
    
    # Generate all random shocks at once for efficiency
    Z = np.random.standard_normal((simulations, time_horizon_days))
    
    # GBM formula: S_t = S_0 * exp((μ - σ²/2)*Δt + σ*√Δt*Z)
    # For daily simulation: Δt = 1
    dt = 1
    drift = (mu_daily - 0.5 * sigma_daily**2) * dt
    volatility = sigma_daily * np.sqrt(dt)
    
    for i in range(simulations):
        # Generate price path using GBM
        price_path = start_price * np.exp(
            np.cumsum(drift + volatility * Z[i, :])
        )
        final_prices[i] = price_path[-1]

    # Calculate simulated returns
    simulated_returns = (final_prices / start_price) - 1
    
    # Calculate VaR from the simulated returns
    var_percentile = 1 - confidence_level
    worst_loss_percentile = np.percentile(simulated_returns, var_percentile * 100)
    
    # VaR should be between 0% and 25% for realistic FX scenarios
    var_result = max(0.0, min(abs(worst_loss_percentile), 0.25))
    
    # Debug info
    if var_result > 0.15:
        print(f"High VaR detected for {calculation_date.strftime('%Y-%m')}: {var_result*100:.1f}% "
              f"(mu: {mu_daily*100:.3f}%, sigma: {sigma_daily*100:.3f}%)")
    
    return var_result

def run_analysis(monthly_exposures, bond_amount, lookback_days=252, time_horizon_days=21):
    """
    Runs the main analysis with rolling windows and plots results.
    """
    exposure_df = pd.DataFrame(
        list(monthly_exposures.items()),
        columns=['Date', 'GBP_Exposure']
    )
    exposure_df['Date'] = pd.to_datetime(exposure_df['Date'])
    exposure_df = exposure_df.set_index('Date').sort_index()

    # --- Data Fetching ---
    end_date_user = exposure_df.index.max()
    # Fetch daily data starting well before the first exposure date
    start_date_mc = exposure_df.index.min() - timedelta(days=lookback_days + 365)
    
    daily_rates_history = get_exchange_rate_data(start_date_mc, end_date_user, daily=True)

    if daily_rates_history is None:
        print("Error: Could not fetch exchange rate data. Exiting.")
        return

    # Convert to datetime index for consistency
    daily_rates_history.index = pd.to_datetime(daily_rates_history.index)
    
    # Map end-of-month rates to the exposure DataFrame by finding the closest available date
    exposure_df['GBP_EUR_Rate'] = [
        daily_rates_history.asof(date) for date in exposure_df.index
    ]
    
    # Forward/backward fill any gaps
    exposure_df['GBP_EUR_Rate'].ffill(inplace=True)
    exposure_df['GBP_EUR_Rate'].bfill(inplace=True)

    # --- Calculations ---
    print("\nCalculating monthly VaR using Monte Carlo simulations with rolling windows...")
    print(f"  - Lookback period: {lookback_days} trading days")
    print(f"  - Time horizon: {time_horizon_days} days")
    print(f"  - Simulations: 10,000 per month")
    print(f"  - Confidence level: 95%\n")
    
    # Calculate VaR for each month
    var_results = []
    for calc_date in exposure_df.index:
        var_pct = calculate_monte_carlo_var(
            calc_date, 
            daily_rates_history,
            confidence_level=0.95,
            simulations=10000,
            time_horizon_days=time_horizon_days,
            lookback_days=lookback_days
        )
        var_results.append(var_pct)
    
    exposure_df['VaR_Percentage'] = var_results
    
    print("\nMonte Carlo simulations complete.")
    print(f"VaR range: {exposure_df['VaR_Percentage'].min()*100:.1f}% to {exposure_df['VaR_Percentage'].max()*100:.1f}%")
    
    # --- Financial Calculations ---
    exposure_df['Bond_Amount_GBP'] = bond_amount
    exposure_df['Uncovered_Exposure_GBP'] = exposure_df['GBP_Exposure'] - exposure_df['Bond_Amount_GBP']
    
    exposure_df['EUR_Exposure'] = exposure_df['GBP_Exposure'] * exposure_df['GBP_EUR_Rate']
    exposure_df['Uncovered_Exposure_EUR'] = exposure_df['Uncovered_Exposure_GBP'] * exposure_df['GBP_EUR_Rate']
    
    # Calculate Monthly VaR in EUR only on the uncovered (positive) exposure
    exposure_df['Monthly_VaR_EUR'] = exposure_df['Uncovered_Exposure_EUR'].clip(lower=0) * exposure_df['VaR_Percentage']
    
    # Calculate monthly loss scenarios (expected loss)
    exposure_df['Expected_Loss_EUR'] = exposure_df['Monthly_VaR_EUR']
    
    print("\n--- Analysis Results (All Months) ---")
    display_df = exposure_df[['GBP_Exposure', 'Uncovered_Exposure_GBP', 'GBP_EUR_Rate', 'VaR_Percentage', 'Monthly_VaR_EUR']].copy()
    display_df['GBP_Exposure'] = display_df['GBP_Exposure'].apply(lambda x: f'£{x/1e6:.1f}M')
    display_df['Uncovered_Exposure_GBP'] = display_df['Uncovered_Exposure_GBP'].apply(lambda x: f'£{x/1e6:+.1f}M')
    display_df['GBP_EUR_Rate'] = display_df['GBP_EUR_Rate'].apply(lambda x: f'{x:.4f}')
    display_df['VaR_Percentage'] = display_df['VaR_Percentage'].apply(lambda x: f'{x*100:.1f}%')
    display_df['Monthly_VaR_EUR'] = display_df['Monthly_VaR_EUR'].apply(lambda x: f'€{x/1e6:.1f}M')
    print(display_df.to_string())
    
    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Average Monthly GBP Exposure: £{exposure_df['GBP_Exposure'].mean():,.0f}")
    print(f"Average Uncovered Exposure: £{exposure_df['Uncovered_Exposure_GBP'].mean():,.0f}")
    print(f"Average GBP/EUR Rate: {exposure_df['GBP_EUR_Rate'].mean():.4f}")
    print(f"Average VaR (Monthly): {exposure_df['VaR_Percentage'].mean()*100:.1f}%")
    print(f"Average Monthly VaR (EUR): €{exposure_df['Monthly_VaR_EUR'].mean():,.0f}")
    print(f"Maximum Monthly VaR (EUR): €{exposure_df['Monthly_VaR_EUR'].max():,.0f}")
    print(f"Total Annual Risk Exposure: €{exposure_df['Monthly_VaR_EUR'].sum():,.0f}")

    # --- Plotting ---
    plot_results(exposure_df)
    
    return exposure_df


def plot_results(df):
    """
    Generates and displays comprehensive plots for the analysis.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    fig.suptitle('Monthly GBP Currency Exposure Analysis (Monte Carlo VaR with Rolling Windows)', 
                 fontsize=16, y=0.995)
    
    formatter_gbp = mticker.FuncFormatter(lambda x, p: f'£{x/1e6:.1f}M')
    formatter_eur = mticker.FuncFormatter(lambda x, p: f'€{x/1e6:.1f}M')
    formatter_pct = mticker.PercentFormatter(decimals=1)
    
    # Plot 1: Booking Curve
    ax1 = axes[0]
    ax1.bar(df.index, df['GBP_Exposure'], label='Monthly GBP Exposure', color='skyblue', width=20)
    ax1.axhline(y=df['Bond_Amount_GBP'].iloc[0], color='r', linestyle='--', linewidth=2,
                label=f'Fixed Bond (£{df["Bond_Amount_GBP"].iloc[0]/1e6:.1f}M)')
    ax1.set_title('Booking Curve: Monthly GBP Exposure vs. Fixed Bond', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Amount (GBP)')
    ax1.yaxis.set_major_formatter(formatter_gbp)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Plot 2: Uncovered Exposure
    ax2 = axes[1]
    uncovered_colors = ['lightcoral' if x > 0 else 'lightgreen' for x in df['Uncovered_Exposure_GBP']]
    ax2.bar(df.index, df['Uncovered_Exposure_GBP'], color=uncovered_colors, width=20)
    ax2.set_title('Uncovered Exposure (Exposure - Bond)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Amount (GBP)')
    ax2.yaxis.set_major_formatter(formatter_gbp)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.legend(handles=[
        plt.Rectangle((0,0),1,1, color='lightcoral', label='Exposure > Bond (Uncovered)'),
        plt.Rectangle((0,0),1,1, color='lightgreen', label='Exposure < Bond (Covered)')
    ], loc='upper left')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # Plot 3: Rolling VaR Percentage
    ax3 = axes[2]
    ax3.plot(df.index, df['VaR_Percentage'] * 100, marker='o', linestyle='-', 
            color='darkorange', linewidth=2, markersize=6, label='VaR % (21-day horizon)')
    ax3.fill_between(df.index, 0, df['VaR_Percentage'] * 100, alpha=0.3, color='darkorange')
    ax3.set_title('Rolling Monthly VaR % (95% Confidence, 21-day horizon)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('VaR (%)')
    ax3.yaxis.set_major_formatter(formatter_pct)
    ax3.legend(loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # --- DYNAMIC Y-AXIS SCALING ---
    # Calculate dynamic y-axis limits based on your data
    var_values = df['VaR_Percentage'] * 100
    var_max = var_values.max()
    var_min = var_values.min()

    # Add some padding (10% of the range or 2% minimum)
    padding = max((var_max - var_min) * 0.1, 2)

    # Set y-axis limits with dynamic padding
    y_lower = max(0, var_min - padding)  # Don't go below 0%
    y_upper = var_max + padding

    # Add realistic upper bound reference line at 25% if your values approach it
    if var_max > 20:  # If values are getting close to realistic upper bound
        ax3.axhline(y=25, color='red', linestyle=':', alpha=0.7, 
                    label='Realistic VaR Upper Bound (25%)')

    ax3.set_ylim(y_lower, y_upper)
    
    # Add realistic VaR bounds
    ax3.axhline(y=25, color='red', linestyle=':', alpha=0.7, label='Realistic VaR Upper Bound')
    ax3.set_ylim(0, 30)

    # Plot 4: Monthly VaR in EUR
    ax4 = axes[3]
    ax4.bar(df.index, df['Monthly_VaR_EUR'], color='purple', alpha=0.7, width=20, 
            label='Monthly VaR (95% confidence)')
    ax4.set_title('Potential Monthly Risk (Value at Risk in EUR)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Potential Loss (EUR)')
    ax4.yaxis.set_major_formatter(formatter_eur)
    ax4.legend(loc='upper left')
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Format x-axis
    plt.xticks(df.index, [d.strftime('%Y-%m') for d in df.index], rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


if __name__ == '__main__':
    # Uncomment the line below to use interactive input
    # monthly_exposures, bond_amount = get_user_input()

    print("--- Using Sample Data for Demonstration ---")
    today = datetime.today()
    sample_exposures = {
        pd.to_datetime(today - timedelta(days=12*30)).to_period('M').end_time.date(): 33_800_000,
        pd.to_datetime(today - timedelta(days=11*30)).to_period('M').end_time.date(): 33_900_000,
        pd.to_datetime(today - timedelta(days=10*30)).to_period('M').end_time.date(): 27_300_000,
        pd.to_datetime(today - timedelta(days=9*30)).to_period('M').end_time.date(): 43_800_000,
        pd.to_datetime(today - timedelta(days=8*30)).to_period('M').end_time.date(): 48_500_000,
        pd.to_datetime(today - timedelta(days=7*30)).to_period('M').end_time.date(): 58_600_000,
        pd.to_datetime(today - timedelta(days=6*30)).to_period('M').end_time.date(): 56_800_000,
        pd.to_datetime(today - timedelta(days=5*30)).to_period('M').end_time.date(): 58_500_000,
        pd.to_datetime(today - timedelta(days=4*30)).to_period('M').end_time.date(): 60_500_000,
        pd.to_datetime(today - timedelta(days=3*30)).to_period('M').end_time.date(): 60_200_000,
        pd.to_datetime(today - timedelta(days=2*30)).to_period('M').end_time.date(): 48_800_000,
        pd.to_datetime(today - timedelta(days=1*30)).to_period('M').end_time.date(): 45_000_000,
    }
    bond_amount = 15_000_000
    print(f"Sample Bond Amount: £{bond_amount:,.0f}\n")
    
    # Run with standard parameters: 252 trading days lookback, 21-day horizon
    results_df = run_analysis(sample_exposures, bond_amount, lookback_days=252, time_horizon_days=21)