# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 19:08:19 2026

@author: gsirtori
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_deposit_risk(file_path):
    print("="*80)
    print("      DEPOSIT RISK & FINANCIAL IMPACT ANALYSIS")
    print("="*80)

    # 1. Load Data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # 2. Feature Engineering
    # -----------------------
    # Calculate Deposit Percentage
    df['PERC_DEPOSIT'] = df['MIN_TRANSACTION'] / df['GTV_AMOUNT_MCX']
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['PERC_DEPOSIT'])

    # Calculate Total Cost Unpaid (Summing components)
    cost_cols = ['TOTAL_COST_Pure_To_Resp', 'TOTAL_COST_Bad_debts', 
                 'TOTAL_COST_Unpaid_Vol', 'TOTAL_COST_Unpaid_Inv']
    df['TOTAL_COST_UNPAID'] = df[cost_cols].sum(axis=1)

    # Prepare Categorical Variable (Refundable vs Non-Refundable)
    df['is_refundable'] = (df['is_hotel_refundable'] == 'HOTEL REFUNDABLE').astype(int)
    
    # Target Variable for Probability
    df['UNPAID_FLAG_INT'] = df['UNPAID_FLAG'].astype(int)

    print(f"Data successfully loaded. Analyzing {len(df):,} bookings.\n")

    # ---------------------------------------------------------
    # PART A: Probability of Default (Logistic Regression)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" PART A: Probability of Default (Logistic Regression)")
    print("="*50)
    print("GOAL: Determine how Deposit % affects the likelihood of a booking going unpaid.")
    print("FORMULA: P(Default) = 1 / (1 + exp(-(beta0 + beta1*Deposit + beta2*Refundable)))")
    
    y_prob = df['UNPAID_FLAG_INT']
    X_prob = sm.add_constant(df[['PERC_DEPOSIT', 'is_refundable']])
    logit_model = sm.Logit(y_prob, X_prob).fit(disp=0)
    
    print(logit_model.summary())
    
    margeff = logit_model.get_margeff()
    print("\nMarginal Effects (Impact of 100% change):")
    print(margeff.summary())

    # ---------------------------------------------------------
    # PART B: Cost of Default (Linear Regression)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" PART B: Cost of Default (Conditional on Unpaid)")
    print("="*50)
    print("GOAL: Estimate financial loss IF a booking goes unpaid.")
    print("FORMULA: Cost = alpha + beta1*Deposit + beta2*Refundable + error")
    
    df_unpaid = df[df['UNPAID_FLAG'] == True].copy()
    y_cost = df_unpaid['TOTAL_COST_UNPAID']
    X_cost = sm.add_constant(df_unpaid[['PERC_DEPOSIT', 'is_refundable']])
    ols_model = sm.OLS(y_cost, X_cost).fit()
    
    print(ols_model.summary())

    # ---------------------------------------------------------
    # PART C: Jan 2026 Baseline & Impact Analysis
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print(" PART C: TRUE BASELINE COMPARISON (JAN 2025 vs JAN 2026)")
    print("="*80)
    
    # 1. Define Cohorts
    df_jan25 = df[df['BK_DATE_YEAR_MONTH'] == 202501]
    df_jan26 = df[df['BK_DATE_YEAR_MONTH'] == 202601].copy()

    if df_jan26.empty:
        print("No data for Jan 2026 found.")
        return

    # 2. Calculate Shift
    avg_dep_25 = df_jan25['PERC_DEPOSIT'].mean()
    avg_dep_26 = df_jan26['PERC_DEPOSIT'].mean()
    deposit_shift = avg_dep_25 - avg_dep_26
    pct_drop = (avg_dep_26 - avg_dep_25) / avg_dep_25

    print(f"1. DEPOSIT TREND ANALYSIS")
    print(f"   - Average Deposit Jan 2025: {avg_dep_25:.2%}")
    print(f"   - Average Deposit Jan 2026: {avg_dep_26:.2%}")
    print(f"   - Change: {pct_drop:.1%} decrease relative to last year.")
    print(f"   - Absolute Gap: {deposit_shift*100:.2f} percentage points.")
    
    print("\n2. CALCULATION METHODOLOGY")
    print("   We calculate the 'Expected Risk' for every single booking in Jan 2026 using two scenarios:")
    print("   [Scenario A: Actual]   Uses the actual deposit % paid by the customer.")
    print("   [Scenario B: Baseline] Simulates what the risk WOULD be if deposits were restored to 2025 levels.")
    print(f"   Adjustment Formula:    Simulated_Deposit = Actual_Deposit + {deposit_shift:.4f}")
    print("   Risk Calculation:      Expected_Loss = P(Default) * Estimated_Cost")

    # 3. Apply Models
    def calculate_risk(data, deposit_column):
        # Prepare inputs exactly as model expects
        X_in = pd.DataFrame({
            'const': 1.0,
            'PERC_DEPOSIT': data[deposit_column],
            'is_refundable': data['is_refundable']
        })
        X_in = X_in[['const', 'PERC_DEPOSIT', 'is_refundable']] # Ensure order
        
        # Predict
        probs = logit_model.predict(X_in)
        costs = ols_model.predict(X_in)
        
        # Total Risk
        return (probs * costs).sum()

    # Scenario A: Actual Reality
    risk_actual = calculate_risk(df_jan26, 'PERC_DEPOSIT')

    # Scenario B: Simulated Baseline (Restore 2025 levels)
    df_jan26['PERC_DEPOSIT_BASELINE'] = (df_jan26['PERC_DEPOSIT'] + deposit_shift).clip(0, 1)
    risk_baseline = calculate_risk(df_jan26, 'PERC_DEPOSIT_BASELINE')

    # Impact
    impact = risk_actual - risk_baseline

    print("\n3. FINANCIAL IMPACT RESULTS (January 2026)")
    print("-" * 60)
    print(f"   Projected Loss (Current Strategy):     {risk_actual:,.2f} EUR")
    print(f"   Projected Loss (Baseline / 2025 Avg):  {risk_baseline:,.2f} EUR")
    print("-" * 60)
    print(f"   NET COST OF DEPOSIT REDUCTION:         {impact:,.2f} EUR")
    print("-" * 60)
    print(f"\n   CONCLUSION: The {abs(pct_drop):.1%} drop in average deposit coverage is estimated")
    print(f"   to increase unpaid losses by {abs(impact):,.0f} EUR for this month alone.")

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    print("\nGenerating charts...")
    
    # CHART 1: Probability of Default (Bar Chart)
    plt.figure(figsize=(10, 6))
    df['deposit_bin'] = pd.cut(df['PERC_DEPOSIT'], bins=np.linspace(0, 1, 11))
    prob_by_bin = df.groupby('deposit_bin', observed=False)['UNPAID_FLAG_INT'].mean()
    
    prob_by_bin.plot(kind='bar', color='#ff9900', alpha=0.9, edgecolor='black')
    plt.title('Risk Profile: Probability of Default by Deposit Level', fontsize=14)
    plt.xlabel('Deposit Percentage Range', fontsize=12)
    plt.ylabel('Probability of Default', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # CHART 2: Regression Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(x='PERC_DEPOSIT', y='TOTAL_COST_UNPAID', data=df_unpaid, 
                scatter_kws={'alpha': 0.3, 'color': '#0066cc'}, 
                line_kws={'color': 'red', 'label': 'Linear Trend', 'linewidth': 2})
    
    plt.title('Financial Impact: Deposit % vs Cost of Default', fontsize=14)
    plt.xlabel('Deposit Percentage (0.0 - 1.0)', fontsize=12)
    plt.ylabel('Total Cost (Unpaid Bookings)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # CHART 3: Refundability Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_hotel_refundable', y='TOTAL_COST_UNPAID', data=df_unpaid, 
                showfliers=False, palette='Set2')
    
    plt.title('Impact of Hotel Refundability on Loss Severity', fontsize=14)
    plt.xlabel('Hotel Refundability Status', fontsize=12)
    plt.ylabel('Total Cost (Unpaid Bookings)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    print("Analysis Complete.")

# Run the function
if __name__ == "__main__":
    # Replace 'Deposit.csv' with your actual file path
    analyze_deposit_risk('Deposit.csv')