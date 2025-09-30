import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from datetime import datetime
import warnings
import yfinance as yf

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set page configuration for a better user experience
st.set_page_config(
    page_title="FX Portfolio Risk Analysis",
    page_icon="📊",
    layout="wide"
)

class FXPortfolioAnalyzer:
    """
    A class to encapsulate the logic for FX portfolio risk analysis,
    including VaR calculations, Monte Carlo simulations, and efficient frontier analysis.
    """
    def __init__(self):
        # Individual volatilities (annualized) from the Citi report (Page 5)
        self.currency_volatilities = {
            'AUD': 0.067, 'DKK': 0.011, 'GBP': 0.054, 'NOK': 0.075, 'PLN': 0.048,
            'USD': 0.068, 'CHF': 0.048, 'CZK': 0.035, 'HKD': 0.069, 'HUF': 0.069,
            'SEK': 0.058, 'ZAR': 0.103, 'EUR': 0.060  # Assumed vol for EUR for matrix completeness
        }

        # FX pairs for fetching historical data from Yahoo Finance (vs. EUR)
        self.fx_pairs = {
            'AUD': 'AUDEUR=X', 'DKK': 'EURDKK=X', 'GBP': 'GBPEUR=X', 'NOK': 'EURNOK=X',
            'PLN': 'EURPLN=X', 'USD': 'USDEUR=X', 'CHF': 'CHFEUR=X', 'CZK': 'EURCZK=X',
            'HKD': 'EURHKD=X', 'HUF': 'EURHUF=X', 'SEK': 'EURSEK=X', 'ZAR': 'EURZAR=X'
        }

        # Initialize with a default correlation matrix and empty historical data
        self.correlation_matrix = self._generate_default_correlation_matrix()
        self.historical_correlations = None
        self.historical_data = None

    def _make_matrix_positive_definite(self, matrix):
        min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
        if min_eig < 1e-8:
            matrix += np.eye(matrix.shape[0]) * (1e-8 - min_eig)
        return matrix

    def _generate_default_correlation_matrix(self):
        currencies = list(self.currency_volatilities.keys())
        n = len(currencies)
        corr_matrix = np.full((n, n), 0.3)
        np.fill_diagonal(corr_matrix, 1.0)
        correlations = {
            ('DKK', 'SEK'): 0.8, ('DKK', 'NOK'): 0.75, ('NOK', 'SEK'): 0.7,
            ('EUR', 'DKK'): 0.9, ('EUR', 'CHF'): 0.8, ('EUR', 'SEK'): 0.65,
            ('PLN', 'CZK'): 0.7, ('PLN', 'HUF'): 0.65, ('CZK', 'HUF'): 0.6,
            ('USD', 'CHF'): 0.7, ('USD', 'GBP'): 0.6, ('AUD', 'ZAR'): 0.5,
        }
        for (c1, c2), val in correlations.items():
            if c1 in currencies and c2 in currencies:
                idx1, idx2 = currencies.index(c1), currencies.index(c2)
                corr_matrix[idx1, idx2] = corr_matrix[idx2, idx1] = val
        corr_matrix = self._make_matrix_positive_definite(corr_matrix)
        return pd.DataFrame(corr_matrix, index=currencies, columns=currencies)

    def fetch_historical_data(self, period="6mo"):
        st.info(f"Fetching historical data for period: {period}...")
        historical_returns = {}
        failed_pairs = []
        for currency, symbol in self.fx_pairs.items():
            try:
                data = yf.Ticker(symbol).history(period=period)
                if data.empty or len(data) < 10:
                    failed_pairs.append(currency)
                    continue
                returns = data['Close'].pct_change().dropna()
                if symbol.startswith('EUR'):
                    returns = -returns
                historical_returns[currency] = returns
            except Exception:
                failed_pairs.append(currency)
        if failed_pairs:
            st.warning(f"Could not fetch data for: {', '.join(failed_pairs)}. Default correlations will be used for these pairs.")
        self.historical_data = historical_returns
        return historical_returns

    def calculate_historical_correlations(self, period="6mo"):
        if self.historical_data is None:
            self.fetch_historical_data(period)
        if not self.historical_data:
            st.error("No historical data available. Using default correlations.")
            return self.correlation_matrix
        returns_df = pd.DataFrame(self.historical_data).dropna()
        historical_corr = returns_df.corr()
        full_corr_matrix = self.correlation_matrix.copy()
        for c1 in historical_corr.columns:
            for c2 in historical_corr.columns:
                if c1 in full_corr_matrix.index and c2 in full_corr_matrix.columns and not np.isnan(historical_corr.loc[c1, c2]):
                    full_corr_matrix.loc[c1, c2] = historical_corr.loc[c1, c2]
        full_corr_matrix_values = self._make_matrix_positive_definite(full_corr_matrix.values)
        self.historical_correlations = pd.DataFrame(
            full_corr_matrix_values,
            index=full_corr_matrix.index,
            columns=full_corr_matrix.columns
        )
        return self.historical_correlations

    def calculate_parametric_var(self, exposures, confidence_level=0.95, time_horizon=0.5, use_historical=False):
        active_currencies = [c for c, e in exposures.items() if e != 0 and c != 'EUR']
        if not active_currencies:
            return 0.0, {}, {}

        corr_matrix = self.historical_correlations if use_historical and self.historical_correlations is not None else self.correlation_matrix
        
        vols = np.array([self.currency_volatilities[c] for c in active_currencies])
        exposures_arr = np.array([exposures[c] for c in active_currencies])
        
        # Ensure the subset of the correlation matrix is valid
        active_corr_matrix = corr_matrix.loc[active_currencies, active_currencies]
        corr_subset = self._make_matrix_positive_definite(active_corr_matrix.values)
        
        cov_matrix = np.outer(vols, vols) * corr_subset
        portfolio_variance = exposures_arr.T @ cov_matrix @ exposures_arr
        
        # Ensure variance is not negative due to floating point inaccuracies
        portfolio_variance = max(0, portfolio_variance)
        portfolio_vol = np.sqrt(portfolio_variance)
        
        horizon_vol = portfolio_vol * np.sqrt(time_horizon)
        z_score = stats.norm.ppf(confidence_level)
        portfolio_var = z_score * horizon_vol
        
        if portfolio_vol > 1e-9:
            marginal_var_array = (cov_matrix @ exposures_arr) / portfolio_vol * z_score * np.sqrt(time_horizon)
        else:
            marginal_var_array = np.zeros_like(exposures_arr, dtype=float)

        component_var = {c: exposures_arr[i] * marginal_var_array[i] for i, c in enumerate(active_currencies)}
        contribution_pct = {c: (v / portfolio_var * 100) if portfolio_var > 1e-9 else 0 for c, v in component_var.items()}
        
        return portfolio_var, component_var, contribution_pct

    def calculate_individual_var(self, exposures, confidence_level=0.95, time_horizon=0.5):
        individual_vars = {}
        for currency, exposure in exposures.items():
            if currency != 'EUR' and exposure != 0:
                vol = self.currency_volatilities[currency]
                individual_vars[currency] = abs(exposure) * vol * np.sqrt(time_horizon) * stats.norm.ppf(confidence_level)
        return individual_vars

    def monte_carlo_simulation(self, exposures, num_simulations=10000, time_horizon=0.5, use_historical=False):
        active_currencies = [c for c, e in exposures.items() if e != 0 and c != 'EUR']
        if not active_currencies:
            return np.zeros(num_simulations)
        corr_matrix = self.historical_correlations if use_historical and self.historical_correlations is not None else self.correlation_matrix
        vols = np.array([self.currency_volatilities[c] for c in active_currencies])
        exposures_arr = np.array([exposures[c] for c in active_currencies])
        corr_subset = self._make_matrix_positive_definite(corr_matrix.loc[active_currencies, active_currencies].values)
        L = np.linalg.cholesky(corr_subset)
        uncorrelated_returns = np.random.normal(0, 1, (num_simulations, len(active_currencies)))
        correlated_returns = uncorrelated_returns @ L.T
        scaled_returns = correlated_returns * vols * np.sqrt(time_horizon)
        portfolio_returns = scaled_returns @ exposures_arr
        return portfolio_returns
    
    def calculate_efficient_frontier(self, exposures, time_horizon=0.5, num_points=20, use_historical=False):
        # --- ROBUST EFFICIENT FRONTIER CALCULATION ---
        # This function has been rewritten to avoid sending a zero-exposure portfolio
        # to the VaR calculation function, which was the source of the error.
        
        # 1. Calculate the starting (unhedged) and ending (fully hedged) points
        unhedged_var, _, _ = self.calculate_parametric_var(exposures, time_horizon=time_horizon, use_historical=use_historical)
        unhedged_point = (0, unhedged_var)
        
        carry_rates = {
            'AUD': -0.0162, 'DKK': 0.0035, 'GBP': -0.0203, 'NOK': -0.0199, 'PLN': -0.0268,
            'USD': -0.0196, 'CHF': 0.0211, 'CZK': -0.0123, 'HKD': -0.0129, 'HUF': -0.0430,
            'SEK': 0.0009, 'ZAR': -0.0477
        }
        fully_hedged_cost = -sum(exposures.get(c, 0) * carry_rates.get(c, 0) for c in exposures) * time_horizon
        fully_hedged_point = (fully_hedged_cost, 0.0)

        # 2. Initialize lists with the starting point
        cost_points = [unhedged_point[0]]
        var_points = [unhedged_point[1]]
        
        # 3. Loop through intermediate hedge ratios, but stop before 100%
        # This prevents passing a zero-exposure portfolio to the VaR function.
        hedge_ratios = np.linspace(0, 1, num_points)[1:-1]
        
        for ratio in hedge_ratios:
            hedged_exposures = {c: e * (1 - ratio) for c, e in exposures.items()}
            var, _, _ = self.calculate_parametric_var(hedged_exposures, time_horizon=time_horizon, use_historical=use_historical)
            cost = -sum(exposures.get(c, 0) * carry_rates.get(c, 0) * ratio for c in exposures) * time_horizon
            cost_points.append(cost)
            var_points.append(var)
        
        # 4. Manually add the final, fully hedged point
        cost_points.append(fully_hedged_point[0])
        var_points.append(fully_hedged_point[1])
        
        return cost_points, var_points

def main():
    st.title("FX Portfolio Risk Analysis 📊")
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FXPortfolioAnalyzer()
    analyzer = st.session_state.analyzer

    st.sidebar.header("Portfolio & Analysis Setup")

    if 'exposures' not in st.session_state:
        st.session_state.exposures = {
            'AUD': 4897283, 'DKK': 26459697, 'GBP': 227665224, 'NOK': 22402481,
            'PLN': 6434597, 'USD': -36068199, 'CHF': 81333542, 'CZK': 1430589,
            'HKD': 1050, 'HUF': 12098801, 'SEK': 53791300, 'ZAR': 1217337,
        }

    st.sidebar.subheader("Currency Exposures (EUR Equivalent)")
    exposures = {}
    for currency in sorted(st.session_state.exposures.keys()):
        exposures[currency] = st.sidebar.number_input(
            f"{currency} Exposure",
            value=st.session_state.exposures[currency],
            key=f"exp_{currency}",
            step=100000,
            format="%d"
        )
    st.session_state.exposures = exposures

    st.sidebar.subheader("Risk Parameters")
    confidence_level = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, help="The probability level for the VaR calculation. 95% is standard.")
    time_horizon = st.sidebar.slider("Time Horizon (Years)", 0.25, 2.0, 0.5, 0.25, help="The time frame for the risk assessment. The report uses 6 months (0.5 years).")
    num_simulations = st.sidebar.select_slider("Monte Carlo Simulations", options=[1000, 5000, 10000, 20000, 50000], value=10000)

    st.sidebar.subheader("Correlation Settings")
    correlation_period = st.sidebar.selectbox("Historical Period", ["3mo", "6mo", "1y", "2y", "5y"], index=1, help="The look-back period for calculating historical correlations. The report uses 6 months.")
    use_historical = st.sidebar.checkbox("Use Live Historical Correlations", value=True, help="If checked, fetches live market data to compute correlations. If unchecked, uses a default matrix.")

    if use_historical and st.sidebar.button("Fetch Market Data & Recalculate"):
        with st.spinner("Fetching data and computing correlations..."):
            analyzer.calculate_historical_correlations(correlation_period)
        st.sidebar.success("Correlations updated!")
    
    if use_historical and analyzer.historical_correlations is None:
        with st.spinner("Fetching initial market data..."):
            analyzer.calculate_historical_correlations(correlation_period)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Portfolio Overview", "📊 Parametric VaR", "🎲 Monte Carlo", "🛡️ Efficient Frontier", "🔗 Correlations"
    ])

    portfolio_var, _, _ = analyzer.calculate_parametric_var(exposures, confidence_level, time_horizon, use_historical)
    individual_vars = analyzer.calculate_individual_var(exposures, confidence_level, time_horizon)
    sum_individual_var = sum(individual_vars.values())
    diversification_benefit = sum_individual_var - portfolio_var
    diversification_pct = (diversification_benefit / sum_individual_var * 100) if sum_individual_var > 0 else 0

    with tab1:
        st.header("Portfolio Overview")
        st.markdown("A summary of the current currency exposures. The report's functional currency is EUR.")
        col1, col2, col3 = st.columns(3)
        total_long = sum(v for v in exposures.values() if v > 0)
        total_short = sum(abs(v) for v in exposures.values() if v < 0)
        net_exposure = sum(exposures.values())
        col1.metric("Total Long Exposure", f"€{total_long:,.0f}")
        col2.metric("Total Short Exposure", f"€{total_short:,.0f}")
        col3.metric("Net Exposure", f"€{net_exposure:,.0f}")
        exp_df = pd.DataFrame(exposures.items(), columns=['Currency', 'Exposure']).sort_values('Exposure', ascending=False)
        fig_exp = px.bar(exp_df, x='Currency', y='Exposure', title='EUR Equivalent Exposure by Currency',
                         color='Exposure', color_continuous_scale='RdBu_r',
                         labels={'Exposure': 'Exposure (EUR)'})
        st.plotly_chart(fig_exp, use_container_width=True)

    with tab2:
        st.header(f"Parametric VaR Analysis ({confidence_level:.0%})")
        st.markdown("This analysis uses the **variance-covariance method**, assuming returns follow a normal distribution, to calculate the portfolio's Value at Risk (VaR).")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolio VaR", f"€{portfolio_var:,.0f}", help="The maximum expected loss for the entire portfolio after accounting for diversification.")
        col2.metric("Undiversified VaR", f"€{sum_individual_var:,.0f}", help="The sum of the VaRs of each currency position, calculated independently.")
        col3.metric("Diversification Benefit", f"€{diversification_benefit:,.0f}", f"{diversification_pct:.1f}%", help="The risk reduction achieved by holding a diversified portfolio.")
        
        _, _, contribution_pct = analyzer.calculate_parametric_var(exposures, confidence_level, time_horizon, use_historical)
        if contribution_pct:
            contrib_df = pd.DataFrame(contribution_pct.items(), columns=['Currency', 'Contribution']).sort_values('Contribution', ascending=False)
            fig_contrib = px.bar(contrib_df, x='Currency', y='Contribution', title='Contribution to Portfolio Risk (%)',
                                 color='Contribution', color_continuous_scale='viridis',
                                 labels={'Contribution': 'Contribution to Risk (%)'})
            st.plotly_chart(fig_contrib, use_container_width=True)
            st.markdown("**Component VaR** measures each position's contribution to the total portfolio VaR in currency terms.")
        else:
            st.info("No active exposures to analyze for risk contribution.")

    with tab3:
        st.header("Monte Carlo Simulation")
        st.markdown(f"This simulation models potential portfolio outcomes by generating thousands of random market scenarios ({num_simulations:,} runs).")
        with st.spinner("Running Monte Carlo simulation..."):
            portfolio_returns = analyzer.monte_carlo_simulation(exposures, num_simulations, time_horizon, use_historical)
        simulated_var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        expected_shortfall = -portfolio_returns[portfolio_returns <= -simulated_var].mean() if any(portfolio_returns <= -simulated_var) else 0
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Simulated {confidence_level:.0%} VaR", f"€{simulated_var:,.0f}")
        col2.metric("Expected Shortfall (CVaR)", f"€{expected_shortfall:,.0f}", help="The average loss when the VaR threshold is breached.")
        col3.metric("Difference from Parametric VaR", f"€{simulated_var - portfolio_var:,.0f}")
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=portfolio_returns, nbinsx=100, name="P&L Distribution", marker_color='royalblue'))
        fig_mc.add_vline(x=-simulated_var, line_dash="dash", line_color="red", annotation_text=f"MC VaR: €{-simulated_var:,.0f}")
        fig_mc.update_layout(title="Monte Carlo Simulation - Portfolio P&L Distribution", xaxis_title="Profit/Loss (EUR)", yaxis_title="Frequency")
        st.plotly_chart(fig_mc, use_container_width=True)

    with tab4:
       # st.header("Efficient Frontier for Hedging")
       # st.markdown("The **Efficient Frontier** illustrates the optimal trade-off between reducing portfolio risk (VaR) and the cost or benefit of hedging using FX forwards.")
       # cost_points, var_points = analyzer.calculate_efficient_frontier(exposures, time_horizon, use_historical=use_historical)
        
       # fig_ef = go.Figure()
       # fig_ef.add_trace(go.Scatter(x=cost_points, y=var_points, mode='lines+markers', name='Efficient Frontier'))
       # fig_ef.add_trace(go.Scatter(x=[cost_points[0]], y=[var_points[0]], mode='markers', name='Unhedged', marker=dict(color='red', size=12)))
       # fig_ef.add_trace(go.Scatter(x=[cost_points[-1]], y=[var_points[-1]], mode='markers', name='Fully Hedged', marker=dict(color='green', size=12)))
        
       # zero_cost_idx = np.argmin(np.abs(cost_points))
       # zero_cost_var = var_points[zero_cost_idx]
       # fig_ef.add_annotation(x=cost_points[zero_cost_idx], y=var_points[zero_cost_idx], text=f"Zero-Cost Hedge<br>VaR: €{zero_cost_var:,.0f}", showarrow=True, arrowhead=1)
        
      #  fig_ef.update_layout(
      #      title="VaR vs. Hedging Costs/Benefits",
      #      xaxis_title="Hedging Costs / Benefits (EUR)",
      #      yaxis_title=f"{confidence_level:.0%} C.L. VaR (EUR)",
      #      yaxis_zeroline=False,
      #      xaxis_zeroline=True,
      #  )
      #  st.plotly_chart(fig_ef, use_container_width=True)
      #  st.markdown(f"The analysis shows a potential **zero-cost hedge** that can reduce VaR to **€{zero_cost_var:,.0f}** from an unhedged VaR of **€{var_points[0]:,.0f}**.")
         st.markdown("PLACEHOLDER")
    with tab5:
        st.header("Correlation Analysis")
        st.markdown("Currency correlations are a critical driver of portfolio diversification. Low or negative correlation can significantly reduce overall risk.")
        corr_matrix = analyzer.historical_correlations if use_historical and analyzer.historical_correlations is not None else analyzer.correlation_matrix
        active_currencies = [c for c, e in exposures.items() if e != 0 and c != 'EUR']
        if active_currencies:
            corr_subset = corr_matrix.loc[active_currencies, active_currencies]
            fig_corr = px.imshow(
                corr_subset,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title=f"Currency Correlation Matrix ({'Historical' if use_historical else 'Default'})",
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("No active currency exposures to analyze.")

if __name__ == "__main__":
    main()