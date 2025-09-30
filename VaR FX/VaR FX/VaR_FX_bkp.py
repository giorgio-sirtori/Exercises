import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="FX Portfolio Risk Analysis",
    page_icon="📊",
    layout="wide"
)

class FXPortfolioAnalyzer:
    def __init__(self):
        # Historical volatility data (annualized)
        self.currency_volatilities = {
            'AUD': 0.135, 'DKK': 0.085, 'GBP': 0.120, 'NOK': 0.140, 'PLN': 0.160,
            'USD': 0.105, 'CHF': 0.095, 'CZK': 0.170, 'HKD': 0.075, 'HUF': 0.180,
            'SEK': 0.150, 'ZAR': 0.200, 'EUR': 0.080
        }
        
        # Corrected FX pairs mapping to Yahoo Finance symbols
        self.fx_pairs = {
            'AUD': 'AUDUSD=X', 'DKK': 'USDDKK=X', 'GBP': 'GBPUSD=X', 'NOK': 'NOKUSD=X',
            'PLN': 'PLNUSD=X', 'USD': 'USDEUR=X', 'CHF': 'CHFUSD=X', 'CZK': 'CZKUSD=X',
            'HKD': 'HKDUSD=X', 'HUF': 'HUFUSD=X', 'SEK': 'SEKUSD=X', 'ZAR': 'ZARUSD=X'
        }
        
        # Initialize with default correlations
        self.correlation_matrix = self._generate_realistic_correlation_matrix()
        self.historical_correlations = None
        self.historical_data = None
        
    def _make_matrix_positive_definite(self, matrix):
        """
        Ensure matrix is positive definite for Cholesky decomposition
        """
        n = matrix.shape[0]
        
        # Add small value to diagonal if needed
        min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
        if min_eig < 1e-8:
            matrix += np.eye(n) * (1e-8 - min_eig)
        
        return matrix
    
    def _generate_realistic_correlation_matrix(self):
        currencies = ['AUD', 'DKK', 'GBP', 'NOK', 'PLN', 'USD', 'CHF', 'CZK', 'HKD', 'HUF', 'SEK', 'ZAR', 'EUR']
        n = len(currencies)
        
        corr_matrix = np.full((n, n), 0.4)
        np.fill_diagonal(corr_matrix, 1.0)
        
        correlations = {
            ('DKK', 'NOK'): 0.8, ('DKK', 'SEK'): 0.85, ('NOK', 'SEK'): 0.75,
            ('EUR', 'DKK'): 0.9, ('EUR', 'SEK'): 0.7, ('EUR', 'NOK'): 0.65,
            ('EUR', 'CHF'): 0.8, ('EUR', 'PLN'): 0.6, ('EUR', 'CZK'): 0.55, ('EUR', 'HUF'): 0.5,
            ('USD', 'HKD'): 0.9, ('USD', 'AUD'): 0.6,
            ('ZAR', 'HUF'): 0.5, ('ZAR', 'PLN'): 0.4,
            ('CHF', 'USD'): 0.7,
            ('AUD', 'NOK'): 0.5, ('AUD', 'ZAR'): 0.4,
        }
        
        for (curr1, curr2), corr_val in correlations.items():
            if curr1 in currencies and curr2 in currencies:
                i1, i2 = currencies.index(curr1), currencies.index(curr2)
                corr_matrix[i1, i2] = corr_val
                corr_matrix[i2, i1] = corr_val
        
        # Ensure matrix is positive definite
        corr_matrix = self._make_matrix_positive_definite(corr_matrix)
        
        return pd.DataFrame(corr_matrix, index=currencies, columns=currencies)
    
    def fetch_historical_data(self, period="6mo"):
        """
        Fetch historical FX data from Yahoo Finance using USD pairs
        """
        st.info(f"Fetching historical data for period: {period}...")
        
        historical_data = {}
        failed_pairs = []
        
        for currency, symbol in self.fx_pairs.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                if not data.empty and len(data) > 10:  # Ensure we have enough data
                    # Calculate daily returns
                    returns = data['Close'].pct_change().dropna()
                    historical_data[currency] = returns
                else:
                    failed_pairs.append(currency)
            except Exception as e:
                failed_pairs.append(currency)
                continue
        
        if failed_pairs:
            st.warning(f"Could not fetch data for: {', '.join(failed_pairs)}. Using default correlations.")
        
        self.historical_data = historical_data
        return historical_data
    
    def calculate_historical_correlations(self, period="6mo"):
        """
        Calculate correlations from historical data
        """
        if self.historical_data is None:
            self.fetch_historical_data(period)
        
        if not self.historical_data:
            st.error("No historical data available. Using default correlations.")
            return self.correlation_matrix
        
        # Create a DataFrame with all returns
        all_returns = {}
        for currency, returns in self.historical_data.items():
            all_returns[currency] = returns
        
        returns_df = pd.DataFrame(all_returns)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Fill missing values with default correlations and ensure positive definite
        all_currencies = list(self.currency_volatilities.keys())
        full_corr_matrix = self.correlation_matrix.copy()
        
        for curr1 in all_currencies:
            for curr2 in all_currencies:
                if curr1 in correlation_matrix.columns and curr2 in correlation_matrix.columns:
                    if not np.isnan(correlation_matrix.loc[curr1, curr2]):
                        full_corr_matrix.loc[curr1, curr2] = correlation_matrix.loc[curr1, curr2]
        
        # Ensure the matrix is positive definite
        full_corr_matrix_values = self._make_matrix_positive_definite(full_corr_matrix.values)
        full_corr_matrix = pd.DataFrame(full_corr_matrix_values, 
                                      index=full_corr_matrix.index, 
                                      columns=full_corr_matrix.columns)
        
        self.historical_correlations = full_corr_matrix
        return full_corr_matrix
    
    def update_correlation(self, currency1, currency2, new_correlation):
        """
        Update a specific correlation value
        """
        if currency1 in self.correlation_matrix.index and currency2 in self.correlation_matrix.columns:
            self.correlation_matrix.loc[currency1, currency2] = new_correlation
            self.correlation_matrix.loc[currency2, currency1] = new_correlation
            
            # Ensure matrix remains positive definite
            corr_matrix_values = self._make_matrix_positive_definite(self.correlation_matrix.values)
            self.correlation_matrix = pd.DataFrame(corr_matrix_values,
                                                 index=self.correlation_matrix.index,
                                                 columns=self.correlation_matrix.columns)
    
    def calculate_variance_covariance_var(self, exposures, confidence_level=0.95, time_horizon=0.5, use_historical=False):
        """
        Calculate VaR using the variance-covariance method
        """
        # Use historical correlations if requested
        if use_historical and self.historical_correlations is not None:
            corr_matrix_to_use = self.historical_correlations
        else:
            corr_matrix_to_use = self.correlation_matrix
        
        # Filter to currencies with non-zero exposure (excluding EUR as base currency)
        active_currencies = [curr for curr, exp in exposures.items() if exp != 0 and curr != 'EUR']
        if not active_currencies:
            return 0, {}, {}, {}
        
        # Get volatilities and create covariance matrix
        volatilities = np.array([self.currency_volatilities[curr] for curr in active_currencies])
        corr_subset = corr_matrix_to_use.loc[active_currencies, active_currencies]
        
        # Ensure the correlation matrix is positive definite
        corr_values = self._make_matrix_positive_definite(corr_subset.values)
        cov_matrix = np.outer(volatilities, volatilities) * corr_values
        
        # Portfolio exposures
        exposures_array = np.array([exposures[curr] for curr in active_currencies])
        
        # Portfolio variance and volatility
        portfolio_variance = exposures_array.T @ cov_matrix @ exposures_array
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Time scaling
        horizon_volatility = portfolio_volatility * np.sqrt(time_horizon)
        
        # VaR calculation
        z_score = stats.norm.ppf(confidence_level)
        portfolio_var = z_score * horizon_volatility
        
        # Calculate component contributions
        component_var = {}
        marginal_var = {}
        contribution_pct = {}
        
        # Marginal VaR: ∂VaR/∂w_i = (cov_matrix @ w) / σ_p * z_score
        marginal_var_array = (cov_matrix @ exposures_array) / portfolio_volatility * z_score * np.sqrt(time_horizon)
        
        for i, currency in enumerate(active_currencies):
            marginal_var[currency] = marginal_var_array[i]
            component_var[currency] = exposures_array[i] * marginal_var_array[i]
            contribution_pct[currency] = (component_var[currency] / portfolio_var * 100) if portfolio_var != 0 else 0
        
        return portfolio_var, component_var, marginal_var, contribution_pct
    
    def calculate_individual_var(self, exposures, confidence_level=0.95, time_horizon=0.5):
        """Calculate individual VaR for each currency (undiversified)"""
        individual_var = {}
        for currency, exposure in exposures.items():
            if currency != 'EUR' and exposure != 0:
                vol = self.currency_volatilities[currency]
                individual_var[currency] = abs(exposure) * vol * np.sqrt(time_horizon) * stats.norm.ppf(confidence_level)
        return individual_var
    
    def monte_carlo_simulation(self, exposures, num_simulations=10000, time_horizon=0.5, use_historical=False):
        """
        Run Monte Carlo simulation for portfolio returns
        """
        # Use historical correlations if requested
        if use_historical and self.historical_correlations is not None:
            corr_matrix_to_use = self.historical_correlations
        else:
            corr_matrix_to_use = self.correlation_matrix
        
        active_currencies = [curr for curr, exp in exposures.items() if exp != 0 and curr != 'EUR']
        if not active_currencies:
            return np.zeros(num_simulations)
        
        n = len(active_currencies)
        volatilities = np.array([self.currency_volatilities[curr] for curr in active_currencies])
        exposures_array = np.array([exposures[curr] for curr in active_currencies])
        
        # Get correlation matrix for active currencies
        corr_subset = corr_matrix_to_use.loc[active_currencies, active_currencies]
        
        # Ensure matrix is positive definite
        corr_values = self._make_matrix_positive_definite(corr_subset.values)
        
        # Generate correlated random returns using Cholesky decomposition
        try:
            L = np.linalg.cholesky(corr_values)
        except np.linalg.LinAlgError:
            # If still not positive definite, use nearest correlation matrix
            L = np.linalg.cholesky(corr_values + np.eye(n) * 0.01)
        
        uncorrelated_returns = np.random.normal(0, 1, (num_simulations, n))
        correlated_returns = uncorrelated_returns @ L.T
        
        # Scale by volatility and time
        scaled_returns = correlated_returns * volatilities * np.sqrt(time_horizon)
        
        # Calculate portfolio P&L
        portfolio_returns = scaled_returns @ exposures_array
        
        return portfolio_returns
    
    def calculate_efficient_frontier(self, exposures, time_horizon=0.5, num_points=20, use_historical=False):
        """
        Calculate efficient frontier for hedging strategies
        """
        base_var, _, _, _ = self.calculate_variance_covariance_var(
            exposures, time_horizon=time_horizon, use_historical=use_historical
        )
        
        # Generate different hedging scenarios
        hedge_ratios = np.linspace(0, 1, num_points)
        var_reduction = []
        carry_costs = []
        
        for ratio in hedge_ratios:
            residual_risk_ratio = 0.1
            reduced_var = base_var * (1 - ratio * (1 - residual_risk_ratio))
            var_reduction.append(reduced_var)
            
            carry_rates = {
                'AUD': -0.02, 'GBP': -0.01, 'NOK': -0.015, 'PLN': -0.025,
                'USD': 0.01, 'CHF': 0.005, 'CZK': -0.03, 'HKD': 0.008,
                'HUF': -0.035, 'SEK': 0.005, 'ZAR': -0.04, 'DKK': 0.003
            }
            
            total_carry = 0
            for curr, exposure in exposures.items():
                if curr in carry_rates and curr != 'EUR':
                    total_carry += exposure * carry_rates[curr] * ratio
            
            carry_costs.append(-total_carry * time_horizon)
        
        return hedge_ratios, var_reduction, carry_costs

def main():
    st.title("FX Portfolio Risk Analysis")
      
    # Initialize analyzer
    analyzer = FXPortfolioAnalyzer()
    
    # Sidebar for inputs
    st.sidebar.header("Portfolio Configuration")
    
    # Default currencies matching the Citi analysis
    default_exposures = {
        'AUD': 4497463, 'DKK': 264059697, 'GBP': 227466224, 'NOK': 225050947,
        'PLN': 6434627, 'USD': 360881668, 'CHF': 91333541, 'CZK': 1430589,
        'HKD': 1406, 'HUF': 12098801, 'SEK': 53731929, 'ZAR': 1217332,
        'EUR': 0
    }
    
    exposures = {}
    
    # Allow users to add/remove currencies
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        all_currencies = list(analyzer.currency_volatilities.keys())
        available_currencies = [c for c in all_currencies if c not in default_exposures]
        new_currency = st.selectbox(
            "Add Currency",
            options=available_currencies
        )
    
    with col2:
        new_exposure = st.number_input("Exposure Amount", value=0, step=50000)
    
    if st.sidebar.button("Add Currency"):
        if new_currency and new_exposure != 0:
            default_exposures[new_currency] = new_exposure
            st.sidebar.success(f"Added {new_currency}: {new_exposure:,}")
    
    # Display current exposures for editing
    st.sidebar.subheader("Current Exposures")
    for currency in list(default_exposures.keys()):
        if currency == 'EUR':
            continue
            
        col1, col2, col3 = st.sidebar.columns([2, 2, 1])
        with col1:
            st.write(currency)
        with col2:
            new_val = st.number_input(
                f"{currency} Amount", 
                value=default_exposures[currency],
                key=f"exp_{currency}",
                step=50000
            )
            default_exposures[currency] = new_val
        with col3:
            if st.button("×", key=f"del_{currency}"):
                default_exposures.pop(currency)
                st.rerun()
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    confidence_level = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95)
    time_horizon = st.sidebar.slider("Time Horizon (Years)", 0.25, 2.0, 0.5)
    num_simulations = st.sidebar.slider("Monte Carlo Simulations", 1000, 50000, 10000)
    
    # Correlation analysis parameters
    st.sidebar.subheader("Correlation Analysis")
    correlation_period = st.sidebar.selectbox(
        "Historical Period for Correlations",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=2
    )
    
    use_historical_correlations = st.sidebar.checkbox("Use Historical Correlations", value=False)
    
    if st.sidebar.button("Calculate Historical Correlations"):
        with st.spinner("Fetching historical data and calculating correlations..."):
            analyzer.calculate_historical_correlations(correlation_period)
        st.sidebar.success("Historical correlations calculated!")
    
    # Use the current exposures for analysis
    exposures = {k: v for k, v in default_exposures.items() if v != 0}
    
    if not exposures or len([k for k in exposures.keys() if k != 'EUR']) == 0:
        st.warning("Please add some currency exposures (other than EUR) to begin analysis.")
        return
    
    # Main analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio Overview", "VaR Analysis", "Monte Carlo Simulation", 
        "Efficient Frontier", "Correlation Analysis"
    ])
    
    with tab1:
        st.header("Portfolio Overview")
        
        total_exposure = sum([v for k, v in exposures.items() if k != 'EUR'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Exposure", f"€{total_exposure:,.0f}")
        with col2:
            num_curr = len([k for k in exposures.keys() if k != 'EUR'])
            st.metric("Number of Currencies", num_curr)
        with col3:
            base_var, _, _, _ = analyzer.calculate_variance_covariance_var(
                exposures, confidence_level, time_horizon, use_historical_correlations
            )
            st.metric(f"{confidence_level:.0%} VaR ({time_horizon} years)", f"€{base_var:,.0f}")
        with col4:
            st.metric("Base Currency", "EUR")
        
        # Exposure by currency (excluding EUR)
        non_eur_exposures = {k: v for k, v in exposures.items() if k != 'EUR'}
        if non_eur_exposures:
            fig_exposure = px.pie(
                values=list(non_eur_exposures.values()),
                names=list(non_eur_exposures.keys()),
                title="Portfolio Exposure by Currency"
            )
            st.plotly_chart(fig_exposure, use_container_width=True)
            
            exposure_df = pd.DataFrame({
                'Currency': list(non_eur_exposures.keys()),
                'Exposure (€)': list(non_eur_exposures.values()),
                'Percentage': [f"{(exp/total_exposure)*100:.1f}%" for exp in non_eur_exposures.values()],
                'Volatility': [f"{analyzer.currency_volatilities[curr]*100:.1f}%" for curr in non_eur_exposures.keys()]
            })
            st.dataframe(exposure_df, use_container_width=True)
    
    with tab2:
        st.header("Variance-Covariance VaR Analysis")
        
        portfolio_var, component_var, marginal_var, contribution_pct = analyzer.calculate_variance_covariance_var(
            exposures, confidence_level, time_horizon, use_historical_correlations
        )
        
        individual_var = analyzer.calculate_individual_var(exposures, confidence_level, time_horizon)
        sum_individual_var = sum(individual_var.values())
        diversification_benefit = sum_individual_var - portfolio_var
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio VaR", f"€{portfolio_var:,.0f}")
        with col2:
            st.metric("Sum Individual VaR", f"€{sum_individual_var:,.0f}")
        with col3:
            st.metric("Diversification Benefit", f"€{diversification_benefit:,.0f}")
        with col4:
            diversification_percent = (diversification_benefit / sum_individual_var * 100) if sum_individual_var > 0 else 0
            st.metric("Diversification %", f"{diversification_percent:.1f}%")
        
        contribution_data = []
        for currency in component_var.keys():
            contribution_data.append({
                'Currency': currency,
                'Exposure': exposures[currency],
                'Component VaR': component_var[currency],
                'Contribution %': contribution_pct[currency],
                'Marginal VaR': marginal_var[currency],
                'Individual VaR': individual_var.get(currency, 0)
            })
        
        contribution_df = pd.DataFrame(contribution_data)
        contribution_df = contribution_df.sort_values('Contribution %', ascending=False)
        
        fig_contribution = px.bar(
            contribution_df,
            x='Currency',
            y='Contribution %',
            title="Contribution to Portfolio Risk (%) - Variance-Covariance Method",
            color='Contribution %',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_contribution, use_container_width=True)
        
        display_df = contribution_df.copy()
        display_df['Exposure'] = display_df['Exposure'].apply(lambda x: f"€{x:,.0f}")
        display_df['Component VaR'] = display_df['Component VaR'].apply(lambda x: f"€{x:,.0f}")
        display_df['Individual VaR'] = display_df['Individual VaR'].apply(lambda x: f"€{x:,.0f}")
        display_df['Marginal VaR'] = display_df['Marginal VaR'].apply(lambda x: f"€{x:,.4f}")
        display_df['Contribution %'] = display_df['Contribution %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df[['Currency', 'Exposure', 'Component VaR', 'Individual VaR', 'Contribution %', 'Marginal VaR']], 
                    use_container_width=True)
    
    with tab3:
        st.header("Monte Carlo Simulation")
        
        with st.spinner("Running Monte Carlo simulation..."):
            portfolio_returns = analyzer.monte_carlo_simulation(
                exposures, num_simulations, time_horizon, use_historical_correlations
            )
        
        simulated_var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"Simulated {confidence_level:.0%} VaR", f"€{simulated_var:,.0f}")
        
        with col2:
            expected_shortfall = -portfolio_returns[portfolio_returns <= -simulated_var].mean()
            st.metric("Expected Shortfall (CVaR)", f"€{expected_shortfall:,.0f}")
            
        with col3:
            parametric_var = portfolio_var
            difference = simulated_var - parametric_var
            st.metric("Difference from Parametric", f"€{difference:,.0f}")
        
        fig_mc = go.Figure()
        
        fig_mc.add_trace(go.Histogram(
            x=portfolio_returns,
            nbinsx=50,
            name="Portfolio Returns",
            opacity=0.7,
            marker_color='lightblue'
        ))
        
        fig_mc.add_vline(
            x=-simulated_var, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"MC VaR: €{simulated_var:,.0f}"
        )
        
        fig_mc.add_vline(
            x=-parametric_var, 
            line_dash="dot", 
            line_color="green",
            annotation_text=f"Parametric VaR: €{parametric_var:,.0f}"
        )
        
        fig_mc.update_layout(
            title="Monte Carlo Simulation - Portfolio Return Distribution",
            xaxis_title="Portfolio Return (€)",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        st.plotly_chart(fig_mc, use_container_width=True)
    
    with tab4:
        st.header("Efficient Frontier - Hedging Strategies")
        
        hedge_ratios, var_reduction, carry_costs = analyzer.calculate_efficient_frontier(
            exposures, time_horizon=time_horizon, use_historical=use_historical_correlations
        )
        
        fig_ef = go.Figure()
        
        fig_ef.add_trace(go.Scatter(
            x=carry_costs,
            y=var_reduction,
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        zero_cost_idx = np.argmin(np.abs(carry_costs))
        fig_ef.add_trace(go.Scatter(
            x=[carry_costs[zero_cost_idx]],
            y=[var_reduction[zero_cost_idx]],
            mode='markers',
            marker=dict(size=12, color='red'),
            name='Zero Cost Hedge'
        ))
        
        current_idx = 0
        fig_ef.add_trace(go.Scatter(
            x=[carry_costs[current_idx]],
            y=[var_reduction[current_idx]],
            mode='markers',
            marker=dict(size=12, color='green'),
            name='Current Portfolio'
        ))
        
        fig_ef.update_layout(
            title="Efficient Frontier - Risk vs Hedging Cost",
            xaxis_title="Hedging Cost/Benefit (€)",
            yaxis_title="Portfolio VaR (€)",
            showlegend=True
        )
        
        st.plotly_chart(fig_ef, use_container_width=True)
    
    with tab5:
        st.header("Currency Correlation Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Correlation Matrix")
            
            # Get the current correlation matrix
            if use_historical_correlations and analyzer.historical_correlations is not None:
                corr_matrix = analyzer.historical_correlations
                matrix_type = "Historical"
            else:
                corr_matrix = analyzer.correlation_matrix
                matrix_type = "Default"
            
            # Filter to only include currencies with exposure
            active_currencies = [curr for curr in exposures.keys() if curr != 'EUR' and exposures[curr] != 0]
            if active_currencies:
                corr_subset = corr_matrix.loc[active_currencies, active_currencies]
                
                # Create heatmap
                fig_corr = px.imshow(
                    corr_subset,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    title=f"{matrix_type} Correlation Matrix (Active Currencies)",
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Display correlation table
                st.subheader("Correlation Values")
                st.dataframe(corr_subset.style.format("{:.3f}").background_gradient(cmap='RdBu', vmin=-1, vmax=1), 
                            use_container_width=True)
            else:
                st.warning("No active currencies to display correlations.")
        
        with col2:
            st.subheader("Correlation Management")
            
            st.info(f"Using: {matrix_type} Correlations")
            
            if analyzer.historical_data is not None:
                st.success(f"Historical data loaded: {len(analyzer.historical_data)} currencies")
                st.metric("Data Period", correlation_period)
            else:
                st.info("Click 'Calculate Historical Correlations' to fetch market data")
            
            # Manual correlation adjustment
            st.subheader("Adjust Correlation")
            if active_currencies and len(active_currencies) >= 2:
                curr1 = st.selectbox("First Currency", active_currencies, key="corr_curr1")
                curr2 = st.selectbox("Second Currency", [c for c in active_currencies if c != curr1], key="corr_curr2")
                
                current_corr = corr_matrix.loc[curr1, curr2]
                new_corr = st.slider(
                    f"Correlation {curr1}-{curr2}",
                    min_value=-1.0,
                    max_value=1.0,
                    value=float(current_corr),
                    step=0.05,
                    key=f"corr_slider_{curr1}_{curr2}"
                )
                
                if st.button("Update Correlation"):
                    analyzer.update_correlation(curr1, curr2, new_corr)
                    st.success(f"Updated {curr1}-{curr2} correlation to {new_corr:.3f}")
                    st.rerun()
            
            # Correlation statistics
            if active_currencies and len(active_currencies) >= 2:
                st.subheader("Correlation Statistics")
                correlations = []
                for i, c1 in enumerate(active_currencies):
                    for j, c2 in enumerate(active_currencies):
                        if i < j:  # Avoid duplicates and self-correlations
                            correlations.append(corr_matrix.loc[c1, c2])
                
                if correlations:
                    avg_corr = np.mean(correlations)
                    max_corr = np.max(correlations)
                    min_corr = np.min(correlations)
                    
                    st.metric("Average Correlation", f"{avg_corr:.3f}")
                    st.metric("Maximum Correlation", f"{max_corr:.3f}")
                    st.metric("Minimum Correlation", f"{min_corr:.3f}")
            
            # Download correlation matrix
            if active_currencies:
                csv = corr_subset.to_csv()
                st.download_button(
                    label="Download Correlation Matrix",
                    data=csv,
                    file_name=f"correlation_matrix_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()