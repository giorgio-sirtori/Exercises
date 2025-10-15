import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Currency VaR Analysis", layout="wide")
st.title("💱 Currency Value at Risk (VaR) Analysis")

# Sidebar configuration
st.sidebar.header("Configuration")
base_currency = st.sidebar.selectbox("Base Currency", ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"])
confidence_level = st.sidebar.slider("Confidence Level (%)", 90.0, 99.9, 95.0, 0.1)
time_horizon = st.sidebar.slider("Time Horizon (Days)", 1, 250, 10)
lookback_period = st.sidebar.slider("Historical Data Lookback (Days)", 30, 1000, 252)

# Initialize session state for exposures
if 'exposures' not in st.session_state:
    st.session_state.exposures = []

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Setup", "VaR Analysis", "Risk Metrics", "Diagnostics"])

with tab1:
    st.subheader("Add Currency Exposures")
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        currency_input = st.text_input("Enter Currency Code (e.g., EUR, GBP, JPY)", 
                                       value="EUR", key="cur_select").upper()
        currency = currency_input if currency_input else "EUR"
    
    with col2:
        amount = st.number_input("Exposure Amount", min_value=0.0, value=1000000.0, step=100000.0)
    
    with col3:
        position_type = st.selectbox("Position", ["Long", "Short"], key="pos_type")
    
    with col4:
        if st.button("➕ Add"):
            pair = f"{currency}/{base_currency}"
            final_amount = amount if position_type == "Long" else -amount
            st.session_state.exposures.append({
                'currency': currency,
                'pair': pair,
                'amount': final_amount,
                'position_type': position_type,
                'abs_amount': amount
            })
    
    if st.session_state.exposures:
        st.write("---")
        st.subheader("Current Exposures")
        
        exposure_data = []
        for i, exp in enumerate(st.session_state.exposures):
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            with col1:
                st.write(f"**{exp.get('pair', exp)}**")
            with col2:
                st.write(f"{exp.get('amount', 0):,.0f}")
            with col3:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.exposures.pop(i)
                    st.rerun()
            
            exposure_data.append({
                'Currency Pair': exp.get('pair', exp),
                'Exposure Amount': f"{exp.get('amount', 0):,.0f}"
            })
        
        if exposure_data:
            st.dataframe(pd.DataFrame(exposure_data), use_container_width=True, hide_index=True)
            total_exposure = sum(e['amount'] for e in st.session_state.exposures)
            st.metric("Total Exposure", f"{total_exposure:,.0f}")

with tab2:
    if not st.session_state.exposures:
        st.info("Please add at least one currency exposure in the 'Portfolio Setup' tab")
    else:
        st.subheader("VaR Computation")
        
        # Fetch historical data
        @st.cache_data
        def fetch_currency_data(pairs, lookback):
            data = {}
            for pair in pairs:
                try:
                    pair_str = pair.replace("/", "")
                    ticker = yf.Ticker(pair_str + "=X")
                    hist = ticker.history(period=f"{lookback}d")
                    if len(hist) > 0:
                        data[pair] = hist['Close'].pct_change().dropna()
                    else:
                        st.warning(f"Could not fetch data for {pair}")
                except:
                    st.warning(f"Error fetching data for {pair}")
            return data
        
        pairs = [e['pair'] for e in st.session_state.exposures]
        returns_data = fetch_currency_data(pairs, lookback_period)
        
        if not returns_data:
            st.error("Could not fetch market data. Please check your internet connection.")
        else:
            # Compute correlations
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # VaR Calculation Functions
            def var_parametric(returns, confidence, horizon):
                """Parametric (Normal Distribution) VaR"""
                returns_clean = returns.dropna()
                if len(returns_clean) == 0:
                    return np.nan
                mu = returns_clean.mean()
                sigma = returns_clean.std()
                if sigma == 0:
                    return np.nan
                z_score = stats.norm.ppf(1 - confidence / 100)
                daily_var = mu - sigma * z_score
                return daily_var * np.sqrt(horizon)
            
            def var_historical(returns, confidence, horizon):
                """Historical Simulation VaR"""
                returns_clean = returns.dropna()
                if len(returns_clean) == 0:
                    return np.nan
                confidence_level_decimal = confidence / 100
                var_daily = returns_clean.quantile(1 - confidence_level_decimal)
                return var_daily * np.sqrt(horizon)
            
            def var_monte_carlo(returns, confidence, horizon, simulations=10000):
                """Monte Carlo VaR"""
                returns_clean = returns.dropna()
                if len(returns_clean) == 0:
                    return np.nan, None
                mu = returns_clean.mean()
                sigma = returns_clean.std()
                if sigma == 0:
                    return np.nan, None
                terminal_returns = []
                
                for _ in range(simulations):
                    daily_returns = np.random.normal(mu, sigma, horizon)
                    cumulative_return = np.sum(daily_returns)
                    terminal_returns.append(cumulative_return)
                
                confidence_level_decimal = confidence / 100
                var_mc = np.percentile(terminal_returns, (1 - confidence_level_decimal) * 100)
                return var_mc, terminal_returns
            
            def var_cornish_fisher(returns, confidence, horizon):
                """Cornish-Fisher VaR (accounts for skewness and kurtosis)"""
                returns_clean = returns.dropna()
                if len(returns_clean) == 0:
                    return np.nan
                mu = returns_clean.mean()
                sigma = returns_clean.std()
                if sigma == 0:
                    return np.nan
                skewness = returns_clean.skew()
                kurtosis = returns_clean.kurtosis()
                
                z = stats.norm.ppf(1 - confidence / 100)
                z_cf = z + (z**2 - 1) * skewness / 6 + (z**3 - 3*z) * kurtosis / 24 - (2*z**3 - 5*z) * skewness**2 / 36
                
                daily_var = mu - sigma * z_cf
                return daily_var * np.sqrt(horizon)
            
            # Portfolio calculations
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Individual Exposures VaR**")
                var_results = []
                
                for exp in st.session_state.exposures:
                    if exp['pair'] in returns_data:
                        returns = returns_data[exp['pair']]
                        amount = exp['amount']
                        
                        var_param = var_parametric(returns, confidence_level, time_horizon)
                        var_hist = var_historical(returns, confidence_level, time_horizon)
                        var_mc, mc_sim = var_monte_carlo(returns, confidence_level, time_horizon)
                        var_cf = var_cornish_fisher(returns, confidence_level, time_horizon)
                        
                        var_results.append({
                            'pair': exp['pair'],
                            'amount': amount,
                            'parametric': var_param,
                            'historical': var_hist,
                            'monte_carlo': var_mc,
                            'cornish_fisher': var_cf,
                            'mc_simulation': mc_sim
                        })
                
                var_df = pd.DataFrame([{
                    'Pair': r['pair'],
                    'Exposure': f"{r['amount']:,.0f}",
                    'Param (%)': f"{r['parametric']*100:.2f}%",
                    'Hist (%)': f"{r['historical']*100:.2f}%",
                    'MC (%)': f"{r['monte_carlo']*100:.2f}%",
                    'CF (%)': f"{r['cornish_fisher']*100:.2f}%"
                } for r in var_results])
                
                st.dataframe(var_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.write("**VaR in Currency Units (Loss Amounts)**")
                var_loss_df = pd.DataFrame([{
                    'Pair': r['pair'],
                    'Parametric': f"{r['parametric']*r['amount']:,.0f}",
                    'Historical': f"{r['historical']*r['amount']:,.0f}",
                    'Monte Carlo': f"{r['monte_carlo']*r['amount']:,.0f}",
                    'C-Fisher': f"{r['cornish_fisher']*r['amount']:,.0f}"
                } for r in var_results])
                
                st.dataframe(var_loss_df, use_container_width=True, hide_index=True)
            
            # VaR Comparison Chart
            st.write("---")
            st.subheader("VaR Methodology Comparison")
            
            var_comparison_data = []
            for r in var_results:
                var_comparison_data.append({
                    'Currency': r['pair'],
                    'Parametric': r['parametric'] * 100,
                    'Historical': r['historical'] * 100,
                    'Monte Carlo': r['monte_carlo'] * 100,
                    'Cornish-Fisher': r['cornish_fisher'] * 100
                })
            
            var_comp_df = pd.DataFrame(var_comparison_data)
            
            fig_var_comp = go.Figure()
            for method in ['Parametric', 'Historical', 'Monte Carlo', 'Cornish-Fisher']:
                fig_var_comp.add_trace(go.Bar(
                    x=var_comp_df['Currency'],
                    y=var_comp_df[method],
                    name=method
                ))
            
            fig_var_comp.update_layout(
                barmode='group',
                title=f'VaR Comparison by Methodology ({confidence_level}% CL, {time_horizon}D)',
                xaxis_title='Currency Pair',
                yaxis_title='VaR (%)',
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_var_comp, use_container_width=True)
            
            # Monte Carlo Simulation Charts
            st.write("---")
            st.subheader("Monte Carlo Simulation Analysis")
            
            mc_cols = st.columns(len(var_results))
            for idx, (col, result) in enumerate(zip(mc_cols, var_results)):
                if result['mc_simulation'] is not None:
                    with col:
                        mc_returns = result['mc_simulation']
                        var_mc = result['monte_carlo']
                        confidence_level_decimal = confidence_level / 100
                        
                        fig_mc = go.Figure()
                        
                        fig_mc.add_trace(go.Histogram(
                            x=mc_returns,
                            name='Simulated Returns',
                            nbinsx=50,
                            marker_color='rgba(100, 150, 200, 0.7)',
                            showlegend=True
                        ))
                        
                        fig_mc.add_vline(
                            x=var_mc,
                            line_dash='dash',
                            line_color='red',
                            annotation_text=f'VaR ({confidence_level}%): {var_mc*100:.2f}%',
                            annotation_position='top right'
                        )
                        
                        fig_mc.add_vline(
                            x=0,
                            line_dash='dash',
                            line_color='green',
                            annotation_text='No Loss/Gain',
                            annotation_position='top left'
                        )
                        
                        fig_mc.update_layout(
                            title=f'MC Simulation: {result["pair"]}',
                            xaxis_title='Return (%)',
                            yaxis_title='Frequency',
                            template='plotly_white',
                            height=400,
                            showlegend=False,
                            hovermode='x'
                        )
                        
                        st.plotly_chart(fig_mc, use_container_width=True)
            
            # Portfolio VaR
            if len(st.session_state.exposures) > 1:
                st.write("---")
                st.subheader("Portfolio Analysis")
                
                # Weighted returns
                weights = np.array([e['amount'] for e in st.session_state.exposures])
                weights = weights / weights.sum()
                
                portfolio_returns = returns_df.iloc[:, :len(st.session_state.exposures)].dot(weights)
                
                port_var_param = var_parametric(portfolio_returns, confidence_level, time_horizon)
                port_var_hist = var_historical(portfolio_returns, confidence_level, time_horizon)
                port_var_mc, port_mc_sim = var_monte_carlo(portfolio_returns, confidence_level, time_horizon)
                port_var_cf = var_cornish_fisher(portfolio_returns, confidence_level, time_horizon)
                
                total_exposure = sum(e['amount'] for e in st.session_state.exposures)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Parametric VaR", f"{port_var_param*100:.2f}%", f"{port_var_param*total_exposure:,.0f}")
                with col2:
                    st.metric("Historical VaR", f"{port_var_hist*100:.2f}%", f"{port_var_hist*total_exposure:,.0f}")
                with col3:
                    st.metric("Monte Carlo VaR", f"{port_var_mc*100:.2f}%", f"{port_var_mc*total_exposure:,.0f}")
                with col4:
                    st.metric("Cornish-Fisher VaR", f"{port_var_cf*100:.2f}%", f"{port_var_cf*total_exposure:,.0f}")
                
                # Diversification benefit
                individual_var_param = sum(var_parametric(returns_data[e['pair']], confidence_level, time_horizon) * e['amount'] 
                                          for e in st.session_state.exposures)
                div_benefit = (individual_var_param - port_var_param * total_exposure) / individual_var_param * 100 if individual_var_param != 0 else 0
                
                st.info(f"**Diversification Benefit (Parametric):** {div_benefit:.2f}%\n\n"
                       f"This represents the risk reduction from holding multiple currencies "
                       f"due to imperfect correlations.")
                
                # Portfolio composition and MC simulation
                st.write("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Portfolio weights pie chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=[e['pair'] for e in st.session_state.exposures],
                        values=[e['amount'] for e in st.session_state.exposures],
                        hole=0,
                        marker=dict(line=dict(color='white', width=2))
                    )])
                    
                    fig_pie.update_layout(
                        title='Portfolio Composition',
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Portfolio MC simulation
                    if port_mc_sim is not None:
                        fig_port_mc = go.Figure()
                        
                        fig_port_mc.add_trace(go.Histogram(
                            x=port_mc_sim,
                            name='Simulated Returns',
                            nbinsx=50,
                            marker_color='rgba(100, 150, 200, 0.7)',
                            showlegend=True
                        ))
                        
                        fig_port_mc.add_vline(
                            x=port_var_mc,
                            line_dash='dash',
                            line_color='red',
                            annotation_text=f'Portfolio VaR: {port_var_mc*100:.2f}%',
                            annotation_position='top right'
                        )
                        
                        fig_port_mc.add_vline(
                            x=0,
                            line_dash='dash',
                            line_color='green',
                            annotation_text='Break-even'
                        )
                        
                        fig_port_mc.update_layout(
                            title='Portfolio MC Simulation',
                            xaxis_title='Return (%)',
                            yaxis_title='Frequency',
                            template='plotly_white',
                            height=400,
                            showlegend=False,
                            hovermode='x'
                        )
                        st.plotly_chart(fig_port_mc, use_container_width=True)

with tab3:
    if not st.session_state.exposures:
        st.info("Please add at least one currency exposure")
    else:
        st.subheader("Risk Metrics & Statistics")
        
        pairs = [e['pair'] for e in st.session_state.exposures]
        returns_data = fetch_currency_data(pairs, lookback_period)
        
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Correlation Matrix**")
                fig_corr = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=np.round(correlation_matrix.values, 3),
                    texttemplate='%{text:.3f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                
                fig_corr.update_layout(
                    title='Currency Pair Correlations',
                    height=450,
                    template='plotly_white'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                st.write("**Return Statistics**")
                stats_data = []
                for col in returns_df.columns:
                    stats_data.append({
                        'Currency': col,
                        'Mean Return': f"{returns_df[col].mean()*100:.3f}%",
                        'Std Dev': f"{returns_df[col].std()*100:.3f}%",
                        'Skewness': f"{returns_df[col].skew():.3f}",
                        'Kurtosis': f"{returns_df[col].kurtosis():.3f}"
                    })
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
            
            st.write("---")
            st.write("**Return Distributions**")
            
            figs = []
            for col in returns_df.columns:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=returns_df[col],
                    nbinsx=50,
                    marker_color='rgba(0, 100, 200, 0.7)',
                    name='Returns'
                ))
                
                fig_dist.add_vline(
                    x=returns_df[col].mean(),
                    line_dash='dash',
                    line_color='green',
                    annotation_text=f'Mean: {returns_df[col].mean()*100:.3f}%',
                    annotation_position='top right'
                )
                
                fig_dist.update_layout(
                    title=f'{col} Return Distribution',
                    xaxis_title='Daily Return (%)',
                    yaxis_title='Frequency',
                    template='plotly_white',
                    height=350,
                    showlegend=False
                )
                figs.append(fig_dist)
            
            cols_dist = st.columns(len(figs))
            for col, fig in zip(cols_dist, figs):
                with col:
                    st.plotly_chart(fig, use_container_width=True)

with tab4:
    if not st.session_state.exposures:
        st.info("Please add at least one currency exposure")
    else:
        st.subheader("Model Diagnostics")
        
        pairs = [e['pair'] for e in st.session_state.exposures]
        returns_data = fetch_currency_data(pairs, lookback_period)
        
        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            
            # Normality test
            st.write("**Normality Tests (Jarque-Bera)**")
            normality_results = []
            for col in returns_df.columns:
                statistic, pvalue = stats.jarque_bera(returns_df[col])
                normality_results.append({
                    'Currency': col,
                    'Test Statistic': f"{statistic:.4f}",
                    'P-Value': f"{pvalue:.6f}",
                    'Normal?': "✓ Yes" if pvalue > 0.05 else "✗ No"
                })
            st.dataframe(pd.DataFrame(normality_results), use_container_width=True, hide_index=True)
            
            st.info("P-Value > 0.05 suggests returns are normally distributed. "
                   "Non-normal distributions may require Cornish-Fisher or Monte Carlo methods.")
            
            # Q-Q plots
            st.write("---")
            st.write("**Q-Q Plots (Normal Distribution Test)**")
            qq_figs = []
            for col in returns_df.columns:
                qq_quantiles = stats.probplot(returns_df[col], dist="norm")
                
                fig_qq = go.Figure()
                fig_qq.add_trace(go.Scatter(
                    x=qq_quantiles[0][0],
                    y=qq_quantiles[0][1],
                    mode='markers',
                    marker=dict(size=4, color='blue'),
                    name='Returns'
                ))
                
                # Add reference line
                min_val = min(qq_quantiles[0][0])
                max_val = max(qq_quantiles[0][0])
                fig_qq.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val * qq_quantiles[1][0] + qq_quantiles[1][1],
                       max_val * qq_quantiles[1][0] + qq_quantiles[1][1]],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Normal Reference'
                ))
                
                fig_qq.update_layout(
                    title=f'{col} Q-Q Plot',
                    xaxis_title='Theoretical Quantiles',
                    yaxis_title='Sample Quantiles',
                    template='plotly_white',
                    height=350,
                    hovermode='closest'
                )
                qq_figs.append(fig_qq)
            
            cols_qq = st.columns(len(qq_figs))
            for col, fig in zip(cols_qq, qq_figs):
                with col:
                    st.plotly_chart(fig, use_container_width=True)
            
            # ACF for independence
            st.write("---")
            st.write("**Autocorrelation Analysis**")
            from pandas.plotting import autocorrelation_plot
            
            acf_figs = []
            for col in returns_df.columns:
                acf_vals = [returns_df[col].autocorr(lag=i) for i in range(21)]
                
                fig_acf = go.Figure()
                fig_acf.add_trace(go.Bar(
                    x=list(range(21)),
                    y=acf_vals,
                    marker_color='lightblue',
                    marker_line_color='darkblue'
                ))
                
                # Add significance bounds
                n = len(returns_df[col])
                conf_bound = 1.96 / np.sqrt(n)
                fig_acf.add_hline(y=conf_bound, line_dash="dash", line_color="red", annotation_text="95% CI")
                fig_acf.add_hline(y=-conf_bound, line_dash="dash", line_color="red")
                
                fig_acf.update_layout(
                    title=f'{col} Autocorrelation',
                    xaxis_title='Lag',
                    yaxis_title='ACF',
                    template='plotly_white',
                    height=350,
                    showlegend=False
                )
                acf_figs.append(fig_acf)
            
            cols_acf = st.columns(len(acf_figs))
            for col, fig in zip(cols_acf, acf_figs):
                with col:
                    st.plotly_chart(fig, use_container_width=True)