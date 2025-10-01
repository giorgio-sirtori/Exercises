import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Currency Pair Correlation Analyzer",
    page_icon="💹",
    layout="wide"
)

# Title and description
st.title("Currency Pair Correlation Analyzer")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Currency pair selection
st.sidebar.subheader("Select Currency Pairs")

# Common forex pairs with labels
forex_pairs = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X", 
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/CHF": "EURCHF=X",
    "NOK/USD": "NOKUSD=X",
    "SEK/USD": "SEKUSD=X",
    "SEK/EUR": "SEKEUR=X",
    "PLN/EUR": "PLNEUR=X",
    "AUD/EUR": "AUDEUR=X",
    "CZK/EUR": "CZKEUR=X"
}

pair1_label = st.sidebar.selectbox("First Currency Pair:", list(forex_pairs.keys()), index=0)
pair2_label = st.sidebar.selectbox("Second Currency Pair:", list(forex_pairs.keys()), index=1)

pair1 = forex_pairs[pair1_label]
pair2 = forex_pairs[pair2_label]

# Timeframe selection
st.sidebar.subheader("Select Timeframe")
timeframe_options = {
    "1 Month": "1mo",
    "3 Months": "3mo", 
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}

selected_period = st.sidebar.selectbox(
    "Time Period:",
    list(timeframe_options.keys()),
    index=3
)

# Calculate correlation type
st.sidebar.subheader("Correlation Settings")
correlation_type = st.sidebar.radio(
    "Correlation Method:",
    ["Pearson", "Spearman"]
)

# Main content
if st.sidebar.button("Calculate Correlation") or True:
    try:
        # Show loading spinner
        with st.spinner(f"Fetching data for {pair1_label} and {pair2_label}..."):
            # Download data
            data1 = yf.download(pair1, period=timeframe_options[selected_period], progress=False)
            data2 = yf.download(pair2, period=timeframe_options[selected_period], progress=False)
        
        # Check if we have data
        if data1.empty or data2.empty:
            st.error("No data available for one or both currency pairs. Please try different pairs.")
        else:
            # Use closing prices
            prices1 = data1['Close']
            prices2 = data2['Close']
            
            # Create a new DataFrame with aligned dates
            aligned_data = pd.DataFrame(index=prices1.index.union(prices2.index))
            aligned_data[pair1] = prices1
            aligned_data[pair2] = prices2
            
            # Drop any rows with missing values
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) == 0:
                st.error("No overlapping data found between the two currency pairs for the selected timeframe.")
            else:
                # Calculate returns
                returns1 = aligned_data[pair1].pct_change().dropna()
                returns2 = aligned_data[pair2].pct_change().dropna()
                
                # Align returns by index
                returns_df = pd.DataFrame({
                    'returns1': returns1,
                    'returns2': returns2
                }).dropna()
                
                returns1_aligned = returns_df['returns1']
                returns2_aligned = returns_df['returns2']
                
                # Calculate correlation
                if correlation_type == "Pearson":
                    correlation = returns1_aligned.corr(returns2_aligned)
                else:  # Spearman
                    correlation = returns1_aligned.corr(returns2_aligned, method='spearman')
                
                # Display results
                st.subheader("Correlation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label=f"Correlation Coefficient ({correlation_type})",
                        value=f"{correlation:.4f}"
                    )
                
                with col2:
                    # Interpret correlation
                    abs_corr = abs(correlation)
                    if abs_corr >= 0.7:
                        interpretation = "Strong"
                        color = "🔴" if correlation > 0 else "🔵"
                    elif abs_corr >= 0.3:
                        interpretation = "Moderate" 
                        color = "🟠" if correlation > 0 else "🟢"
                    else:
                        interpretation = "Weak"
                        color = "🟡"
                    
                    direction = "positive" if correlation > 0 else "negative"
                    st.metric(
                        label="Correlation Strength",
                        value=f"{interpretation} {direction}"
                    )
                    st.write(color)
                
                with col3:
                    st.metric(
                        label="Data Points",
                        value=f"{len(aligned_data):,}"
                    )
                
                # Create visualizations
                st.subheader("Price and Correlation Analysis")
                
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        f"{pair1_label} Price Chart",
                        f"{pair2_label} Price Chart", 
                        "Daily Returns Scatter Plot",
                        "Rolling Correlation (30-day window)"
                    ),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )
                
                # Plot price charts
                fig.add_trace(
                    go.Scatter(x=aligned_data.index, y=aligned_data[pair1], 
                              name=pair1_label, line=dict(color='blue')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=aligned_data.index, y=aligned_data[pair2], 
                              name=pair2_label, line=dict(color='red')),
                    row=1, col=2
                )
                
                # Scatter plot of returns
                fig.add_trace(
                    go.Scatter(x=returns1_aligned, y=returns2_aligned, 
                              mode='markers', name='Daily Returns',
                              marker=dict(size=4, opacity=0.6, color='green')),
                    row=2, col=1
                )
                
                # Add correlation line to scatter plot
                if len(returns1_aligned) > 1:
                    z = np.polyfit(returns1_aligned, returns2_aligned, 1)
                    p = np.poly1d(z)
                    fig.add_trace(
                        go.Scatter(x=returns1_aligned, y=p(returns1_aligned), 
                                  mode='lines', name='Trend Line',
                                  line=dict(color='red', dash='dash')),
                        row=2, col=1
                    )
                
                # Rolling correlation
                if len(returns_df) >= 30:
                    rolling_corr = returns1_aligned.rolling(window=30).corr(returns2_aligned)
                    fig.add_trace(
                        go.Scatter(x=rolling_corr.index, y=rolling_corr,
                                  name='30-day Rolling Correlation',
                                  line=dict(color='purple', width=2)),
                        row=2, col=2
                    )
                else:
                    # Show constant correlation line if not enough data
                    fig.add_trace(
                        go.Scatter(x=returns_df.index, y=[correlation] * len(returns_df),
                                  name='Overall Correlation',
                                  line=dict(color='purple', width=2, dash='dot')),
                        row=2, col=2
                    )
                
                # Update layout
                fig.update_layout(
                    height=800,
                    showlegend=True,
                    title_text=f"Currency Pair Analysis: {pair1_label} vs {pair2_label}",
                    title_x=0.5
                )
                
                # Update axes labels
                fig.update_xaxes(title_text="Date", row=1, col=1)
                fig.update_xaxes(title_text="Date", row=1, col=2)
                fig.update_xaxes(title_text=f"{pair1_label} Returns", row=2, col=1)
                fig.update_xaxes(title_text="Date", row=2, col=2)
                
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=2)
                fig.update_yaxes(title_text=f"{pair2_label} Returns", row=2, col=1)
                fig.update_yaxes(title_text="Correlation", row=2, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional statistics
                st.subheader("Additional Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(f"Mean Return {pair1_label}", f"{returns1_aligned.mean():.6f}")
                with col2:
                    st.metric(f"Mean Return {pair2_label}", f"{returns2_aligned.mean():.6f}")
                with col3:
                    st.metric(f"Volatility {pair1_label}", f"{returns1_aligned.std():.6f}")
                with col4:
                    st.metric(f"Volatility {pair2_label}", f"{returns2_aligned.std():.6f}")
                
                # Correlation matrix
                st.subheader("Correlation Matrix")
                corr_matrix = returns_df.corr()
                st.dataframe(corr_matrix.style.format("{:.4f}").background_gradient(cmap='coolwarm'))
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        st.info("Please check your internet connection and try again.")

# Footer
st.markdown("---")
st.markdown(
    """
    **Interpretation Guide:**
    - **Strong positive correlation (0.7 to 1.0)**: Pairs tend to move together
    - **Moderate positive correlation (0.3 to 0.7)**: Pairs somewhat move together  
    - **Weak correlation (-0.3 to 0.3)**: Little relationship between pairs
    - **Moderate negative correlation (-0.7 to -0.3)**: Pairs somewhat move oppositely
    - **Strong negative correlation (-1.0 to -0.7)**: Pairs tend to move in opposite directions
    """
)