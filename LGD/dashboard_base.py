import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import statsmodels.api as sm
import os
from sklearn.feature_selection import f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data():
    csv_file = "deposit.csv"
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    else:
        st.error(f"File {csv_file} not found.")
        return None

def preprocess_data(df, x_vars, y_var):
    df_filtered = df[x_vars + [y_var]].dropna()
    categorical_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        df_filtered[col] = df_filtered[col].astype('category')
        df_filtered[col] = df_filtered[col].cat.codes  # Encode categorical variables numerically
    
    return df_filtered

def calculate_vif(df, features):
    X = sm.add_constant(df[features])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(X.values, i + 1) for i in range(len(features))]  # Skip constant
    return vif_data

def regression_analysis(df):
    st.title("Regression Analysis")
    
    if df is not None:
        st.sidebar.header("Data Filters")
        filter_columns = st.sidebar.multiselect("Choose columns to filter by", df.columns)
        for column in filter_columns:
            unique_values = df[column].unique()
            selected_values = st.sidebar.multiselect(f"Select values from {column}", unique_values, default=unique_values)
            df = df[df[column].isin(selected_values)]
        
        st.sidebar.header("Regression Settings")
        x_vars = st.sidebar.multiselect("Select Independent Variables (X)", df.columns)
        y_var = st.sidebar.selectbox("Select Dependent Variable (Y)", df.columns)
        
        if x_vars and y_var:
            df_filtered = preprocess_data(df, x_vars, y_var)
            
            if df_filtered.empty:
                st.error("No valid data available for regression.")
                return
            
            X = sm.add_constant(df_filtered[x_vars])  # Add constant for intercept
            y = df_filtered[y_var]
            
            model = sm.OLS(y, X).fit()
            
            st.write("### Regression Summary")
            st.text(model.summary())
            
            # Feature Selection (F-test)
            st.write("### Feature Selection (F-Test)")
            f_values, p_values = f_regression(df_filtered[x_vars], y)
            feature_scores = pd.DataFrame({"Feature": x_vars, "F-Value": f_values, "P-Value": p_values})
            st.write(feature_scores.sort_values(by="P-Value"))
            
            # Variance Inflation Factor (VIF) Calculation
            st.write("### Variance Inflation Factor (VIF)")
            vif_data = calculate_vif(df_filtered, x_vars)
            st.write(vif_data)
            
            # Correlation Matrix
            st.write("### Correlation Matrix")
            corr_matrix = df_filtered.corr()
            st.write(corr_matrix)
            
            # Visualization
            for x_var in x_vars:
                fig = px.scatter(df_filtered, x=x_var, y=y_var, trendline="ols", 
                                 title=f"Regression: {y_var} vs {x_var}")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data not available.")

# Streamlit Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Loss Given Default Dashboard", "Regression Analysis","Weighted Loss Given Default Dashboard"])

df = load_data()

df['DEPOSIT_PERCENT'] = (df['MIN_TRANSACTION'] / df['GTV_AMOUNT_MCX']  ) * 100

df['LGD_CALC'] = (df['TOTAL_COST_Unpaid_Vol']  + df['TOTAL_COST_Unpaid_Inv']  / df['GTV_AMOUNT_MCX']  ) * 100

# Define deposit percentage clusters
bins = [0, 10, 20, 30, 40, float("inf")]
labels = ["0-10%", "11-20%", "21-30%", "31-40%", "40%+"]

df['DEPOSIT_CLUSTER'] = pd.cut(df['DEPOSIT_PERCENT'], bins=bins, labels=labels, right=True)

if page == "Loss Given Default Dashboard":
    st.title("Loss Given Default Dashboard")
    # (Your existing dashboard code will be here)
    # Check if required columns exist in the CSV file
    required_columns = ['GTV_AMOUNT_MCX', 'TOTAL_COST_Unpaid_Vol', 'TOTAL_COST_Unpaid_Inv']

    if not all(col in df.columns for col in required_columns):
        st.error("CSV file is missing required columns.")
    else:
        # Sidebar Filters
        st.sidebar.header("Filter Data")

        # Add static filters for FLOWN and BK_CATEGORY
        flown_values = df['FLOWN'].unique() if 'FLOWN' in df.columns else []
        selected_flown = st.sidebar.multiselect('Select FLOWN values', flown_values, default=flown_values)

        bk_category_values = df['BK_CATEGORY'].unique() if 'BK_CATEGORY' in df.columns else []
        selected_bk_category = st.sidebar.multiselect('Select BK_CATEGORY values', bk_category_values, default=bk_category_values)

        # Filter the dataframe based on static filters
        df_filtered = df.copy()
        if 'FLOWN' in df.columns:
            df_filtered = df_filtered[df_filtered['FLOWN'].isin(selected_flown)]
        if 'BK_CATEGORY' in df.columns:
            df_filtered = df_filtered[df_filtered['BK_CATEGORY'].isin(selected_bk_category)]

        # Add filters for multiple columns
        filter_columns = st.sidebar.multiselect('Choose additional columns to filter by', df.columns)
        for column in filter_columns:
            unique_values = df_filtered[column].unique()
            selected_values = st.sidebar.multiselect(f'Select values from {column}', unique_values, default=unique_values)
            df_filtered = df_filtered[df_filtered[column].isin(selected_values)]

        # Streamlit Dashboard
        st.title('Loss Given Default Dashboard')

        # Display total values for unfiltered dataframe
        st.subheader("Totals for Unfiltered Dataframe:")
        unfiltered_totals = {
            "Total Unpaid Vol": df['TOTAL_COST_Unpaid_Vol'].sum(),
            "Total Unpaid Inv": df['TOTAL_COST_Unpaid_Inv'].sum(),
            "Total GTV Amount": df['GTV_AMOUNT_MCX'].sum()
        }
        st.write(unfiltered_totals)

        # Display total values for filtered dataframe
        st.subheader("Totals for Filtered Dataframe:")
        filtered_totals = {
            "Total Unpaid Vol": df_filtered['TOTAL_COST_Unpaid_Vol'].sum(),
            "Total Unpaid Inv": df_filtered['TOTAL_COST_Unpaid_Inv'].sum(),
            "Total GTV Amount": df_filtered['GTV_AMOUNT_MCX'].sum()
        }
        st.write(filtered_totals)

        # Choose X and Y axis dimensions
        x_axis = st.selectbox('Choose X-axis dimension', df_filtered.columns)
        y_axis = st.selectbox('Choose Y-axis dimension', df_filtered.columns)

        # Aggregating the values for the chosen dimensions
        agg_df = df_filtered.groupby([x_axis, y_axis]).agg(
            Unpaid_Vol=('TOTAL_COST_Unpaid_Vol', 'sum'),
            Unpaid_Inv=('TOTAL_COST_Unpaid_Inv', 'sum'),
            GTV_Amount=('GTV_AMOUNT_MCX', 'sum'),
            booking_count=('booking_count','sum'),
            min_transaction_sum=('MIN_TRANSACTION','sum'),
            min_transaction_avg=('MIN_TRANSACTION','mean'),
            PMC3=('PMC3','sum')
        ).reset_index()

        # Calculate LGD dynamically for the aggregated values
        agg_df['LGD'] = (agg_df['Unpaid_Vol'] + agg_df['Unpaid_Inv']) / agg_df['GTV_Amount'] * 100
        agg_df['ABV'] = (agg_df['GTV_Amount']) / agg_df['booking_count']  
        agg_df['PMC3_UNIT'] = (agg_df['PMC3']) / agg_df['booking_count'] 
        


        # Create a pivot table for the selected dimensions with LGD as values
        pivot_df = agg_df.pivot_table(values='LGD', index=x_axis, columns=y_axis, aggfunc=np.mean)

        # Define color scale
        def color_scale(val):
            if val < -0.29:
                return 'background-color: red'
            else:
                return 'background-color: green'

        # Apply color scale to the pivot table
        styled_pivot_df = pivot_df.style.applymap(color_scale).format({'LGD': '{:.2f}%'});

        # Show the styled table
        st.write("Loss Given Default Table (in Percentage):")
        st.dataframe(styled_pivot_df)

        # Plotting the heatmap using Plotly
        st.write("Loss Given Default Heatmap (Interactive):")
        fig = px.imshow(
            pivot_df,
            labels=dict(x=y_axis, y=x_axis, color='LGD (%)'),
            color_continuous_scale='RdYlBu',  # Inverted color scale
            zmin=pivot_df.min().min(),
            zmax=pivot_df.max().max()
        )
        fig.update_layout(
            title=f'Loss Given Default Heatmap ({x_axis} vs {y_axis})',
            xaxis_title=y_axis,
            yaxis_title=x_axis,
            coloraxis_colorbar=dict(title='LGD (%)')
        )

        st.plotly_chart(fig, use_container_width=True)
        # Display the aggregated dataframe for inspection
        st.write("Aggregated Dataframe:")
        st.dataframe(agg_df)

        ###########################################################################
        # Added Projection Section: Estimating the impact of deposit reduction on conversion rates, LGD, volume, and PMC3
        ###########################################################################

        st.subheader("Projected Impact of Deposit Reduction")
        st.write("Adjust the parameters below to see the projected changes when lowering deposit levels:")

        # Sidebar inputs for projection parameters
        st.sidebar.header("Projection Settings")
        base_conversion = st.sidebar.number_input("Baseline Conversion Rate", value=0.30, min_value=0.0, max_value=1.0, step=0.01,
                                                  help="Enter the current conversion rate (as a decimal, e.g., 0.30 for 30%).")
        default_rate = st.sidebar.number_input("Default Rate", value=0.05, min_value=0.0, max_value=1.0, step=0.01,
                                               help="Enter the default rate (as a decimal, e.g., 0.05 for 5%).")
        deposit_reduction = st.sidebar.slider("Deposit Reduction (%)", 0, 50, 10, step=1,
                                              help="Select the percentage reduction in the deposit.")
        conversion_improvement_per_10 = st.sidebar.number_input("Conversion Improvement per 10% Deposit Reduction (%)", value=5, min_value=0,
                                                                help="Expected percentage increase in conversion rate per 10% deposit reduction.")
        lgd_increase_per_10 = st.sidebar.number_input("LGD Increase per 10% Deposit Reduction (%)", value=2, min_value=0,
                                                      help="Expected percentage increase in LGD per 10% deposit reduction.")
        volume_improvement_per_10 = st.sidebar.number_input("Volume Increase per 10% Deposit Reduction (%)", value=3, min_value=0,
                                                            help="Expected percentage increase in volume (e.g., booking count) per 10% deposit reduction.")
        pmc3_improvement_per_10 = st.sidebar.number_input("PMC3 Increase per 10% Deposit Reduction (%)", value=1, min_value=0,
                                                          help="Expected percentage increase in PMC3 per 10% deposit reduction.")

        # Calculate new conversion rate based on deposit reduction
        new_conversion = base_conversion * (1 + (deposit_reduction / 10) * (conversion_improvement_per_10 / 100))

        # Compute the average LGD from the aggregated data (converted to a decimal)
        avg_LGD = agg_df['LGD'].mean() / 100  # LGD is in percentage in agg_df
        new_avg_LGD = avg_LGD * (1 + (deposit_reduction / 10) * (lgd_increase_per_10 / 100))

        # Assume exposure is the total GTV Amount from the filtered totals
        exposure = filtered_totals["Total GTV Amount"]

        # Calculate expected loss using the formula:
        # Expected Loss = Conversion Rate x Default Rate x LGD x Exposure
        expected_loss_baseline = base_conversion * default_rate * avg_LGD * exposure
        projected_expected_loss = new_conversion * default_rate * new_avg_LGD * exposure

        st.write("**Baseline Parameters:**")
        st.write(f"Baseline Conversion Rate: {base_conversion:.2%}")
        st.write(f"Default Rate: {default_rate:.2%}")
        st.write(f"Average LGD: {avg_LGD:.2%}")
        st.write(f"Exposure (Total GTV Amount): {exposure:,.2f}")

        st.write("---")
        st.write("**Expected Loss Projection:**")
        st.write(f"New Conversion Rate (after {deposit_reduction}% deposit reduction): {new_conversion:.2%}")
        st.write(f"New Average LGD: {new_avg_LGD:.2%}")
        st.write(f"Baseline Expected Loss: {expected_loss_baseline:,.2f}")
        st.write(f"Projected Expected Loss: {projected_expected_loss:,.2f}")

        loss_increase = projected_expected_loss - expected_loss_baseline
        loss_increase_pct = (loss_increase / expected_loss_baseline * 100) if expected_loss_baseline != 0 else np.nan
        st.write(f"Increase in Expected Loss: {loss_increase:,.2f} ({loss_increase_pct:.2f}%)")

        ###########################################################################
        # Additional Projections for Volumes and PMC3
        ###########################################################################

        # Projection for volume increase based on booking_count
        if 'booking_count' in agg_df.columns:
            baseline_volume = agg_df['booking_count'].sum()
        else:
            baseline_volume = 0
        new_volume = baseline_volume * (1 + (deposit_reduction / 10) * (volume_improvement_per_10 / 100))

        # Projection for PMC3 increase
        if 'PMC3' in agg_df.columns:
            baseline_PMC3 = agg_df['PMC3'].sum()
        else:
            baseline_PMC3 = 0
        new_PMC3 = baseline_PMC3 * (1 + (deposit_reduction / 10) * (pmc3_improvement_per_10 / 100))

        st.write("---")
        st.write("**Volume Projections:**")
        st.write(f"Baseline Volume (booking_count sum): {baseline_volume:,.0f}")
        st.write(f"Projected Volume (after {deposit_reduction}% deposit reduction): {new_volume:,.0f}")
        vol_increase = new_volume - baseline_volume
        vol_increase_pct = (vol_increase / baseline_volume * 100) if baseline_volume != 0 else np.nan
        st.write(f"Increase in Volume: {vol_increase:,.0f} ({vol_increase_pct:.2f}%)")

        st.write("---")
        st.write("**PMC3 Projections:**")
        st.write(f"Baseline PMC3: {baseline_PMC3:,.2f}")
        st.write(f"Projected PMC3 (after {deposit_reduction}% deposit reduction): {new_PMC3:,.2f}")
        pmc3_increase = new_PMC3 - baseline_PMC3
        pmc3_increase_pct = (pmc3_increase / baseline_PMC3 * 100) if baseline_PMC3 != 0 else np.nan
        st.write(f"Increase in PMC3: {pmc3_increase:,.2f} ({pmc3_increase_pct:.2f}%)")
elif page == "Regression Analysis":
    regression_analysis(df)
elif page == "Weighted Loss Given Default Dashboard":
    st.title("Weighted Loss Given Default Dashboard")
    # (Your existing dashboard code will be here)
    # Check if required columns exist in the CSV file
    required_columns = ['GTV_AMOUNT_MCX', 'TOTAL_COST_Unpaid_Vol', 'TOTAL_COST_Unpaid_Inv']

    if not all(col in df.columns for col in required_columns):
        st.error("CSV file is missing required columns.")
    else:
        # Sidebar Filters
        st.sidebar.header("Filter Data")

        # Add static filters for FLOWN and BK_CATEGORY
        flown_values = df['FLOWN'].unique() if 'FLOWN' in df.columns else []
        selected_flown = st.sidebar.multiselect('Select FLOWN values', flown_values, default=flown_values)

        bk_category_values = df['BK_CATEGORY'].unique() if 'BK_CATEGORY' in df.columns else []
        selected_bk_category = st.sidebar.multiselect('Select BK_CATEGORY values', bk_category_values, default=bk_category_values)

        # Filter the dataframe based on static filters
        df_filtered = df.copy()
        if 'FLOWN' in df.columns:
            df_filtered = df_filtered[df_filtered['FLOWN'].isin(selected_flown)]
        if 'BK_CATEGORY' in df.columns:
            df_filtered = df_filtered[df_filtered['BK_CATEGORY'].isin(selected_bk_category)]

        # Add filters for multiple columns
        filter_columns = st.sidebar.multiselect('Choose additional columns to filter by', df.columns)
        for column in filter_columns:
            unique_values = df_filtered[column].unique()
            selected_values = st.sidebar.multiselect(f'Select values from {column}', unique_values, default=unique_values)
            df_filtered = df_filtered[df_filtered[column].isin(selected_values)]

        # Streamlit Dashboard
        st.title('Loss Given Default Dashboard')

        # Display total values for unfiltered dataframe
        st.subheader("Totals for Unfiltered Dataframe:")
        unfiltered_totals = {
            "Total Unpaid Vol": df['TOTAL_COST_Unpaid_Vol'].sum(),
            "Total Unpaid Inv": df['TOTAL_COST_Unpaid_Inv'].sum(),
            "Total GTV Amount": df['GTV_AMOUNT_MCX'].sum()
        }
        st.write(unfiltered_totals)

        # Display total values for filtered dataframe
        st.subheader("Totals for Filtered Dataframe:")
        filtered_totals = {
            "Total Unpaid Vol": df_filtered['TOTAL_COST_Unpaid_Vol'].sum(),
            "Total Unpaid Inv": df_filtered['TOTAL_COST_Unpaid_Inv'].sum(),
            "Total GTV Amount": df_filtered['GTV_AMOUNT_MCX'].sum()
        }
        st.write(filtered_totals)

        # Choose X and Y axis dimensions
        x_axis = st.selectbox('Choose X-axis dimension', df_filtered.columns)
        y_axis = st.selectbox('Choose Y-axis dimension', df_filtered.columns)

        # Aggregating the values for the chosen dimensions
        agg_df = df_filtered.groupby([x_axis, y_axis]).agg(
            Unpaid_Vol=('TOTAL_COST_Unpaid_Vol', 'sum'),
            Unpaid_Inv=('TOTAL_COST_Unpaid_Inv', 'sum'),
            GTV_Amount=('GTV_AMOUNT_MCX', 'sum'),
            booking_count=('booking_count','sum'),
            min_transaction_sum=('MIN_TRANSACTION','sum'),
            min_transaction_avg=('MIN_TRANSACTION','mean'),
            PMC3=('PMC3','sum')
        ).reset_index()

        # Calculate LGD dynamically for the aggregated values
        agg_df['LGD'] = (agg_df['Unpaid_Vol'] + agg_df['Unpaid_Inv']) / agg_df['GTV_Amount'] * 100
        agg_df['ABV'] = (agg_df['GTV_Amount']) / agg_df['booking_count']  
        agg_df['PMC3_UNIT'] = (agg_df['PMC3']) / agg_df['booking_count'] 

        # Compute Weighted LGD to give more importance to GTV and Booking Count
        agg_df['Weighted_LGD'] = agg_df['LGD'] * agg_df['GTV_Amount'] #* agg_df['booking_count']

        # Compute total weight for normalization
        total_weight = (agg_df['GTV_Amount'] * agg_df['booking_count']).sum()

        # Normalize to get final weighted LGD per cluster
        agg_df['Weighted_LGD'] = agg_df['Weighted_LGD'] / total_weight

        # Create a pivot table for the selected dimensions with LGD as values
        pivot_df = agg_df.pivot_table(values='Weighted_LGD', index=x_axis, columns=y_axis, aggfunc=np.mean)

        # Define color scale
        def color_scale(val):
            if val < -0.29:
                return 'background-color: red'
            else:
                return 'background-color: green'

        # Apply color scale to the pivot table
        styled_pivot_df = pivot_df.style.applymap(color_scale).format({'LGD': '{:.2f}%'});

        # Show the styled table
        st.write("Loss Given Default Table (in Percentage):")
        st.dataframe(styled_pivot_df)

        # Plotting the heatmap
        st.write("Loss Given Default Heatmap (Weighted by Booking Count & GTV):")
        fig = px.imshow(
                pivot_df,
                labels=dict(x=y_axis, y=x_axis, color='Weighted LGD (%)'),
                color_continuous_scale='RdYlBu',  # Inverted color scale
                zmin=pivot_df.min().min(),
                zmax=pivot_df.max().max()
            )
        fig.update_layout(
            title=f'Weighted LGD Heatmap ({x_axis} vs {y_axis})',
            xaxis_title=y_axis,
            yaxis_title=x_axis,
            coloraxis_colorbar=dict(title='Weighted LGD (%)')
        )
        st.plotly_chart(fig, use_container_width=True)
        

        st.plotly_chart(fig, use_container_width=True)
        # Display the aggregated dataframe for inspection
        st.write("Aggregated Dataframe:")
        st.dataframe(agg_df)

        ###########################################################################
        # Added Projection Section: Estimating the impact of deposit reduction on conversion rates, LGD, volume, and PMC3
        ###########################################################################

        st.subheader("Projected Impact of Deposit Reduction")
        st.write("Adjust the parameters below to see the projected changes when lowering deposit levels:")

        # Sidebar inputs for projection parameters
        st.sidebar.header("Projection Settings")
        base_conversion = st.sidebar.number_input("Baseline Conversion Rate", value=0.30, min_value=0.0, max_value=1.0, step=0.01,
                                                  help="Enter the current conversion rate (as a decimal, e.g., 0.30 for 30%).")
        default_rate = st.sidebar.number_input("Default Rate", value=0.05, min_value=0.0, max_value=1.0, step=0.01,
                                               help="Enter the default rate (as a decimal, e.g., 0.05 for 5%).")
        deposit_reduction = st.sidebar.slider("Deposit Reduction (%)", 0, 50, 10, step=1,
                                              help="Select the percentage reduction in the deposit.")
        conversion_improvement_per_10 = st.sidebar.number_input("Conversion Improvement per 10% Deposit Reduction (%)", value=5, min_value=0,
                                                                help="Expected percentage increase in conversion rate per 10% deposit reduction.")
        lgd_increase_per_10 = st.sidebar.number_input("LGD Increase per 10% Deposit Reduction (%)", value=2, min_value=0,
                                                      help="Expected percentage increase in LGD per 10% deposit reduction.")
        volume_improvement_per_10 = st.sidebar.number_input("Volume Increase per 10% Deposit Reduction (%)", value=3, min_value=0,
                                                            help="Expected percentage increase in volume (e.g., booking count) per 10% deposit reduction.")
        pmc3_improvement_per_10 = st.sidebar.number_input("PMC3 Increase per 10% Deposit Reduction (%)", value=1, min_value=0,
                                                          help="Expected percentage increase in PMC3 per 10% deposit reduction.")

        # Calculate new conversion rate based on deposit reduction
        new_conversion = base_conversion * (1 + (deposit_reduction / 10) * (conversion_improvement_per_10 / 100))

        # Compute the average LGD from the aggregated data (converted to a decimal)
        avg_LGD = agg_df['LGD'].mean() / 100  # LGD is in percentage in agg_df
        new_avg_LGD = avg_LGD * (1 + (deposit_reduction / 10) * (lgd_increase_per_10 / 100))

        # Assume exposure is the total GTV Amount from the filtered totals
        exposure = filtered_totals["Total GTV Amount"]

        # Calculate expected loss using the formula:
        # Expected Loss = Conversion Rate x Default Rate x LGD x Exposure
        expected_loss_baseline = base_conversion * default_rate * avg_LGD * exposure
        projected_expected_loss = new_conversion * default_rate * new_avg_LGD * exposure

        st.write("**Baseline Parameters:**")
        st.write(f"Baseline Conversion Rate: {base_conversion:.2%}")
        st.write(f"Default Rate: {default_rate:.2%}")
        st.write(f"Average LGD: {avg_LGD:.2%}")
        st.write(f"Exposure (Total GTV Amount): {exposure:,.2f}")

        st.write("---")
        st.write("**Expected Loss Projection:**")
        st.write(f"New Conversion Rate (after {deposit_reduction}% deposit reduction): {new_conversion:.2%}")
        st.write(f"New Average LGD: {new_avg_LGD:.2%}")
        st.write(f"Baseline Expected Loss: {expected_loss_baseline:,.2f}")
        st.write(f"Projected Expected Loss: {projected_expected_loss:,.2f}")

        loss_increase = projected_expected_loss - expected_loss_baseline
        loss_increase_pct = (loss_increase / expected_loss_baseline * 100) if expected_loss_baseline != 0 else np.nan
        st.write(f"Increase in Expected Loss: {loss_increase:,.2f} ({loss_increase_pct:.2f}%)")

        ###########################################################################
        # Additional Projections for Volumes and PMC3
        ###########################################################################

        # Projection for volume increase based on booking_count
        if 'booking_count' in agg_df.columns:
            baseline_volume = agg_df['booking_count'].sum()
        else:
            baseline_volume = 0
        new_volume = baseline_volume * (1 + (deposit_reduction / 10) * (volume_improvement_per_10 / 100))

        # Projection for PMC3 increase
        if 'PMC3' in agg_df.columns:
            baseline_PMC3 = agg_df['PMC3'].sum()
        else:
            baseline_PMC3 = 0
        new_PMC3 = baseline_PMC3 * (1 + (deposit_reduction / 10) * (pmc3_improvement_per_10 / 100))

        st.write("---")
        st.write("**Volume Projections:**")
        st.write(f"Baseline Volume (booking_count sum): {baseline_volume:,.0f}")
        st.write(f"Projected Volume (after {deposit_reduction}% deposit reduction): {new_volume:,.0f}")
        vol_increase = new_volume - baseline_volume
        vol_increase_pct = (vol_increase / baseline_volume * 100) if baseline_volume != 0 else np.nan
        st.write(f"Increase in Volume: {vol_increase:,.0f} ({vol_increase_pct:.2f}%)")

        st.write("---")
        st.write("**PMC3 Projections:**")
        st.write(f"Baseline PMC3: {baseline_PMC3:,.2f}")
        st.write(f"Projected PMC3 (after {deposit_reduction}% deposit reduction): {new_PMC3:,.2f}")
        pmc3_increase = new_PMC3 - baseline_PMC3
        pmc3_increase_pct = (pmc3_increase / baseline_PMC3 * 100) if baseline_PMC3 != 0 else np.nan
        st.write(f"Increase in PMC3: {pmc3_increase:,.2f} ({pmc3_increase_pct:.2f}%)")
