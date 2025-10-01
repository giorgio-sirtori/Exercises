# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:17:14 2025

@author: gsirtori
"""

import streamlit as st
import pandas as pd
# --- Page Configuration ---
st.set_page_config(
    page_title="CSV Joiner & Grouper",
    page_icon=":link:",
    layout="wide"
)
st.title(":link: CSV Joiner & Grouper")
st.write("Upload two CSV files, define how to group them, select the join columns, and get a full outer join result.")
# --- Helper function to perform grouping ---
def group_dataframe(df, group_cols, agg_col, agg_func):
    """Groups a DataFrame and returns the result."""
    if not group_cols or not agg_col or not agg_func:
        return df # Return original if no grouping is defined
    # Create the aggregation dictionary for the .agg() function
    agg_dict = {agg_col: agg_func}
    try:
        grouped_df = df.groupby(group_cols).agg(agg_dict).reset_index()
        return grouped_df
    except Exception as e:
        st.error(f"Error during grouping: {e}")
        return None
# --- Main Application ---
# Step 1: File Uploads
col1, col2 = st.columns(2)
with col1:
    st.header("Source A")
    uploaded_file_A = st.file_uploader("Upload your first CSV", type="csv", key="A")
with col2:
    st.header("Source B")
    uploaded_file_B = st.file_uploader("Upload your second CSV", type="csv", key="B")
# Proceed only if both files are uploaded
if uploaded_file_A and uploaded_file_B:
    df_A = pd.read_csv(uploaded_file_A)
    df_B = pd.read_csv(uploaded_file_B)
    st.success("Files uploaded successfully! Now, define your groupings.")
    st.markdown("---")
    # --- Step 2: Grouping Interface ---
    st.header("2. Define Grouping Operations")
    g_col1, g_col2 = st.columns(2)
    # --- Grouping for DataFrame A ---
    with g_col1:
        st.subheader("Grouping for Source A")
        cols_A = df_A.columns.tolist()
        group_cols_A = st.multiselect("Columns to group by", options=cols_A, key="group_A")
        agg_col_A = st.selectbox("Column to aggregate", options=cols_A, key="agg_col_A")
        agg_func_A = st.selectbox("Aggregation function", options=['sum', 'count', 'mean', 'min', 'max'], key="agg_func_A")
    # --- Grouping for DataFrame B ---
    with g_col2:
        st.subheader("Grouping for Source B")
        cols_B = df_B.columns.tolist()
        group_cols_B = st.multiselect("Columns to group by", options=cols_B, key="group_B")
        agg_col_B = st.selectbox("Column to aggregate", options=cols_B, key="agg_col_B")
        agg_func_B = st.selectbox("Aggregation function", options=['sum', 'count', 'mean', 'min', 'max'], key="agg_func_B")
    # Apply the grouping
    grouped_A = group_dataframe(df_A, group_cols_A, agg_col_A, agg_func_A)
    grouped_B = group_dataframe(df_B, group_cols_B, agg_col_B, agg_func_B)
    st.markdown("---")
    # --- Step 3: Join Interface ---
    st.header("3. Define the Join")
    # Use the grouped columns as potential join keys
    # Note: A true drag-and-drop is complex; this mapping is a very intuitive alternative.
    join_col1, join_col2 = st.columns(2)
    with join_col1:
        st.subheader("Join Keys from Source A")
        # Let user select which of their chosen group_cols to use for the join
        join_keys_A = st.multiselect(
            "Select join keys from grouped Source A:",
            options=group_cols_A
        )
    with join_col2:
        st.subheader("Join Keys from Source B")
        # Let user select which of their chosen group_cols to use for the join
        join_keys_B = st.multiselect(
            "Select join keys from grouped Source B:",
            options=group_cols_B
        )
    # --- Step 4: Perform the Join and Display Output ---
    if st.button(":rocket: Perform Full Outer Join"):
        if not join_keys_A or not join_keys_B:
            st.warning("Please select join keys for both sources.")
        elif len(join_keys_A) != len(join_keys_B):
            st.error("You must select the same number of join keys for both sources.")
        else:
            try:
                # Perform the join
                merged_df = pd.merge(
                    grouped_A,
                    grouped_B,
                    left_on=join_keys_A,
                    right_on=join_keys_B,
                    how='outer',
                    suffixes=('_A', '_B')
                )
                st.header(":sparkles: Join Result")
                st.dataframe(merged_df)
                # Provide a download button for the result
                csv_data = merged_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=":inbox_tray: Download Result as CSV",
                    data=csv_data,
                    file_name="joined_output.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"An error occurred during the join: {e}")





