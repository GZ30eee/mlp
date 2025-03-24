import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from io import BytesIO

# Function to perform equal width binning
def equal_width_binning(data, num_bins):
    min_val, max_val = min(data), max(data)
    bin_width = (max_val - min_val) / num_bins
    bins = [(min_val + i * bin_width, min_val + (i + 1) * bin_width) for i in range(num_bins)]
    binned_data = {f"Bin {i+1} ({bins[i][0]:.2f} - {bins[i][1]:.2f})": [] for i in range(num_bins)}
    
    for d in data:
        for i, bin_range in enumerate(bins):
            if bin_range[0] <= d < bin_range[1]:
                binned_data[f"Bin {i+1} ({bin_range[0]:.2f} - {bin_range[1]:.2f})"].append(d)
    return bins, binned_data

# Function to perform equal depth binning
def equal_depth_binning(data, num_bins):
    sorted_data = sorted(data)
    bin_size = len(data) // num_bins
    bins = [sorted_data[i * bin_size: (i + 1) * bin_size] for i in range(num_bins)]
    bin_ranges = [(min(bin), max(bin)) for bin in bins]
    binned_data = {f"Bin {i+1} ({bin_ranges[i][0]:.2f} - {bin_ranges[i][1]:.2f})": bins[i] for i in range(num_bins)}
    return bin_ranges, binned_data

# Function to generate random data
def generate_random_data():
    num_data_points = 20  # Fixed number of points
    return [random.uniform(1, 100) for _ in range(num_data_points)]

# Function to plot histogram
def plot_histogram(data, num_bins, binning_type):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=num_bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Data - {binning_type} Binning')
    st.pyplot(plt)

# Function to download results
def get_csv_download_link(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output

# Streamlit App
st.title("🗑️ Binning Application")
st.sidebar.header("User Input Controls")

# User input for binning type
binning_type = st.sidebar.selectbox("Select Binning Type", ["Equal Width", "Equal Depth"])

# User input for data source
data_source = st.sidebar.selectbox("Select Data Source", ["Enter Data", "Generate Random", "Upload CSV"])

# Display rules if CSV upload is selected
if data_source == "Upload CSV":
    st.subheader("📜 Rules for Uploading CSV")
    st.markdown("""
    - The file must be in **CSV format** (.csv).
    - Ensure there is **at least one numeric column**.
    - The application will automatically detect numeric columns.
    - If the CSV has missing values, they will be ignored.
    """)

# Input for data or generation of random data
data = []
if data_source == "Enter Data":
    user_data = st.sidebar.text_area("Enter Data (comma-separated)", value="10, 20, 30, 40, 50")
    data = list(map(float, user_data.split(',')))
elif data_source == "Generate Random":
    data = generate_random_data()
    st.write("Generated Random Data:", data)
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("CSV Data Preview:", df.head())
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            st.error("❌ No numeric columns found. Please upload a valid CSV file.")
        else:
            selected_column = st.sidebar.selectbox("Select Numeric Column", numeric_columns)
            data = df[selected_column].dropna().tolist()

# Input for the number of bins
num_bins = st.sidebar.number_input("Number of Bins", min_value=1, max_value=20, value=5, step=1)

# Perform binning and display the results
if st.sidebar.button("Perform Binning") and data:
    st.subheader("Binning Results")
    if binning_type == "Equal Width":
        bins, binned_data = equal_width_binning(data, num_bins)
        for i, bin_range in enumerate(bins):
            st.write(f"Bin {i+1} ({bin_range[0]:.2f} - {bin_range[1]:.2f}): {binned_data[f'Bin {i+1} ({bin_range[0]:.2f} - {bin_range[1]:.2f})']}")
    else:
        bin_ranges, binned_data = equal_depth_binning(data, num_bins)
        for i, bin_range in enumerate(bin_ranges):
            st.write(f"Bin {i+1} ({bin_range[0]:.2f} - {bin_range[1]:.2f}): {binned_data[f'Bin {i+1} ({bin_range[0]:.2f} - {bin_range[1]:.2f})']}")

    # Plot histogram
    plot_histogram(data, num_bins, binning_type)

    # Download results as CSV
    bin_df = pd.DataFrame([(bin_label, values) for bin_label, values in binned_data.items()], columns=["Bin", "Values"])
    csv_output = get_csv_download_link(bin_df)
    st.download_button("Download Binned Data", csv_output, "binned_data.csv", "text/csv")
