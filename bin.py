import streamlit as st
import numpy as np
import random

# Function to perform equal width binning
def equal_width_binning(data, num_bins):
    min_val = min(data)
    max_val = max(data)
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

# Streamlit App
def main():
    st.title("Binning Demo")

    # User input for binning type
    binning_type = st.selectbox("Select Binning Type", ["Equal Width", "Equal Depth"])

    # User input for data source
    data_source = st.selectbox("Select Data Source", ["Enter Data", "Generate Random"])

    # Input for data or generation of random data
    if data_source == "Enter Data":
        user_data = st.text_input("Enter Data (comma-separated)", value="10, 20, 30, 40, 50")
        data = list(map(float, user_data.split(',')))
    elif data_source == "Generate Random":
        data = generate_random_data()
        st.write("Generated Random Data:", data)

    # Input for the number of bins
    num_bins = st.number_input("Number of Bins", min_value=1, max_value=20, value=5, step=1)

    # Perform binning and display the results
    if st.button("Perform Binning"):
        if binning_type == "Equal Width":
            bins, binned_data = equal_width_binning(data, num_bins)
            st.subheader("Equal Width Binning Steps:")
            for i, bin_range in enumerate(bins):
                st.write(f"Bin {i+1} ({bin_range[0]:.2f} - {bin_range[1]:.2f}): {binned_data[f'Bin {i+1} ({bin_range[0]:.2f} - {bin_range[1]:.2f})']}")
        
        elif binning_type == "Equal Depth":
            bin_ranges, binned_data = equal_depth_binning(data, num_bins)
            st.subheader("Equal Depth Binning Steps:")
            for i, bin_range in enumerate(bin_ranges):
                st.write(f"Bin {i+1} ({bin_range[0]:.2f} - {bin_range[1]:.2f}): {binned_data[f'Bin {i+1} ({bin_range[0]:.2f} - {bin_range[1]:.2f})']}")

if __name__ == "__main__":
    main()