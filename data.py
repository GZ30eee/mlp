import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# Predefined Sample CSVs with Imperfections
sample_data = {
    "Sales Data": pd.DataFrame({
        "Product": ["A", "B", "C", "D", None, "B", "C"],  # Duplicate and None value
        "Revenue": [1000, 1500, 2000, 2500, 3000, None, 2500],  # Missing revenue and duplicate
        "Units Sold": [10, 15, 20, 25, 30, 'fifteen', 25]  # Inconsistent data type (string)
    }),
    "Employee Data": pd.DataFrame({
        "Employee": ["John", "Alice", "Bob", "Eve", "John", None],  # Duplicate and None value
        "Salary": [50000, 60000, 55000, 'not available', 70000, 60000],  # Inconsistent salary entry
        "Department": ["HR", "IT", None, "Marketing", "HR", "IT"]  # Missing department
    }),
    "Customer Reviews": pd.DataFrame({
        "Customer": ["Tom", "Jerry", None, "Donald"],  # Missing customer name
        "Rating": [4.5, 3.8, 'five', 5.0],  # Inconsistent rating entry
        "Review": ["Good", None, "Nice", "Excellent"]  # Missing review text
    }),
    "Stock Prices": pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", None, "2024-01-04"],  # Missing date
        "Price": [150, 'high', 148, 155]  # Inconsistent price entry (string)
    }),
    "Student Grades": pd.DataFrame({
        "Name": ["Sam", None, "Lily", "Mark"],  # Missing student name
        "Math": [90, 85, 'eighty-eight', 92],  # Inconsistent grade entry (string)
        "Science": [85, None, 89, 'ninety-one']  # Missing science grade and string entry
    }),
    "Website Traffic": pd.DataFrame({
        "Day": ["Monday", None, "Wednesday", "Thursday"],  # Missing day
        "Visitors": [500, 'six hundred', None, 700]  # Inconsistent visitor count (string)
    }),
}

# Title
st.title('🧹 Data Cleaning Assistant')

# Sidebar: Choose between Uploading or Selecting Sample Data
option = st.selectbox("Choose Data Source:", ["Use Sample Data","Upload CSV"])

# Display CSV Rules if 'Upload CSV' is selected
if option == "Upload CSV":
    st.markdown("""
    ### 📌 Rules for Uploading CSV:
    - File must be **CSV or Excel** format.
    - Column names should be **consistent and clear**.
    - Avoid special characters in column names.
    - Handle missing values properly before uploading.
    - Ensure data is clean and structured.
    """)

# File uploader
uploaded_file = None
if option == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Choose a file", type=['csv', 'xlsx'])

# Load predefined data
if option == "Use Sample Data":
    selected_sample = st.selectbox("Choose a sample dataset:", list(sample_data.keys()))
    data = sample_data[selected_sample]
else:
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
    else:
        st.warning("Please upload a file to proceed.")
        st.stop()

# Display dataset preview
st.subheader('📊 Dataset Preview')
st.write(data.head())

# Data Summary
st.subheader('📋 Data Summary')
st.write(data.describe())
st.write("Missing Values Count:")
st.write(data.isnull().sum())

# Data Visualization
st.subheader('📈 Data Visualization')
col_to_plot = st.selectbox("Select a column to visualize", data.columns)

fig, ax = plt.subplots()
if data[col_to_plot].dtype in ['int64', 'float64']:
    ax.hist(data[col_to_plot].dropna(), bins=20, color='blue', edgecolor='black')  
    ax.set_title(f'Distribution of {col_to_plot}')
    ax.set_xlabel(col_to_plot)
    ax.set_ylabel("Frequency")
else:
    data[col_to_plot].value_counts().plot(kind='bar', ax=ax, color='green')
    ax.set_title(f'Count of {col_to_plot}')
    ax.set_xlabel(col_to_plot)
    ax.set_ylabel("Count")

st.pyplot(fig)

# Handling Missing Values
st.subheader('🚀 Handle Missing Values')
action = st.selectbox('Choose an action:', ['Drop rows', 'Fill with value'])
if action == 'Drop rows':
    data.dropna(inplace=True)
    st.success("Missing values dropped!")
else:
    fill_value = st.text_input("Enter value to fill missing data:", '0')
    data.fillna(fill_value, inplace=True)
    st.success(f"Missing values filled with {fill_value}.")

# Removing Duplicates
if st.checkbox("Remove Duplicates"):
    data.drop_duplicates(inplace=True)
    st.success("Duplicate rows removed!")

# Change Data Type
st.subheader('🔄 Change Data Type')
col_name = st.selectbox("Select a column to change its type", data.columns)
dtype = st.selectbox("New Data Type", ['int64', 'float64', 'object'])
if st.button("Convert Type"):
    try:
        data[col_name] = data[col_name].astype(dtype)
        st.success(f'Column {col_name} converted to {dtype}!')
    except ValueError as e:
        st.error(f"Error converting column: {e}")

# Search & Replace
st.subheader("🔍 Search & Replace Values")
search_col = st.selectbox("Select Column", data.columns)
search_val = st.text_input("Search for value")
replace_val = st.text_input("Replace with")
if st.button("Replace"):
    data[search_col] = data[search_col].replace(search_val, replace_val)
    st.success(f"Replaced '{search_val}' with '{replace_val}' in {search_col}!")

# Cleaned Data Preview
st.subheader('📝 Cleaned Data Preview')
st.write(data.head())

# Exporting Cleaned Data
st.subheader("📤 Download Cleaned Data")
file_format = st.selectbox("Choose Format", ["CSV", "Excel"])
if file_format == "CSV":
    csv_data = data.to_csv(index=False).encode()
    st.download_button("Download CSV", csv_data, "cleaned_data.csv", "text/csv")
else:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Cleaned Data')
        writer._save()
    st.download_button("Download Excel", output.getvalue(), "cleaned_data.xlsx", 
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
