import streamlit as st
import pandas as pd

# Title of the app
st.title('Data Cleaning Assistant')

# Upload CSV/Excel files
uploaded_file = st.file_uploader("Choose a file (CSV, Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    # Load the dataset
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)

    # Show dataset preview
    st.subheader('Dataset Preview')
    st.write(data.head())

    # Handling missing values
    st.subheader('Handle Missing Values')
    missing_value_action = st.selectbox('Choose an action for missing values:', ['Drop', 'Fill with default value'])

    if missing_value_action == 'Drop':
        data = data.dropna()
        st.write('Missing values dropped.')
    elif missing_value_action == 'Fill with default value':
        fill_value = st.text_input('Enter the value to fill missing data:', '0')
        data = data.fillna(fill_value)
        st.write(f'Missing values filled with {fill_value}.')

    # Handling duplicates
    st.subheader('Remove Duplicates')
    remove_duplicates = st.checkbox('Remove duplicate rows', value=False)
    if remove_duplicates:
        data = data.drop_duplicates()
        st.write('Duplicates removed.')

    # Data type correction
    st.subheader('Correct Data Types')
    column_name = st.selectbox('Select a column to change its data type:', data.columns)
    new_dtype = st.selectbox('Select new data type for the column:', ['int64', 'float64', 'object'])
    
    if st.button('Change Data Type'):
        data[column_name] = data[column_name].astype(new_dtype)
        st.write(f'Column "{column_name}" converted to {new_dtype}.')

    # Show cleaned data preview
    st.subheader('Cleaned Data Preview')
    st.write(data.head())

    # Export cleaned dataset
    st.subheader('Download Cleaned Data')
    csv = data.to_csv(index=False).encode()
    st.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
