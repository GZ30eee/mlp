import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from io import StringIO

# Streamlit app title
st.title('Apriori Algorithm for Association Rule Mining')

# List of sample CSV files
sample_csv_files = {
    "Sample 1": "milk,bread,cheese\ncheese,butter,apple\nbanana,orange,yogurt\neggs,spinach,carrot\ncereal,granola,muffin",
    "Sample 2": "eggs,bread,butter\njam,cheese,milk\napple,banana,orange\npasta,sauce,meatballs\nrice,beans,tortilla",
    "Sample 3": "milk,cheese,eggs\nbread,butter,banana\njam,orange,yogurt\ntomato,salad,dressing\nchicken,rice,broccoli",
    "Sample 4": "apple,banana,bread\nmilk,cheese,butter\norange,eggs,yogurt\ngrapes,pineapple,mango\npeanut_butter,jelly,toast"
}

# Step 1: User Input (CSV Upload or Sample Selection)
st.subheader("Upload CSV or Try Sample Data")

data_input_option = st.radio("Choose your option", ('Try Samples','Upload CSV'))

if data_input_option == 'Upload CSV':
    uploaded_file = st.file_uploader("Upload a CSV file with transactions", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        transactions = df.values.tolist()
    else:
        transactions = []
    
    st.markdown("### CSV Format Rules:")
    st.markdown("- Each row represents a transaction.")
    st.markdown("- Items in a transaction should be separated by commas.")
    st.markdown("- No empty rows or missing values.")
    st.markdown("**Example:**")
    st.code("""
milk,bread,cheese
eggs,butter,apple
banana,orange,yogurt
""", language='csv')

elif data_input_option == 'Try Samples':
    sample_choice = st.selectbox("Choose a sample dataset", list(sample_csv_files.keys()))
    df = pd.read_csv(StringIO(sample_csv_files[sample_choice]), header=None)
    transactions = df.values.tolist()
    st.write("Sample Transactions:")
    st.write(df)

# Step 2: Convert Transactions to One-Hot Encoding
if transactions:
    df = pd.DataFrame(transactions)
    df = df.stack().str.get_dummies().groupby(level=0).sum()
    df = df.astype(bool)
    
    # User-defined parameters
    min_support = st.slider("Select Minimum Support", 0.01, 1.0, 0.2, 0.01)
    min_confidence = st.slider("Select Minimum Confidence", 0.1, 1.0, 0.5, 0.05)
    min_lift = st.slider("Select Minimum Lift", 0.5, 5.0, 1.0, 0.1)
    min_leverage = st.slider("Select Minimum Leverage", 0.0, 1.0, 0.0, 0.01)

    # Step 3: Apply Apriori Algorithm
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    st.subheader("Frequent Itemsets")
    st.write(frequent_itemsets)
    
    # Visualization of Frequent Itemsets
    if not frequent_itemsets.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=frequent_itemsets['support'], y=frequent_itemsets['itemsets'].astype(str))
        plt.xlabel("Support")
        plt.ylabel("Itemsets")
        plt.title("Frequent Itemsets")
        st.pyplot(plt)
    
    # Step 4: Generate Association Rules
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules[(rules['lift'] >= min_lift) & (rules['leverage'] >= min_leverage)]
        
        st.subheader("Association Rules")
        if rules.empty:
            st.warning("No association rules found with the given parameters.")
        else:
            st.write(rules)
            
            # Visualization of Rules
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=rules['support'], y=rules['confidence'], size=rules['lift'], hue=rules['leverage'], palette='coolwarm', legend=True)
            plt.xlabel("Support")
            plt.ylabel("Confidence")
            plt.title("Association Rules Visualization")
            st.pyplot(plt)
            
            # Step 5: Download Option
            csv_data = rules.to_csv(index=False).encode('utf-8')
            st.download_button("Download Rules as CSV", csv_data, "association_rules.csv", "text/csv")
else:
    st.warning("No transactions to analyze. Please upload a CSV or try a sample dataset.")
