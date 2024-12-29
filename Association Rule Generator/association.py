import streamlit as st
import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules

# Streamlit app title
st.title('Apriori Algorithm for Association Rule Mining')

# List of possible items (for random transactions)
items_list = ['milk', 'bread', 'butter', 'jam', 'cheese', 'eggs', 'apple', 'banana', 'orange', 'yogurt']

# Step 1: Get User Input for Transactions or Generate Random Transactions
st.subheader("Enter Transactions or Generate Random Data")

# Option to enter custom transactions or use random data
transaction_input_option = st.radio("Choose your option", ('Enter Transactions', 'Generate Random Transactions'))

if transaction_input_option == 'Enter Transactions':
    # Text area to input transactions
    transaction_input = st.text_area("Enter transactions (one per line, items separated by commas)", height=200)

    # Process user input
    if transaction_input:
        transactions = [line.split(',') for line in transaction_input.strip().split('\n')]
        transactions = [[item.strip() for item in transaction] for transaction in transactions]
    else:
        transactions = []

elif transaction_input_option == 'Generate Random Transactions':
    # Generate random transactions
    num_transactions = random.randint(5, 15)  # Random number of transactions
    transactions = []
    for _ in range(num_transactions):
        # Randomly decide how many items to include in each transaction (between 1 and 5 items)
        num_items_in_transaction = random.randint(1, 5)
        transaction = random.sample(items_list, num_items_in_transaction)  # Random sample of items
        transactions.append(transaction)

    # Display the generated transactions
    st.write("Generated Random Transactions:")
    st.write(transactions)

# Step 2: Convert Data to One-Hot Encoding if there are transactions
if transactions:
    df = pd.DataFrame(transactions)
    df = df.stack().str.get_dummies().groupby(level=0).sum()

    # Convert to boolean values (True/False)
    df = df.astype(bool)

    # Step 3: Get minimum support and confidence from user input
    min_support = st.slider("Select Minimum Support", 0.0, 1.0, 0.4, 0.05)
    min_confidence = st.slider("Select Minimum Confidence", 0.0, 1.0, 0.7, 0.05)

    # Step 4: Run Apriori Algorithm
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # Display Frequent Itemsets to see if any itemsets were found
    st.subheader("Frequent Itemsets")
    if frequent_itemsets.empty:
        st.warning(f"No frequent itemsets found with support >= {min_support}")
    else:
        st.write(frequent_itemsets)

    # Step 5: Generate Association Rules (only if there are frequent itemsets)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        # Display Association Rules
        st.subheader("Association Rules")
        if rules.empty:
            st.warning(f"No association rules found with confidence >= {min_confidence}")
        else:
            st.write(rules)

else:
    st.warning("No transactions to analyze. Please enter some data or generate random transactions.")
