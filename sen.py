import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import re

# Set up Streamlit App
st.title("Sentiment Analysis App")
st.markdown("Upload a CSV file with a `text` column for sentiment analysis.")

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Step 1: Load Data
        data = pd.read_csv(uploaded_file)

        if 'text' not in data.columns:
            st.error("The uploaded CSV file must contain a 'text' column.")
        else:
            st.write("### Sample Data (First 5 Rows)")
            st.write(data.head())

            # Step 2: Preprocess Text
            def clean_text(text):
                if isinstance(text, str):  # Ensure the input is a string
                    text = re.sub(r"http\S+", "", text)  # Remove URLs
                    text = re.sub(r"@\w+", "", text)    # Remove mentions
                    text = re.sub(r"#\w+", "", text)    # Remove hashtags
                    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation
                    return text.strip().lower()         # Normalize case and trim spaces
                return ""  # Return an empty string for non-string values

            data['cleaned_text'] = data['text'].apply(clean_text)

            # Remove rows with empty 'cleaned_text'
            data = data[data['cleaned_text'] != ""]

            if data.empty:
                st.error("No valid text data after cleaning. Please check your file.")
            else:
                # Step 3: Sentiment Analysis
                def analyze_sentiment(text):
                    if text.strip():  # Only analyze non-empty text
                        analysis = TextBlob(text)
                        if analysis.sentiment.polarity > 0:
                            return "Positive"
                        elif analysis.sentiment.polarity == 0:
                            return "Neutral"
                        else:
                            return "Negative"
                    return "Neutral"  # Default to Neutral for empty or invalid text

                data['Sentiment'] = data['cleaned_text'].apply(analyze_sentiment)

                st.write("### Sentiment Analysis Results")
                st.write(data.head())

                # Step 4: Visualize Sentiment Distribution
                sentiment_counts = data['Sentiment'].value_counts()

                # Pie Chart
                st.write("### Sentiment Distribution (Pie Chart)")
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                sentiment_counts.plot.pie(
                    autopct='%1.1f%%', 
                    colors=['lightgreen', 'gold', 'red'], 
                    startangle=140, 
                    labels=sentiment_counts.index,
                    ax=ax1
                )
                ax1.set_ylabel("")  # Remove default ylabel
                st.pyplot(fig1)

                # Bar Graph
                st.write("### Sentiment Distribution (Bar Graph)")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sentiment_counts.plot(
                    kind='bar', 
                    color=['lightgreen', 'gold', 'red'], 
                    edgecolor='black',
                    ax=ax2
                )
                ax2.set_title("Sentiment Distribution")
                ax2.set_xlabel("Sentiment")
                ax2.set_ylabel("Number of Posts")
                ax2.set_xticks(range(len(sentiment_counts)))
                ax2.set_xticklabels(sentiment_counts.index, rotation=0)
                st.pyplot(fig2)

                # Step 5: Provide Download Option for Processed Data
                st.write("### Download Processed Data")
                processed_file = data.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=processed_file,
                    file_name="processed_sentiment_data.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to proceed.")
