import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from io import BytesIO
import json

# Set up Streamlit App
st.title("Sentiment Analysis App")
st.markdown("Upload a CSV file with a `text` column or choose a sample dataset.")

# Prebuilt sample datasets
def load_sample_data():
    return {
        "Sample 1": pd.DataFrame({"text": [
            "I love this product!", "Worst experience ever.", "It was okay.", "Absolutely fantastic!", "Not great, not terrible.",
            "Highly recommend this!", "I would never buy this again.", "Just mediocre.", "Incredible value for money!", "Very disappointing.",
            "Exceeded my expectations!", "I’m really unhappy with my purchase.", "It’s just alright.", "Best decision I ever made!", "I regret buying this.", "Satisfactory but could be better."
        ]}),
        "Sample 2": pd.DataFrame({"text": [
            "Best service I've had!", "I hate it!", "Meh, could be better.", "Loved it so much!", "Average experience.",
            "Exceptional customer support!", "Terrible and frustrating.", "Not bad, but not great either.", "Absolutely loved every moment!", "I wouldn't recommend this to anyone.",
            "Service was decent, nothing special.", "Fabulous experience overall!", "Completely unsatisfied with the service.", "It was just okay.", "Really impressed with the attention to detail!", "Disappointing service, I expected more."
        ]}),
        "Sample 3": pd.DataFrame({"text": [
            "So bad!", "Not impressed.", "Excellent product!", "Could be worse.", "Very happy with my purchase!",
            "Absolutely terrible experience!", "Mediocre at best.", "Fantastic quality!", "I’m quite satisfied with this.", "Displeased with my choice.",
            "Surpassed my expectations!", "It’s not what I hoped for.", "Thrilled with the results!", "Could definitely improve.", "I love it!"
        ]}),
        "Sample 4": pd.DataFrame({"text": [
            "Terrible service!", "Pretty decent.", "I wouldn't recommend it.", "Awesome deal!", "Just okay.",
            "Horrible experience overall!", "Fairly good service.", "Not worth the price.", "Great value for the money!", "Satisfactory but not remarkable.",
            "Very poor customer support.", "Exceeded my expectations on quality!", "It was just fine, nothing more.", "Would definitely buy again!", "Average at best."
        ]})
    }

sample_data = load_sample_data()
selected_sample = st.selectbox("Or choose a sample dataset:", list(sample_data.keys()))

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

data = pd.read_csv(uploaded_file) if uploaded_file else sample_data[selected_sample]

if 'text' not in data.columns:
    st.error("The dataset must contain a 'text' column.")
    st.stop()

# Text Preprocessing
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip().lower()

data['cleaned_text'] = data['text'].apply(lambda x: clean_text(str(x)))

# Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return sentiment, polarity, subjectivity

data[['Sentiment', 'Polarity', 'Subjectivity']] = data['cleaned_text'].apply(lambda x: pd.Series(analyze_sentiment(x)))

# Display Results
st.write("### Sentiment Analysis Results")
st.write(data)

# Sentiment Distribution Visualization
st.write("### Sentiment Distribution")
sentiment_counts = data['Sentiment'].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=['lightgreen', 'gold', 'red'], ax=ax)
ax.set_title("Sentiment Distribution")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Entries")
st.pyplot(fig)

# Word Cloud Visualization
def generate_wordcloud(sentiment):
    text = " ".join(data[data['Sentiment'] == sentiment]['cleaned_text'])
    if text:
        wc = WordCloud(background_color='white', max_words=100, colormap='coolwarm').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        return fig
    return None

st.write("### Word Clouds")
for sentiment in ["Positive", "Neutral", "Negative"]:
    st.write(f"**{sentiment} Sentiment**")
    fig = generate_wordcloud(sentiment)
    if fig:
        st.pyplot(fig)
    else:
        st.write("No words available for this sentiment.")

# Export Options
st.write("### Download Processed Data")
data_csv = data.to_csv(index=False).encode()
data_json = data.to_json(orient='records').encode()
data_excel = BytesIO()
data.to_excel(data_excel, index=False, engine='openpyxl')

download_format = st.radio("Choose format:", ["CSV", "JSON", "Excel"])
if download_format == "CSV":
    st.download_button("Download CSV", data=data_csv, file_name="processed_data.csv", mime="text/csv")
elif download_format == "JSON":
    st.download_button("Download JSON", data=data_json, file_name="processed_data.json", mime="application/json")
elif download_format == "Excel":
    st.download_button("Download Excel", data=data_excel.getvalue(), file_name="processed_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
