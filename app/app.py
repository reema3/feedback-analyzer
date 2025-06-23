import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from keybert import KeyBERT
import io

st.set_page_config(page_title="Customer Feedback Analyzer", layout="wide")

st.title("📊 Customer Feedback Analyzer")

uploaded_file = st.file_uploader("📁 Upload a CSV file with customer reviews", type="csv")

import nltk
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Sentiment Analysis Setup
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    df['Sentiment Score'] = df['Text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['Sentiment Label'] = df['Sentiment Score'].apply(
        lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral')
    )

    # Summarization Setup
    from transformers import pipeline
    summarizer = pipeline("summarization", model="t5-small", device=-1)
    all_text = " ".join(df['Text'].astype(str).tolist())[:3000]
    summary = summarizer(all_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

    # Keyword Extraction Setup
    from keybert import KeyBERT
    kw_model = KeyBERT()

    # Start Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Preview", "📊 Sentiment", "🧠 Summary", "🔑 Keywords"])

    # ---------------- Tab 1: Data Preview ----------------
    with tab1:
        st.subheader("📁 Uploaded Data")
        st.write(df[['Text']].head())

    # ---------------- Tab 2: Sentiment Analysis ----------------
    with tab2:
        st.subheader("📊 Sentiment Results")
        st.write(df[['Text', 'Sentiment Score', 'Sentiment Label']].head())

        st.subheader("Sentiment Distribution")
        import matplotlib.pyplot as plt
        sentiment_counts = df['Sentiment Label'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
        ax.set_title("Customer Sentiment Overview")
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sentiment Data as CSV",
            data=csv,
            file_name='sentiment_results.csv',
            mime='text/csv',
        )

    # ---------------- Tab 3: Summary ----------------
    with tab3:
        st.subheader("🧠 Summary of Reviews")
        # st.text_area("Summary:", value=summary, height=150)
        st.code(summary, language="text")


    # ---------------- Tab 4: Keywords ----------------
    with tab4:
        st.subheader("🔑 Top Keywords")
        keywords = kw_model.extract_keywords(all_text, top_n=10, stop_words='english')
        for word, score in keywords:
            st.write(f"• {word} (Score: {score:.2f})")

        st.subheader("💬 Keywords by Sentiment")
        for label in ['Positive', 'Neutral', 'Negative']:
            subset = df[df['Sentiment Label'] == label]
            text_block = " ".join(subset['Text'].astype(str).tolist())
            st.markdown(f"**{label} Reviews:**")
            if len(text_block.strip()) > 0:
                keywords = kw_model.extract_keywords(text_block, top_n=5, stop_words='english')
                for word, score in keywords:
                    st.write(f"• {word} (Score: {score:.2f})")
            else:
                st.write("No reviews available.")
else:
    st.info("Please upload a CSV file with a column named 'Text' or 'Review'.")



