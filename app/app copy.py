import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from keybert import KeyBERT

st.set_page_config(page_title="Customer Feedback Analyzer", layout="wide")

st.title("📊 Customer Feedback Analyzer")

uploaded_file = st.file_uploader("📁 Upload a CSV file with customer reviews", type="csv")

import nltk
nltk.download('vader_lexicon')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # Initialize VADER
    sid = SentimentIntensityAnalyzer()

    # Apply sentiment analysis
    df['Sentiment Score'] = df['Text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['Sentiment Label'] = df['Sentiment Score'].apply(
        lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral')
    )

    st.subheader("Sentiment Results")
    st.write(df[['Text', 'Sentiment Score', 'Sentiment Label']].head())

    st.success("Sentiment analysis completed!")

    st.subheader("Sentiment Distribution")

    sentiment_counts = df['Sentiment Label'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
    ax.set_title("Customer Sentiment Overview")
    st.pyplot(fig)

    st.subheader("Summary of All Customer Feedback")

    # Join all reviews into a single block of text (you can tweak this)
    all_text = " ".join(df['Text'].astype(str).tolist())[:3000]  # limit to 3000 chars to avoid timeout

    with st.spinner("Generating summary..."):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

    summary = summarizer(all_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

    st.success("Summary generated:")
    st.write(summary)

    # st.subheader("🔑 Top Keywords from Reviews")

    # # Initialize the keyword extractor
    # kw_model = KeyBERT()

    # # Combine all reviews into one text block
    # review_texts = " ".join(df["Text"].astype(str).tolist())

    # # Extract top 10 keywords
    # keywords = kw_model.extract_keywords(review_texts, top_n=10, stop_words='english')

    # # Display results
    # for word, score in keywords:
    #     st.write(f"• {word} (Score: {score:.2f})")

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



