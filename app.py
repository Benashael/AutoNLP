import streamlit as st
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import docx2txt
from langdetect import detect
import base64
from io import BytesIO

# Set up Streamlit app
st.title("AutoNLP Streamlit Web App")

st.set_page_config(page_title="AutoNLP Application", page_icon="📚", layout="wide")

page = st.sidebar.selectbox("Choose a task", ["Home Page", "Tokenization", "Stopwords Removal", "Stemming", "Lemmatization", "POS Tagging", "Word Cloud", "N-Grams"])

# Function to tokenize text
def tokenize_text(text, tokenization_type):
    if tokenization_type == "Word Tokenization":
        tokens = word_tokenize(text)
    else:
        tokens = sent_tokenize(text)
    return tokens

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word.lower() not in stop_words]
    return filtered_text

# Function to perform stemming
def perform_stemming(text):
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in text]
    return stemmed_text

# Function to perform lemmatization
def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text]
    return lemmatized_text

# Function for Part-of-Speech (POS) tagging
def pos_tagging(text):
    pos_tags = nltk.pos_tag(text)
    return pos_tags

# Function to create a word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Function to create n-grams
def create_ngrams(tokens, n):
    n_grams = list(ngrams(tokens, n))
    return n_grams

# Function to generate n-grams text
def generate_ngrams_text(n_grams):
    n_grams_text = [" ".join(gram) for gram in n_grams]
    return n_grams_text

if page == "Home Page":
    # Home page content
    st.title("Welcome to the AutoNLP Web App!")
    
    st.write(
        "This web app is designed to automate various Natural Language Processing (NLP) tasks. "
        "You can perform the following tasks:"
    )
    
    st.markdown("1. **Tokenization:** Tokenize text into words or sentences.")
    st.markdown("2. **Stopwords Removal:** Remove common stopwords from the text.")
    st.markdown("3. **Stemming:** Apply word stemming to the text.")
    st.markdown("4. **Lemmatization:** Perform word lemmatization on the text.")
    st.markdown("5. **Part-of-Speech (POS) Tagging:** Tag words with their grammatical roles.")
    st.markdown("6. **Word Cloud Generation:** Generate a word cloud from the text.")
    st.markdown("7. **N-Grams:** Create uni-grams, bi-grams, or tri-grams from the text.")
    
    st.write(
        "Please select a task from the sidebar on the left to get started. You can choose the input type, "
        "input your text, and perform the desired NLP tasks. Enjoy exploring the capabilities of this AutoNLP app!"
    )

# Tokenization Page
elif page == "Tokenization":
    st.subheader("Tokenization")
    tokenization_type = st.radio("Choose tokenization type", ["Word Tokenization", "Sentence Tokenization"])
    input_type = st.radio("Choose input type", ["Text Input", "TXT File Import"])
    
    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("Enter text:")
        if len(word_tokenize(text_input)) > max_word_limit:
            st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
        else:
            tokens = tokenize_text(text_input, tokenization_type)
            st.write("Tokens:", tokens)
            
            # Download tokenized content as a txt file
            if st.button("Download Tokenized Content"):
                tokenized_content = " ".join(tokens)
                tokenized_file = BytesIO(tokenized_content.encode())
                st.download_button(
                    label="Download Tokenized Content",
                    data=tokenized_file,
                    key="tokenized_content.txt",
                    on_click=None,
                )
    
    elif input_type == "TXT File Import":
        max_word_limit = 3000
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if uploaded_file is not None:
            file_contents = uploaded_file.read()
            try:
                file_contents = file_contents.decode("utf-8")
                if len(word_tokenize(file_contents)) > max_word_limit:
                    st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
                else:
                    tokens = tokenize_text(file_contents, tokenization_type)
                    st.write("Tokens:", tokens)
                    
                    # Download tokenized content as a txt file
                    if st.button("Download Tokenized Content"):
                        tokenized_content = " ".join(tokens)
                        tokenized_file = BytesIO(tokenized_content.encode())
                        st.download_button(
                            label="Download Tokenized Content",
                            data=tokenized_file,
                            key="tokenized_content.txt",
                            on_click=None,
                        )
            except UnicodeDecodeError:
                st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
        else:
            st.info("Please upload a .txt file.")
