import streamlit as st
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import docx2txt
from langdetect import detect
import base64
from io import BytesIO
import pandas as pd

# Set up Streamlit app
st.set_page_config(page_title="AutoNLP Application", page_icon="📚", layout="wide")

st.title("AutoNLP Streamlit Web App")

page = st.sidebar.radio("**Select a Page**", ["Home Page", "Tokenization", "Stopwords Removal", "Stemming", "Lemmatization", "POS Tagging", "Word Cloud", "N-Grams", "About"])

# Function to tokenize text
@st.cache_resource
def tokenize_text(text, tokenization_type):
    if tokenization_type == "Word Tokenization":
        tokens = word_tokenize(text)
    else:
        tokens = sent_tokenize(text)
    return tokens

# Function to remove stopwords
@st.cache_resource
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word.lower() not in stop_words]
    return filtered_text

# Function to perform stemming
@st.cache_resource
def perform_stemming(text):
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in text]
    return stemmed_text

# Function to perform lemmatization
@st.cache_resource
def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text]
    return lemmatized_text

# Function for Part-of-Speech (POS) tagging
@st.cache_resource
def pos_tagging(text):
    pos_tags = nltk.pos_tag(text)
    return pos_tags

# Function to create a word cloud
@st.cache_resource
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Function to create n-grams
@st.cache_resource
def create_ngrams(tokens, n):
    n_grams = list(ngrams(tokens, n))
    return n_grams

# Function to generate n-grams text
@st.cache_resource
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
    
    st.markdown("1. **Tokenization Page:** Tokenize text into words or sentences. And, you also have the option to copy the processed content to clipboard.")
    st.markdown("2. **Stopwords Removal Page:** Remove common stopwords from the text. And, you also have the option to copy the processed content to clipboard.")
    st.markdown("3. **Stemming Page:** Apply word stemming to the text. And, you also have the option to copy the processed content to clipboard.")
    st.markdown("4. **Lemmatization Page:** Perform word lemmatization on the text. And, you also have the option to copy the processed content to clipboard.")
    st.markdown("5. **Part-of-Speech (POS) Tagging Page:** Tag words with their grammatical roles.")
    st.markdown("6. **Word Cloud Generation Page:** Generate a word cloud from the text.")
    st.markdown("7. **N-Grams Page:** Create uni-grams, bi-grams, or tri-grams from the text.")
    
    st.write(
        "Please select a task from the sidebar on the left to get started. You can choose the input type, "
        "input your text, and perform the desired NLP tasks. Enjoy exploring the capabilities of this AutoNLP app!"
    )

# Tokenization Page
elif page == "Tokenization":
    st.title("Tokenization Page")
    tokenization_type = st.radio("Choose tokenization type", ["Word Tokenization", "Sentence Tokenization"])
    input_type = st.radio("Choose input type", ["Text Input", "TXT File Upload"])
    
    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("Enter text:")
        if st.button("Perform Tokenization"):
            if len(word_tokenize(text_input)) > max_word_limit:
                st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
            else:
                tokens = tokenize_text(text_input, tokenization_type)
                st.subheader("Tokens:")
                st.write(tokens)
    
    elif input_type == "TXT File Upload":
        max_word_limit = 3000
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if st.button("Perform Tokenization"):
            if uploaded_file is not None:
                file_contents = uploaded_file.read()
                try:
                    file_contents = file_contents.decode("utf-8")
                    if len(word_tokenize(file_contents)) > max_word_limit:
                        st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
                    else:
                        tokens = tokenize_text(file_contents, tokenization_type)
                        st.subheader("Tokens:")
                        st.write(tokens)
                        
                except UnicodeDecodeError:
                    st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
            else:
                st.info("Please upload a .txt file.")

# Stopwords Removal Page
elif page == "Stopwords Removal":
    st.title("Stopwords Removal Page")
    tokenization_type = "Word Tokenization"
    input_type = st.radio("Choose input type", ["Text Input", "TXT File Upload"])
    
    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("Enter text:")
        if st.button("Perform Stopwords Removal"):
            if len(word_tokenize(text_input)) > max_word_limit:
                st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
            else:
                tokens = tokenize_text(text_input, tokenization_type)
                st.subheader("Tokens (Before Stopwords Removal):")
                st.write(tokens)
                
                # Remove stopwords
                filtered_tokens = remove_stopwords(tokens)
                st.subheader("Tokens (After Stopwords Removal):")
                st.write(filtered_tokens)
    
    elif input_type == "TXT File Upload":
        max_word_limit = 3000
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if st.button("Perform Stopwords Removal"):
            if uploaded_file is not None:
                file_contents = uploaded_file.read()
                try:
                    file_contents = file_contents.decode("utf-8")
                    if len(word_tokenize(file_contents)) > max_word_limit:
                        st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
                    else:
                        tokens = tokenize_text(file_contents, tokenization_type)
                        st.subheader("Tokens (Before Stopwords Removal):")
                        st.write(tokens)
                        
                        # Remove stopwords
                        filtered_tokens = remove_stopwords(tokens)
                        st.subheader("Tokens (After Stopwords Removal):")
                        st.write(filtered_tokens)
                    
            except UnicodeDecodeError:
                st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
        else:
            st.info("Please upload a .txt file.")

# Stemming Page
elif page == "Stemming":
    st.title("Stemming Page")
    tokenization_type = "Word Tokenization"
    input_type = st.radio("Choose input type", ["Text Input", "TXT File Upload"])
    
    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("Enter text:")
        if st.button("Perform Stemming"):
            if len(word_tokenize(text_input)) > max_word_limit:
                st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
            else:
                tokens = tokenize_text(text_input, tokenization_type)
                st.subheader("Tokens (Before Stemming):")
                st.write(tokens)
                
                # Perform stemming
                stemmed_tokens = perform_stemming(tokens)
                st.subheader("Tokens (After Stemming):")
                st.write(stemmed_tokens)
    
    elif input_type == "TXT File Upload":
        max_word_limit = 3000
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if st.button("Perform Stemming"):
            if uploaded_file is not None:
                file_contents = uploaded_file.read()
                try:
                    file_contents = file_contents.decode("utf-8")
                    if len(word_tokenize(file_contents)) > max_word_limit:
                        st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
                    else:
                        tokens = tokenize_text(file_contents, tokenization_type)
                        st.subheader("Tokens (Before Stemming):")
                        st.write(tokens)
                        
                        # Perform stemming
                        stemmed_tokens = perform_stemming(tokens)
                        st.subheader("Tokens (After Stemming):")
                        st.write(stemmed_tokens)
                    
            except UnicodeDecodeError:
                st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
        else:
            st.info("Please upload a .txt file.")

# Lemmatization Page
elif page == "Lemmatization":
    st.title("Lemmatization Page")
    tokenization_type = "Word Tokenization"
    input_type = st.radio("Choose input type", ["Text Input", "TXT File Upload"])
    
    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("Enter text:")
        if st.button("Perform Lemmatization):
            if len(word_tokenize(text_input)) > max_word_limit:
                st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
            else:
                tokens = tokenize_text(text_input, tokenization_type)
                st.subheader("Tokens (Before Lemmatization):")
                st.write(tokens)
                
                # Perform lemmatization
                lemmatized_tokens = perform_lemmatization(tokens)
                st.subheader("Tokens (After Lemmatization):")
                st.write(lemmatized_tokens)
    
    elif input_type == "TXT File Upload":
        max_word_limit = 3000
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if st.button("Perform Lemmatization):
            if uploaded_file is not None:
                file_contents = uploaded_file.read()
                try:
                    file_contents = file_contents.decode("utf-8")
                    if len(word_tokenize(file_contents)) > max_word_limit:
                        st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
                    else:
                        tokens = tokenize_text(file_contents, tokenization_type)
                        st.subheader("Tokens (Before Lemmatization):")
                        st.write(tokens)
                        
                        # Perform lemmatization
                        lemmatized_tokens = perform_lemmatization(tokens)
                        st.subheader("Tokens (After Lemmatization):")
                        st.write(lemmatized_tokens)
                    
            except UnicodeDecodeError:
                st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
        else:
            st.info("Please upload a .txt file.")

# POS Tagging Page
elif page == "POS Tagging":
    st.subheader("Part-of-Speech (POS) Tagging Page")
    tokenization_type = "Word Tokenization"
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
            
            # Perform POS tagging
            pos_tags = pos_tagging(tokens)
            pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
            st.write("POS Tags:")
            st.dataframe(pos_df)
    
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
                    
                    # Perform POS tagging
                    pos_tags = pos_tagging(tokens)
                    pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
                    st.write("POS Tags:")
                    st.dataframe(pos_df)
            except UnicodeDecodeError:
                st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
        else:
            st.info("Please upload a .txt file.")

# Word Cloud Page
elif page == "Word Cloud":
    st.subheader("Word Cloud Page")
    tokenization_type = "Word Tokenization"
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
            
            # Generate and display the word cloud
            if st.button("Generate Word Cloud"):
                generate_word_cloud(tokens)
    
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
                    
                    # Generate and display the word cloud
                    if st.button("Generate Word Cloud"):
                        generate_word_cloud(tokens)
            except UnicodeDecodeError:
                st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
        else:
            st.info("Please upload a .txt file.")

# N-Grams Page
elif page == "N-Grams":
    st.subheader("N-Grams Page")
    tokenization_type = "Word Tokenization"
    input_type = st.radio("Choose input type", ["Text Input", "TXT File Import"])
    n_gram_type = st.radio("Choose N-Gram Type", ["Uni-Grams (1-Grams)", "Bi-Grams (2-Grams)", "Tri-Grams (3-Grams)"])
    
    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("Enter text:")
        if len(word_tokenize(text_input)) > max_word_limit:
            st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
        else:
            tokens = tokenize_text(text_input, tokenization_type)
            st.write("Tokens:", tokens)
            
            if n_gram_type == "Uni-Grams (1-Grams)":
                n = 1
            elif n_gram_type == "Bi-Grams (2-Grams)":
                n = 2
            elif n_gram_type == "Tri-Grams (3-Grams)":
                n = 3
            
            n_grams = create_ngrams(tokens, n)
            n_grams_text = generate_ngrams_text(n_grams)
            
            st.write(f"{n}-Grams Text:")
            st.write(n_grams_text)
            
            # Download n-grams performed text as a txt file
            if st.button(f"Download {n}-Grams Text"):
                n_grams_content = "\n".join(n_grams_text)
                n_grams_file = BytesIO(n_grams_content.encode())
                st.download_button(
                    label=f"Download {n}-Grams Text",
                    data=n_grams_file,
                    key=f"{n}_grams_text.txt",
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
                    
                    if n_gram_type == "Uni-Grams (1-Grams)":
                        n = 1
                    elif n_gram_type == "Bi-Grams (2-Grams)":
                        n = 2
                    elif n_gram_type == "Tri-Grams (3-Grams)":
                        n = 3
                    
                    n_grams = create_ngrams(tokens, n)
                    n_grams_text = generate_ngrams_text(n_grams)
                    
                    st.write(f"{n}-Grams Text:")
                    st.write(n_grams_text)
                    
                    # Download n-grams performed text as a txt file
                    if st.button(f"Download {n}-Grams Text"):
                        n_grams_content = "\n".join(n_grams_text)
                        n_grams_file = BytesIO(n_grams_content.encode())
                        st.download_button(
                            label=f"Download {n}-Grams Text",
                            data=n_grams_file,
                            key=f"{n}_grams_text.txt",
                            on_click=None,
                        )
            except UnicodeDecodeError:
                st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
        else:
            st.info("Please upload a .txt file.")
