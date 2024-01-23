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
from nltk import FreqDist
import docx2txt
from langdetect import detect
import base64
import io
from io import BytesIO
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up Streamlit app
st.set_page_config(page_title="AutoNLP Application", page_icon="📚", layout="wide")

st.title("AutoNLP Streamlit Web App")

if st.button("Return to AIHub"):
    st.markdown("https://sites.google.com/view/aihub-1?usp=sharing", unsafe_allow_html=True)
    
page = st.sidebar.radio("**Select a Page**", ["Home Page", "Tokenization", "Stopwords Removal", "Stemming", "Lemmatization", "POS Tagging", "Word Cloud", "N-Grams", "Keyword Extraction", "Text Similarity", "About"])

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
    if len(text) == 0:
        st.warning("Cannot generate a word cloud from empty text.")

    else:
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

# Function to extract keywords
@st.cache_resource
def extract_keywords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    
    # Calculate word frequency
    word_freq = FreqDist(filtered_words)

    # Create a DataFrame with keywords and frequencies
    keywords_df = pd.DataFrame(word_freq.items(), columns=['Keyword', 'Frequency'])
    keywords_df = keywords_df.sort_values(by='Frequency', ascending=False)
    
    # Display keywords and their frequencies
    st.subheader("Keywords and Their Frequencies (Dataframe):")
    st.dataframe(keywords_df)

    csv = keywords_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
    href = f'data:file/csv;base64,{b64}'
    st.markdown(f'<a href="{href}" download="keyword_extraction_content.csv">Click here to download the document with Keywords and Their Frequencies</a>', unsafe_allow_html=True)
    
    # Plot keyword frequency distribution
    st.subheader("Keywords and Their Frequencies (Visualization Plot):")
    plt.figure(figsize=(10, 5))
    word_freq.plot(20, cumulative=False)
    st.pyplot(plt)

# Function to calculate text similarity
@st.cache_resource
def calculate_similarity(text1, text2):
    # Tokenize the input texts
    tokens = word_tokenize(text1 + " " + text2)
    
    # Create TF-IDF vectors for the texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1]).flatten()[0]
    
    return similarity_score

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
    st.markdown("7. **N-Grams Page:** Create uni-grams, bi-grams, or tri-grams from the text. And, you also have the option to copy the processed content to clipboard.")
    st.markdown("8. **Keyword Extraction Page:** Extract keywords along with frequencies.")
    st.markdown("9. **Text Similarity Page:** Finds the cosine similarity between two text inputs.")
    
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
        if st.button("Perform Lemmatization"):
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
        if st.button("Perform Lemmatization"):
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
    st.title("Part-of-Speech (POS) Tagging Page")
    tokenization_type = "Word Tokenization"
    input_type = st.radio("Choose input type", ["Text Input", "TXT File Upload"])
    
    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("Enter text:")
        if st.button("Perform POS Tagging"):
            if len(word_tokenize(text_input)) > max_word_limit:
                st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
            else:
                tokens = tokenize_text(text_input, tokenization_type)
                st.subheader("Tokens:")
                st.write(tokens)
                
                # Perform POS tagging
                pos_tags = pos_tagging(tokens)
                pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
                st.subheader("POS Tags:")
                st.dataframe(pos_df)

                # Download the dataset using base64 encoding
                csv =pos_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
                href = f'data:file/csv;base64,{b64}'
                st.markdown(f'<a href="{href}" download="pos_tagged_content.csv">Click here to download POS Tagged document</a>', unsafe_allow_html=True)
        
    elif input_type == "TXT File Upload":
        max_word_limit = 3000
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if st.button("Perform POS Tagging"):
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
                        
                        # Perform POS tagging
                        pos_tags = pos_tagging(tokens)
                        pos_df = pd.DataFrame(pos_tags, columns=["Word", "POS Tag"])
                        st.subheader("POS Tags:")
                        st.dataframe(pos_df)

                        # Download the dataset using base64 encoding
                        csv =pos_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
                        href = f'data:file/csv;base64,{b64}'
                        st.markdown(f'<a href="{href}" download="pos_tagged_content.csv">Click here to download POS Tagged Document</a>', unsafe_allow_html=True)
                        
                except UnicodeDecodeError:
                    st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
            else:
                st.info("Please upload a .txt file.")

# Word Cloud Page
elif page == "Word Cloud":
    st.title("Word Cloud Page")
    tokenization_type = "Word Tokenization"
    input_type = st.radio("Choose input type", ["Text Input", "TXT File Upload"])
    
    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("Enter text:")
        if st.button("Generate Word Cloud"):
            if len(word_tokenize(text_input)) > max_word_limit:
                st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
            else:
                tokens = tokenize_text(text_input, tokenization_type)
                st.subheader("Tokens:")
                st.write(tokens)
                
                # Generate and display the word cloud
                st.subheader("Word Cloud:")
                generate_word_cloud(tokens)
    
    elif input_type == "TXT File Upload":
        max_word_limit = 3000
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if st.button("Generate Word Cloud"):
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
                        
                        # Generate and display the word cloud
                        st.subheader("Word Cloud:")
                        generate_word_cloud(tokens)
                           
                except UnicodeDecodeError:
                    st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
            else:
                st.info("Please upload a .txt file.")

# N-Grams Page
elif page == "N-Grams":
    st.title("N-Grams Page")
    tokenization_type = "Word Tokenization"
    input_type = st.radio("Choose input type", ["Text Input", "TXT File Upload"])
    n_gram_type = st.radio("Choose N-Gram Type", ["Uni-Grams (1-Grams)", "Bi-Grams (2-Grams)", "Tri-Grams (3-Grams)"])
    
    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("Enter text:")
        if st.button("Generate N-Grams"):
            if len(word_tokenize(text_input)) > max_word_limit:
                st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
            else:
                tokens = tokenize_text(text_input, tokenization_type)
                st.subheader("Tokens:")
                st.write(tokens)
                
                if n_gram_type == "Uni-Grams (1-Grams)":
                    n = 1
                elif n_gram_type == "Bi-Grams (2-Grams)":
                    n = 2
                elif n_gram_type == "Tri-Grams (3-Grams)":
                    n = 3
                
                n_grams = create_ngrams(tokens, n)
                n_grams_text = generate_ngrams_text(n_grams)
                
                st.subheader(f"{n}-Grams Text:")
                st.write(n_grams_text)
                
    elif input_type == "TXT File Upload":
        max_word_limit = 3000
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
        if st.button("Generate N-Grams"):
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
                        
                        if n_gram_type == "Uni-Grams (1-Grams)":
                            n = 1
                        elif n_gram_type == "Bi-Grams (2-Grams)":
                            n = 2
                        elif n_gram_type == "Tri-Grams (3-Grams)":
                            n = 3
                        
                        n_grams = create_ngrams(tokens, n)
                        n_grams_text = generate_ngrams_text(n_grams)
                        
                        st.subheader(f"{n}-Grams Text:")
                        st.write(n_grams_text)
                        
                except UnicodeDecodeError:
                    st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
            else:
                st.info("Please upload a .txt file.")

# Keyword Extraction Page
elif page == "Keyword Extraction":
    st.title("Keyword Extraction Page")
    input_type = st.radio("Choose input type", ["Text Input", "TXT File Import"])
    
    if input_type == "Text Input":
        max_word_limit = 300
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        text_input = st.text_area("Enter text:")
        
        if st.button("Extract Keywords"):
            # Check for empty input text
            if not text_input.strip():
                st.error("Input text is empty. Please provide text for keyword extraction.")
            # Check for word limit in text input
            elif len(word_tokenize(text_input)) > max_word_limit:
                st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
            else:
                extract_keywords(text_input)
    
    elif input_type == "TXT File Import":
        max_word_limit = 2000
        st.write(f"Maximum Word Limit: {max_word_limit} words")
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

        if st.button("Extract Keywords"):
            if uploaded_file is not None:
                file_contents = uploaded_file.read()
                try:
                    file_contents = file_contents.decode("utf-8")
                    # Check for word limit in uploaded file
                    if len(word_tokenize(file_contents)) > max_word_limit:
                        st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
                    else:
                        extract_keywords(file_contents)
                except UnicodeDecodeError:
                    st.error("Invalid input: The uploaded file contains non-text data or is not in UTF-8 format.")
            else:
                st.info("Please upload a .txt file.")

# Text Similarity Page
elif page == "Text Similarity":
    st.title("Text Similarity Page")
    max_word_limit = 300
    st.write(f"Maximum Word Limit: {max_word_limit} words")
    text1 = st.text_area("Enter Text 1:")
    text2 = st.text_area("Enter Text 2:")

    if st.button("Find Text Similarity"):
        # Check for empty input texts
        if not text1.strip() or not text2.strip():
            st.error("Please provide both texts for similarity comparison.")
        elif len(word_tokenize(text1)) > max_word_limit or len(word_tokenize(text2)) > max_word_limit:
            st.error(f"Word count exceeds the maximum limit of {max_word_limit} words.")
        else:
            similarity_score = calculate_similarity(text1, text2)
            
            # Display similarity score
            st.subheader("Similarity Score:")
            st.write(f"The cosine similarity between the two texts is: {similarity_score:.2f}")

# About Page
elif page == "About":
    st.title("🚀 About the AutoNLP Web App")
    
    st.markdown("""
    Welcome to the AutoNLP web app, a versatile tool designed to streamline Natural Language Processing (NLP) tasks and make text analysis easier than ever! 🎉

    This app empowers you to perform various NLP tasks, including tokenization, stopwords removal, stemming, lemmatization, Part-of-Speech (POS) tagging, word cloud generation, N-Grams analysis, keyword extraction and text similarity. Whether you're a data scientist, developer, or language enthusiast, AutoNLP is here to simplify and enhance your text processing.

    It's built using Streamlit and Python, offering a user-friendly interface to explore the fascinating world of NLP. Feel free to experiment, analyze, and visualize text data effortlessly. And, it's brought to you by Team AI Hub.

    Ready to embark on your NLP journey? Let's start exploring the world of natural language processing using different inputs! 💡
    """)

    st.markdown("""
    *_Regards,_*
    
    *_Team AI Hub_*
    """)
