import time
import os
import tempfile
import spacy
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
from swarm import Swarm, Agent
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from .graph_llm import extract_relationships, graph_llm_agent, create_pyvis_graph, preprocess_graph_data


# Download NLTK data if not already present
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Set up Flask app
app = Flask(__name__)

# Set your OpenAI API key
api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key

# Initialize Swarm client
swarm_client = Swarm()

# Load spaCy models for each language
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")
nlp_es = spacy.load("es_core_news_sm")
nlp_de = spacy.load("de_core_news_sm")

# Load fastText language identification model
fasttext_model_path = "/Users/josemarchena/Python/rag5/lid.176.bin"
fasttext_model = fasttext.load_model(fasttext_model_path)

# Define the agents
english_agent = Agent(name="English Agent", instructions="Summarize the document content in English.")
spanish_agent = Agent(name="Spanish Agent", instructions="Summarize the document content in Spanish.")
french_agent = Agent(name="French Agent", instructions="Summarize the document content in French.")
german_agent = Agent(name="German Agent", instructions="Summarize the document content in German.")
legal_agent = Agent(name="Legal Agent", instructions="Focus on summarizing legal aspects of the document.")

# Define the Keyword Extraction Agent
keyword_agent = Agent(name="Keyword Agent", instructions="Extract key terms and phrases from the document.")

# Function to detect language and select appropriate agent
def detect_language_and_select_agent(text):
    try:
        # Predict language using fastText
        prediction = fasttext_model.predict(text, k=1)  # Get top prediction
        lang = prediction[0][0].replace("__label__", "")  # Extract language label
        confidence = prediction[1][0]  # Extract confidence score
        print(f"Detected language: {lang} with confidence {confidence}")

        # Check for legal keywords
        legal_keywords = ["clause", "statute", "regulation", "penalty", "compliance", "law", "rights", "obligation"]
        if any(keyword in text.lower() for keyword in legal_keywords):
            return legal_agent

        # Assign agent based on detected language
        if lang == 'es':
            return spanish_agent
        elif lang == 'fr':
            return french_agent
        elif lang == 'de':
            return german_agent
        else:
            return english_agent
    except Exception as e:
        print(f"Error detecting language or selecting agent: {e}")
        return english_agent

# Function to extract text from a single PDF file using PyPDFLoader
def extract_text_pypdf_loader(file):
    try:
        if hasattr(file, 'save'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                file.save(temp_file.name)
                temp_file_path = temp_file.name
        else:
            temp_file_path = file

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        text = "".join(page.page_content for page in pages if page.page_content)
        os.remove(temp_file_path)
        return text
    except Exception as e:
        print(f"Error reading PDF with PyPDFLoader: {e}")
        return None

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [WordNetLemmatizer().lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Function to chunk text into smaller pieces
def chunk_text(text, max_length=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # Account for spaces
        if current_length >= max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def is_legal_chunk(chunk):
    """
    Checks if a chunk contains legal-related keywords.
    """
    legal_keywords = ["clause", "article", "section", "regulation", "law", "statute", "penalty", "rights", "compliance"]
    return any(keyword in chunk.lower() for keyword in legal_keywords)

# Function to summarize each chunk of text
def summarize_chunk(agent, chunk):
    try:
        if agent.name == "French Agent":
            prompt = f"Faites un résumé du contenu en français:\n\n{chunk}"
        elif agent.name == "German Agent":
            prompt = f"Zusammenfassen Sie den Inhalt auf Deutsch:\n\n{chunk}"
        elif agent.name == "Legal Agent":
            prompt = f"Summarize with legal focus:\n\n{chunk}"
        else:
            prompt = f"Summarize the following: \n\n{chunk}"

        response = swarm_client.run(agent=agent, messages=[{"role": "system", "content": prompt}])
        summary = response.messages[-1]["content"]

        if agent.name != "English Agent":
            summary = swarm_client.run(
                agent=english_agent,
                messages=[{"role": "system", "content": f"Translate to English:\n\n{summary}"}]
            ).messages[-1]["content"]

        return summary
    except Exception as e:
        print(f"Error summarizing chunk: {e}")
        return "Error during summarization."

# Updated keyword extraction function
def extract_keywords(text, lang_code, max_count=10, max_phrase_length=3, min_frequency=2):
    """
    Extracts important keywords from the text using TF-IDF and prioritizes domain-specific legal keywords.
    """
    try:
        # Translate non-English text to English for consistency
        if lang_code != 'en':
            translated_text = swarm_client.run(
                agent=english_agent,
                messages=[{"role": "system", "content": f"Translate to English: {text}"}]
            ).messages[-1]["content"]
        else:
            translated_text = text

        # Vectorize text using TF-IDF
        vectorizer = TfidfVectorizer(max_features=max_count, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([translated_text])
        tfidf_keywords = vectorizer.get_feature_names_out()

        # Combine TF-IDF keywords with legal keywords
        legal_keywords = [
            "clause", "contract", "agreement", "statute", "compliance",
            "liability", "penalty", "rights", "obligation", "termination"
        ]
        keywords = [kw for kw in tfidf_keywords if len(kw.split()) <= max_phrase_length]
        prioritized_keywords = list(set(keywords) | set(legal_keywords))

        return ", ".join(prioritized_keywords[:max_count])
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return "No keywords available."

def extract_legal_clauses(text, lang_code, max_clauses=5):
    """
    Extracts legal clauses, dates, and parties involved from the text.

    Parameters:
        text (str): Input text for clause extraction.
        lang_code (str): Language code (e.g., 'en', 'fr').
        max_clauses (int): Maximum number of clauses to extract.

    Returns:
        str: A formatted string of extracted clauses, dates, and parties.
    """
    try:
        # Translate non-English text to English for consistency
        if lang_code != 'en':
            translated_text = swarm_client.run(
                agent=english_agent,
                messages=[{"role": "system", "content": f"Translate to English: {text}"}]
            ).messages[-1]["content"]
        else:
            translated_text = text

        # Use spaCy's NLP pipeline for named entity recognition
        doc = nlp_en(translated_text)

        # Extract legal clauses using pattern matching
        clauses = [
            sent.text for sent in doc.sents if any(
                kw in sent.text.lower() for kw in ["clause", "article", "section"]
            )
        ]

        # Extract dates
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

        # Extract parties (organizations and individuals)
        parties = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON"]]

        # Format the output
        return (
            f"Clauses: {', '.join(clauses[:max_clauses]) or 'None'}\n"
            f"Dates: {', '.join(dates) or 'None'}\n"
            f"Parties: {', '.join(parties) or 'None'}"
        )
    except Exception as e:
        print(f"Error extracting legal clauses: {e}")
        return "No legal clauses available."

# Function to analyze document
def analyze_document(file):
    extracted_text = extract_text_pypdf_loader(file)
    if not extracted_text:
        return (
            "Failed to extract text from the document.",
            "No keywords available.",
            "No legal clauses available.",
            set(),
            []
        )

    # Preprocess text for consistency
    cleaned_text = preprocess_text(extracted_text)

    # Detect language and select the appropriate agent
    lang_code = fasttext_model.predict(cleaned_text, k=1)[0][0].replace("__label__", "")
    agent = detect_language_and_select_agent(cleaned_text)
    print(f"Using agent for summarization: {agent.name}")

    # Extract relationships for graph creation
    nodes, edges = extract_relationships(graph_llm_agent, cleaned_text)

    # Chunk the text into smaller parts
    chunks = chunk_text(cleaned_text)
    print(f"Total chunks: {len(chunks)}")

    # Filter chunks for legal content
    legal_chunks = [chunk for chunk in chunks if is_legal_chunk(chunk)]
    print(f"Legal chunks identified: {len(legal_chunks)}")

    if not legal_chunks:
        return (
            "No relevant legal content found.",
            "No keywords available.",
            "No legal clauses available.",
            nodes,
            edges
        )

    # Summarize legal chunks
    summaries = [summarize_chunk(agent, chunk) for chunk in legal_chunks]
    combined_summary = "\n".join(summaries)

    # Extract categorized keywords
    categorized_keywords = extract_keywords(cleaned_text, lang_code)

    # Extract legal clauses, dates, and parties
    legal_clauses = extract_legal_clauses(cleaned_text, lang_code)

    return combined_summary, categorized_keywords, legal_clauses, nodes, edges

# Expose a reusable entry point for main_entry.py
def process_main_model(file):
    """
    Entry point for processing a PDF file.
    Wraps the `analyze_document` function to return a structured response.
    """
    summary, keywords, legal_clauses, nodes, edges = analyze_document(file)
    return {
        "summary": summary,
        "keywords": keywords,
        "legal_clauses": legal_clauses,
        "graph_data": {
            "nodes": nodes,
            "edges": edges,
        }
    }
