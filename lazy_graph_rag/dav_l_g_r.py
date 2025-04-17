import os
import tempfile
import spacy
import logging
import nltk
from langchain_community.document_loaders import PyPDFLoader
from swarm import Swarm, Agent
import fasttext
from sentence_transformers import SentenceTransformer
from .graph_processor import create_dynamic_graph, validate_nodes_and_edges, lazy_extract_relationships


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK data if not already present
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key


# Initialize Swarm client
swarm_client = Swarm()

# Load spaCy models and other language tools
nlp_en = spacy.load("en_core_web_sm")

# Load fastText language model
fasttext_model_path = "/Users/josemarchena/Python/rag5/lid.176.bin"
fasttext_model = fasttext.load_model(fasttext_model_path)

# Sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define agents
agents = {
    "en": Agent(name="English Agent", instructions="Summarize in English."),
    "es": Agent(name="Spanish Agent", instructions="Summarize in Spanish."),
    "fr": Agent(name="French Agent", instructions="Summarize in French."),
    "de": Agent(name="German Agent", instructions="Summarize in German."),
    "legal": Agent(name="Legal Agent", instructions="Focus on legal aspects."),
}

def detect_language_and_select_agent(text):
    """
    Detect language using fastText and select the corresponding agent.
    """
    try:
        lang = fasttext_model.predict(text, k=1)[0][0].replace("__label__", "")
        logger.info(f"Detected language: {lang}")
        return agents.get(lang, agents["en"])
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return agents["en"]

def extract_text_pypdf_loader(file):
    """
    Extract text from a PDF file using PyPDFLoader.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        # Check if the file is empty
        if os.path.getsize(temp_file_path) == 0:
            raise ValueError("The uploaded file is empty.")

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        text = "".join(page.page_content for page in pages if page.page_content)
        
        if not text.strip():  # Check if text extraction is empty
            raise ValueError("No text content could be extracted from the PDF.")
        
        os.remove(temp_file_path)
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return None


def chunk_text(text, max_length=1000):
    """
    Split text into smaller chunks for processing.
    """
    words = text.split()
    chunks, current_chunk, current_length = [], [], 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def analyze_document(file):
    """
    Analyze a single PDF document and generate graphs.
    """
    extracted_text = extract_text_pypdf_loader(file)
    if not extracted_text:
        logger.error("The file is empty or no text could be extracted.")
        return {"error": "The uploaded file is empty or could not be processed."}

    # Detect language and process relationships
    agent = detect_language_and_select_agent(extracted_text)
    chunks = chunk_text(extracted_text)
    combined_summary = " ".join(chunks)

    try:
        # Extract relationships
        nodes, edges = lazy_extract_relationships(combined_summary, nlp_en, query="legal")
        valid_nodes, valid_edges = validate_nodes_and_edges(nodes, edges)

        # Generate graph
        graph_path = create_dynamic_graph(valid_nodes, valid_edges, output_file="static/lazy_graph_rag.html")
        if not graph_path:
            raise Exception("Failed to generate dynamic graph.")

        return {
            "summary": combined_summary,
            "graph_path": "static/lazy_graph_rag.html",
        }
    except Exception as e:
        logger.error(f"Error analyzing document: {e}", exc_info=True)
        return {"error": f"Analysis failed: {str(e)}"}

def process_secondary_model(file):
    """
    Entry point for processing a PDF file in the lazy graph RAG model.
    """
    return analyze_document(file)