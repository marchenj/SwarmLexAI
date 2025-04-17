from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dynamic_graph(nodes, edges, output_file="static/lazy_graph_rag.html"):
    """
    Create an interactive graph visualization using PyVis.

    Parameters:
        nodes (list): List of nodes.
        edges (list): List of edges.
        output_file (str): Path to save the graph HTML.

    Returns:
        str: Path to the saved HTML file.
    """
    try:
        nodes = list(set(nodes))  # Remove duplicate nodes
        net = Network(height="800px", width="100%", directed=False)

        # Use force-based layout for better visualization
        net.force_atlas_2based(
            gravity=-30,
            central_gravity=0.02,
            spring_length=150,
            spring_strength=0.01,
            damping=0.85,
        )

        # Add nodes
        for node in nodes:
            net.add_node(
                n_id=node,
                label=node,
                color="#3498DB",  # Default blue
                size=25,
            )

        # Add edges
        for source, target in edges:
            if source in nodes and target in nodes:
                net.add_edge(
                    source=source,
                    to=target,
                    color="gray",
                    width=1.5,
                )

        # Save and return the interactive graph
        net.save_graph(output_file)
        logger.info(f"Dynamic graph saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error generating dynamic graph: {e}")
        return None


def extract_keywords(text, nlp):
    """
    Extract meaningful keywords from text for graph nodes, avoiding prepositional phrases.

    Parameters:
        text (str): The text to analyze.
        nlp: The NLP pipeline (spaCy model).

    Returns:
        list: A list of filtered, meaningful keywords with context.
    """
    try:
        doc = nlp(text)
        keywords = set()

        # Add named entities with context
        for ent in doc.ents:
            if ent.label_ in {"LAW", "ORG", "GPE", "DATE"}:  # Filter relevant entities
                keywords.add(f"{ent.text.strip()} ({ent.label_})")

        # Add meaningful noun chunks without prepositions
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 2 and chunk.root.pos_ != "ADP" and chunk.text.lower() not in {"this", "that", "it"}:
                keywords.add(chunk.text.strip())

        # Add descriptive tokens, avoiding isolated prepositions
        for token in doc:
            if (
                token.is_alpha
                and not token.is_stop
                and len(token.text) > 2
                and token.pos_ not in {"ADP", "DET"}
            ):
                keywords.add(token.text.lower())

        return sorted(keywords)  # Return sorted keywords with context
    except Exception as e:
        logger.error(f"Error in extract_keywords: {e}")
        return []


def is_related(node1, node2, model, threshold=0.75):
    """
    Determine if two nodes are related based on semantic similarity.

    Parameters:
        node1 (str): The first node.
        node2 (str): The second node.
        model: The SentenceTransformer model for embedding calculation.
        threshold (float): The similarity threshold.

    Returns:
        bool: True if the nodes are related, False otherwise.
    """
    try:
        embeddings = model.encode([node1, node2])
        similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        logger.debug(f"Similarity: {node1} -> {node2} = {similarity_score}")
        return similarity_score > threshold
    except Exception as e:
        logger.error(f"Error in is_related: {e}")
        return False


def lazy_extract_relationships(summary, nlp, query=None, model=None):
    nodes = []
    edges = []

    try:
        # Initialize the model if not provided
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        # Extract nodes
        if query == "legal":
            nodes = [
                node for node in extract_keywords(summary, nlp)
                if any(term in node.lower() for term in ["clause", "law", "policy", "contract"])
            ]
        else:
            nodes = extract_keywords(summary, nlp)

        # Deduplicate nodes
        nodes = list(set(nodes))

        # Add sentence-level co-occurrence edges
        for sent in nlp(summary).sents:
            sentence_nodes = [node for node in nodes if node in sent.text]
            for i, node1 in enumerate(sentence_nodes):
                for node2 in sentence_nodes[i + 1:]:
                    edges.append((node1, node2))

        # Add semantic similarity edges
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j and is_related(node1, node2, model):
                    edges.append((node1, node2))

        # Deduplicate edges
        edges = list(set(edges))  # Ensure all edges are unique

        logger.info(f"Extracted nodes: {nodes}")
        logger.info(f"Generated edges: {edges}")
    except Exception as e:
        logger.error(f"Error in lazy_extract_relationships: {e}")

    return nodes, edges


def validate_nodes_and_edges(nodes, edges):
    """
    Validate and clean nodes and edges for graph representation.

    Parameters:
        nodes (list): List of nodes.
        edges (list): List of edges.

    Returns:
        tuple: Valid nodes and edges.
    """
    # Deduplicate and clean nodes
    valid_nodes = {node.strip() for node in nodes if node.strip()}  # Use set for deduplication

    # Deduplicate and clean edges
    valid_edges = [
        (source.strip(), target.strip())
        for source, target in set(edges)  # Deduplicate edges
        if source.strip() in valid_nodes and target.strip() in valid_nodes
    ]

    return valid_nodes, valid_edges
