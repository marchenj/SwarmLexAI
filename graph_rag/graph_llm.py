import os
import networkx as nx
from swarm import Swarm, Agent
from pyvis.network import Network
from sklearn.cluster import KMeans
import random
from spacy.tokens import Span
import spacy
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Graph LLM Agent
graph_llm_agent = Agent(
    name="Graph LLM Agent",
    instructions="Extract and map relationships between entities, clauses, dates, and keywords."
)

# Load spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")


def extract_relationships(agent, text):
    """
    Extract relationships using Graph LLM and advanced NLP techniques.

    Parameters:
        agent (Agent): The Graph LLM agent to use for extracting relationships.
        text (str): The input text to analyze for relationships.

    Returns:
        tuple: Nodes and edges representing relationships.
    """
    try:
        doc = nlp(text)
        relationships = []

        for sent in doc.sents:
            sent_relationships = parse_relationships(sent)
            relationships.extend(sent_relationships)

        refined_relationships = refine_relationships_with_agent(agent, text, relationships)
        if not refined_relationships:
            logging.warning("No refined relationships returned from the agent.")

        nodes, edges = parse_relationships_from_text(refined_relationships)
        return nodes, edges
    except Exception as e:
        print(f"Error extracting relationships: {e}")
        return set(), []


def parse_relationships(sent):
    """
    Parse relationships from a single sentence using dependency parsing.

    Parameters:
        sent (Span): A spaCy Span object representing a sentence.

    Returns:
        list: A list of relationships in the form of tuples (entity1, entity2, relationship).
    """
    relationships = []

    for token in sent:
        if token.dep_ in ("nsubj", "dobj", "pobj", "prep"):
            subj = [child.text for child in token.head.children if child.dep_ == "nsubj"]
            obj = [child.text for child in token.children if child.dep_ in ("dobj", "pobj")]
            if subj and obj:
                relationships.append((subj[0], obj[0], token.head.text))

            if token.dep_ == "prep" and token.head.pos_ == "VERB":
                prep_object = [child.text for child in token.children if child.dep_ == "pobj"]
                if prep_object:
                    relationships.append((token.head.text, prep_object[0], f"prep_{token.text}"))

    return relationships


def refine_relationships_with_agent(agent, text, relationships):
    """
    Use a Graph LLM agent to refine and expand relationships.

    Parameters:
        agent (Agent): The Graph LLM agent.
        text (str): Original document text.
        relationships (list): Initial relationships extracted via NLP.

    Returns:
        str: Refined relationships as text.
    """
    try:
        prompt = (
            f"Here are initial relationships:\n\n"
            f"{relationships}\n\n"
            f"Refine these relationships and provide additional ones strictly in this format:\n"
            f"Entity1 -> Entity2: Relationship Type (e.g., causal, associative).\n"
            f"Do not include any other text or explanations.\n\n"
            f"Original text: {text}"
        )

        response = Swarm().run(agent=agent, messages=[{"role": "system", "content": prompt}])
        refined_relationships = response.messages[-1]["content"]

        filtered_relationships = "\n".join(
            line.strip() for line in refined_relationships.splitlines()
            if "->" in line and ":" in line
        )
        return filtered_relationships
    except Exception as e:
        print(f"Error refining relationships with agent: {e}")
        return "\n".join(relationships) 


def parse_relationships_from_text(relationships_text):
    """
    Parse refined relationships into nodes and edges.

    Parameters:
        relationships_text (str): Refined relationships in text format.

    Returns:
        tuple: Nodes and edges.
    """
    nodes = set()
    edges = []
    try:
        for line in relationships_text.splitlines():
            line = line.strip()
            if not line:  
                continue

            if "->" in line and ":" in line:
                try:
                    entity1, rest = line.split("->")
                    entity2, rel_type = rest.split(":")
                    entity1, entity2, rel_type = entity1.strip(), entity2.strip(), rel_type.strip()
                    nodes.add(entity1)
                    nodes.add(entity2)
                    edges.append((entity1, entity2, rel_type))
                except ValueError as e:
                    print(f"Error processing line '{line}': {e}")
            else:
                print(f"Skipping unexpected relationship format: {line}")
    except Exception as e:
        print(f"Error parsing relationships from text: {e}")

    return nodes, edges

###
def validate_nodes_and_edges(nodes, edges):
    """
    Validate and clean nodes and edges for graph representation.

    Parameters:
        nodes (set): A set of nodes.
        edges (list): A list of edges (source, target).

    Returns:
        tuple: Validated nodes and edges.
    """
    valid_nodes = {node.strip() for node in nodes if node.strip()}
    valid_edges = list({
        (source.strip(), target.strip(), rel_type.strip())
        for source, target, rel_type in edges
        if source.strip() in valid_nodes and target.strip() in valid_nodes
    })
    return valid_nodes, valid_edges


def assign_clusters(nodes, edges, n_clusters=4):
    """
    Assign clusters to nodes dynamically based on relationships.

    Parameters:
        nodes (set): A set of nodes.
        edges (list): A list of edges (source, target, rel_type).
        n_clusters (int): Desired number of clusters.

    Returns:
        dict: Node-to-cluster mapping and cluster attributes.
    """
    node_list = list(nodes)
    num_nodes = len(node_list)

    if num_nodes < n_clusters:
        logging.warning(f"Reducing number of clusters from {n_clusters} to {num_nodes} due to insufficient nodes.")
        n_clusters = num_nodes

    edge_matrix = [
        [1 if (source, target) in [(e[0], e[1]) for e in edges] or (target, source) in [(e[0], e[1]) for e in edges]
         else 0 for target in node_list] for source in node_list
    ]

    try:
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering_model.fit_predict(edge_matrix)
    except ValueError as e:
        print(f"Error during clustering: {e}. Nodes may lack sufficient connections.")
        return {}, {}

    cluster_colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A8", "#FFD700"]
    cluster_colors = cluster_colors[:n_clusters]

    node_to_cluster = {node_list[i]: cluster_labels[i] for i in range(num_nodes)}
    cluster_attributes = {
        cluster: {"color": cluster_colors[cluster % len(cluster_colors)], "size": random.randint(20, 40)}
        for cluster in range(n_clusters)
    }

    logging.info(f"Clustering completed with {n_clusters} clusters.")
    return node_to_cluster, cluster_attributes


def preprocess_graph_data(cleaned_text, nodes, edges, n_clusters=4):
    """
    Dynamically assign attributes to nodes and edges based on clustering.
    Also includes logic for multi-hop relationship expansion.

    Parameters:
        cleaned_text (str): Cleaned text.
        nodes (set): A set of nodes.
        edges (list): A list of edges (source, target, rel_type).
        n_clusters (int): Number of clusters.

    Returns:
        tuple: Nodes, edges, node attributes, edge attributes.
    """
    valid_nodes, valid_edges = validate_nodes_and_edges(nodes, edges)

    multi_hop_edges = []
    for source, intermediate, rel_type1 in valid_edges:
        for inter_target, target, rel_type2 in valid_edges:
            if intermediate == inter_target:
                new_edge = (source, target, f"{rel_type1}-{rel_type2}")
                if new_edge not in valid_edges and new_edge not in multi_hop_edges:
                    multi_hop_edges.append(new_edge)

    valid_edges.extend(multi_hop_edges)

    node_to_cluster, cluster_attributes = assign_clusters(valid_nodes, valid_edges, n_clusters)

    node_attributes = {
        node: {
            "color": cluster_attributes[node_to_cluster[node]]["color"],
            "size": cluster_attributes[node_to_cluster[node]]["size"],
            "type": f"Cluster {node_to_cluster[node]}"
        }
        for node in valid_nodes
    }

    edge_attributes = {
        (source, target): {
            "color": node_attributes[source]["color"],
            "label": rel_type,
            "weight": 3 if node_to_cluster[source] == node_to_cluster[target] else 1
        }
        for source, target, rel_type in valid_edges
    }

    for source, target, rel_type in multi_hop_edges:
        edge_attributes[(source, target)] = {
            "color": "#FFC300",
            "label": f"Multi-hop: {rel_type}",
            "weight": 2
        }

    return valid_nodes, valid_edges, node_attributes, edge_attributes


def create_pyvis_graph(nodes, edges, output_file="graph.html", node_attributes=None, edge_attributes=None):
    """
    Generate a cluster-based interactive graph using pyvis.

    Parameters:
        nodes (set): A set of nodes.
        edges (list): A list of edges (source, target, rel_type).
        output_file (str): The file name for saving the graph.
        node_attributes (dict): Attributes of nodes.
        edge_attributes (dict): Attributes of edges.

    Returns:
        str: Path to the generated graph file.
    """
    if not nodes or not edges:
        print("No valid nodes or edges for PyVis graph.")
        return None

    net = Network(height="800px", width="100%", directed=True)

    for node in nodes:
        try:
            net.add_node(
                n_id=node,
                label=node,
                color=node_attributes[node]["color"],
                size=node_attributes[node]["size"],
                title=f"Type: {node_attributes[node]['type']}"
            )
        except KeyError as e:
            print(f"Error adding node {node}: Missing attribute {e}")

    for edge in edges:
        try:
            source, target, *rel_type = edge
            edge_key = (source, target)
            attributes = edge_attributes.get(edge_key, {"color": "#808080", "label": "No label", "weight": 1})
            net.add_edge(
                source=source,
                to=target,
                color=attributes["color"],
                title=attributes["label"],
                value=attributes["weight"]
            )
        except Exception as e:
            print(f"Error adding edge {edge}: {e}")

    try:
        net.show_buttons(filter_=['physics', 'interaction'])
        net.save_graph(output_file)
        print(f"Interactive graph saved as {output_file}")
        return output_file
    except Exception as e:
        print(f"Error creating PyVis graph: {e}")
        return None

 