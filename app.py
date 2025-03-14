from flask import Flask, request, render_template_string
import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

app = Flask(__name__)

### CONFIGURATION ###
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "argentic"  # update with your actual password

# ----------------------------
# 1. Connect to Neo4j and Retrieve Nodes
# ----------------------------
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("RETURN 1")
    print("Connected to Neo4j successfully.")
except ServiceUnavailable as e:
    print("Neo4j connection error:", e)
    exit(1)

def get_nodes(label):
    with driver.session() as session:
        result = session.run(f"MATCH (n:{label}) RETURN n")
        nodes = [record["n"] for record in result]
    return nodes

# Retrieve nodes from the graph.
cities = get_nodes("City")
flights = get_nodes("Flight")
hotels = get_nodes("Hotel")
restaurants = get_nodes("Restaurant")
preferences = get_nodes("Preference")
users = get_nodes("User")

# ----------------------------
# 2. Create Clean Text Representations for Each Node
# ----------------------------
def build_representation(props, fields):
    parts = []
    for field, label in fields.items():
        value = props.get(field)
        if value is not None and str(value).strip() != "":
            parts.append(f"{label}: {value}")
    return "; ".join(parts)

def represent_city(node):
    fields = {
        "City": "City",
        "Country": "Country",
        "weather": "Weather",
        "avg_flight_cost": "Avg Flight Cost",
        "avg_hotel_cost": "Avg Hotel Cost"
    }
    return build_representation(node._properties, fields)

def represent_flight(node):
    fields = {
        "Flight": "Flight",
        "Departure": "Departure",
        "Arrival": "Arrival",
        "avg_cost": "Avg Cost"
    }
    return build_representation(node._properties, fields)

def represent_hotel(node):
    fields = {
        "Hotel": "Hotel",
        "City": "City",
        "Country": "Country",
        "avg_cost": "Avg Cost"
    }
    return build_representation(node._properties, fields)

def represent_restaurant(node):
    fields = {
        "Restaurant": "Restaurant",
        "City": "City",
        "Country": "Country",
        "rating": "Rating"
    }
    return build_representation(node._properties, fields)

def represent_preference(node):
    fields = {
        "user_id": "User ID",
        "preferences": "Preferences"
    }
    return build_representation(node._properties, fields)

def represent_user(node):
    fields = {
        "user_id": "User ID",
        "name": "Name"
    }
    return build_representation(node._properties, fields)

def get_clean_representations(nodes, represent_func):
    reps = []
    for node in nodes:
        rep = represent_func(node)
        if rep and "Unknown" not in rep:
            reps.append(rep)
    return reps

representations = []
representations += get_clean_representations(cities, represent_city)
representations += get_clean_representations(flights, represent_flight)
representations += get_clean_representations(hotels, represent_hotel)
representations += get_clean_representations(restaurants, represent_restaurant)
representations += get_clean_representations(preferences, represent_preference)
representations += get_clean_representations(users, represent_user)

if not representations:
    for node in cities:
        representations.append(represent_city(node))
    for node in flights:
        representations.append(represent_flight(node))
    for node in hotels:
        representations.append(represent_hotel(node))
    for node in restaurants:
        representations.append(represent_restaurant(node))
    for node in preferences:
        representations.append(represent_preference(node))
    for node in users:
        representations.append(represent_user(node))

representations = list(set(representations))
print(f"Total representations for retrieval: {len(representations)}")

# ----------------------------
# 3. Compute Embeddings for All Representations
# ----------------------------
print("Computing embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(representations, convert_to_tensor=True)

# ----------------------------
# 4. Define a Retrieval Function
# ----------------------------
def retrieve_documents(query, top_k=3):
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    cos_scores = cosine_similarity(query_embedding.cpu().numpy(), doc_embeddings.cpu().numpy())[0]
    top_indices = np.argsort(cos_scores)[::-1][:top_k]
    retrieved_docs = [representations[i] for i in top_indices]
    return retrieved_docs

# ----------------------------
# 5. LLM for Final Answer Generation
# ----------------------------
generator = pipeline("text-generation", model="gpt2", max_length=150)

def generate_final_answer(query):
    retrieved_docs = retrieve_documents(query, top_k=3)
    prompt = "Travel Data:\n" + "\n".join(retrieved_docs) + "\n\n"
    prompt += f"Question: {query}\nAnswer:"
    result = generator(prompt, max_length=150, num_return_sequences=1)
    generated_text = result[0]["generated_text"]
    answer = generated_text.replace(prompt, "").strip().split("\n")[0].strip()
    return answer

# ----------------------------
# Flask Web Interface
# ----------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Travel Recommendation Engine</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        form { margin-bottom: 20px; }
        input[type=text] { width: 80%; padding: 10px; font-size: 16px; }
        input[type=submit] { padding: 10px 20px; font-size: 16px; }
        .answer { padding: 10px; background-color: #f0f0f0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Travel Recommendation Engine</h1>
    <form method="post">
        <input type="text" name="question" placeholder="Enter your travel query here" required>
        <input type="submit" value="Submit">
    </form>
    {% if answer %}
    <h2>Answer:</h2>
    <div class="answer">{{ answer }}</div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        question = request.form["question"]
        answer = generate_final_answer(question)
    return render_template_string(HTML_TEMPLATE, answer=answer)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

