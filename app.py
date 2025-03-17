import ast  # to parse Python-like list strings
from flask import Flask, request, render_template_string
import pandas as pd
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
NEO4J_PASSWORD = "argentic"  # Update with your actual password

# CSV file paths
CITIES_CSV      = "adjusted_datasets/adjusted_cities.csv"
FLIGHTS_CSV     = "adjusted_datasets/adjusted_flights.csv"
HOTELS_CSV      = "adjusted_datasets/adjusted_hotels.csv"
RESTAURANTS_CSV = "adjusted_datasets/adjusted_restaurants.csv"
PREFERENCES_CSV = "adjusted_datasets/preferences.csv"
USERS_CSV       = "adjusted_datasets/users.csv"
PASSPORTS_CSV   = "adjusted_datasets/adjusted_passports.csv"
HISTORIES_CSV   = "adjusted_datasets/histories.csv"

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

# ----------------------------
# Node creation (from previous steps)
# ----------------------------
def create_city_node(tx, props):
    query = """
    MERGE (c:City {city_id: $city_id})
    SET c += $props
    """
    tx.run(query, city_id=props["city_id"], props=props)

def create_flight_node(tx, props):
    query = """
    MERGE (f:Flight {flight_id: $flight_id})
    SET f += $props
    """
    tx.run(query, flight_id=props["flight_id"], props=props)

def create_hotel_node(tx, props):
    query = """
    MERGE (h:Hotel {hotel_id: $hotel_id})
    SET h += $props
    """
    tx.run(query, hotel_id=props["hotel_id"], props=props)

def create_restaurant_node(tx, props):
    query = """
    MERGE (r:Restaurant {restaurant_id: $restaurant_id})
    SET r += $props
    """
    tx.run(query, restaurant_id=props["restaurant_id"], props=props)

def create_preference_node(tx, props):
    query = """
    MERGE (p:Preference {preference_id: $preference_id})
    SET p += $props
    """
    tx.run(query, preference_id=props["preference_id"], props=props)

def create_user_node(tx, props):
    user_id = props.get("User_ID") or props.get("user_id")
    if not user_id:
        raise KeyError("User_ID not found in props for user node")
    query = """
    MERGE (u:User {User_ID: $user_id})
    SET u += $props
    """
    tx.run(query, user_id=user_id, props=props)

def create_passport_node(tx, props):
    query = """
    MERGE (pp:Passport {passport_id: $passport_id})
    SET pp += $props
    """
    tx.run(query, passport_id=props["passport_id"], props=props)

def create_history_node(tx, props):
    query = """
    MERGE (h:History {history_id: $history_id})
    SET h += $props
    """
    tx.run(query, history_id=props["history_id"], props=props)

def create_relationship(tx, label_from, key_from, value_from, 
                        rel_type, label_to, key_to, value_to):
    query = f"""
    MATCH (a:{label_from} {{{key_from}: $value_from}})
    MATCH (b:{label_to} {{{key_to}: $value_to}})
    MERGE (a)-[r:{rel_type}]->(b)
    """
    tx.run(query, value_from=value_from, value_to=value_to)

def build_graph():
    with driver.session() as session:
        cities_df      = pd.read_csv(CITIES_CSV)
        flights_df     = pd.read_csv(FLIGHTS_CSV)
        hotels_df      = pd.read_csv(HOTELS_CSV)
        restaurants_df = pd.read_csv(RESTAURANTS_CSV)
        preferences_df = pd.read_csv(PREFERENCES_CSV)
        users_df       = pd.read_csv(USERS_CSV)
        passports_df   = pd.read_csv(PASSPORTS_CSV)
        histories_df   = pd.read_csv(HISTORIES_CSV)
        
        # Merge nodes
        for _, row in cities_df.iterrows():
            session.execute_write(create_city_node, row.to_dict())
        for _, row in flights_df.iterrows():
            session.execute_write(create_flight_node, row.to_dict())
        for _, row in hotels_df.iterrows():
            session.execute_write(create_hotel_node, row.to_dict())
        for _, row in restaurants_df.iterrows():
            session.execute_write(create_restaurant_node, row.to_dict())
        for _, row in preferences_df.iterrows():
            session.execute_write(create_preference_node, row.to_dict())
        for _, row in users_df.iterrows():
            session.execute_write(create_user_node, row.to_dict())
        for _, row in passports_df.iterrows():
            session.execute_write(create_passport_node, row.to_dict())
        for _, row in histories_df.iterrows():
            session.execute_write(create_history_node, row.to_dict())
        
        # Example of new relationships:
        # STAYED_AT (History->Hotel)
        for _, row in histories_df.iterrows():
            hist_id = row.get("history_id")
            hotels_str = row.get("hotels")
            if pd.notnull(hist_id) and isinstance(hotels_str, str) and hotels_str.strip():
                try:
                    hotel_ids = ast.literal_eval(hotels_str)
                    for h_id in hotel_ids:
                        session.execute_write(
                            create_relationship,
                            "History", "history_id", hist_id,
                            "STAYED_AT",
                            "Hotel", "hotel_id", h_id
                        )
                except:
                    pass

        # DINED_AT (History->Restaurant)
        for _, row in histories_df.iterrows():
            hist_id = row.get("history_id")
            rest_str = row.get("restaurants")
            if pd.notnull(hist_id) and isinstance(rest_str, str) and rest_str.strip():
                try:
                    rest_ids = ast.literal_eval(rest_str)
                    for r_id in rest_ids:
                        session.execute_write(
                            create_relationship,
                            "History", "history_id", hist_id,
                            "DINED_AT",
                            "Restaurant", "restaurant_id", r_id
                        )
                except:
                    pass

        # HAS_HOTEL_PREFERENCE (Preference->Hotel)
        for _, row in preferences_df.iterrows():
            pref_id = row.get("preference_id")
            top_hotels_str = row.get("top_hotels")
            if pd.notnull(pref_id) and isinstance(top_hotels_str, str) and top_hotels_str.strip():
                try:
                    hotel_ids = ast.literal_eval(top_hotels_str)
                    for h_id in hotel_ids:
                        session.execute_write(
                            create_relationship,
                            "Preference", "preference_id", pref_id,
                            "HAS_HOTEL_PREFERENCE",
                            "Hotel", "hotel_id", h_id
                        )
                except:
                    pass

        # HAS_RESTAURANT_PREFERENCE (Preference->Restaurant)
        for _, row in preferences_df.iterrows():
            pref_id = row.get("preference_id")
            top_rest_str = row.get("top_restaurants")
            if pd.notnull(pref_id) and isinstance(top_rest_str, str) and top_rest_str.strip():
                try:
                    rest_ids = ast.literal_eval(top_rest_str)
                    for r_id in rest_ids:
                        session.execute_write(
                            create_relationship,
                            "Preference", "preference_id", pref_id,
                            "HAS_RESTAURANT_PREFERENCE",
                            "Restaurant", "restaurant_id", r_id
                        )
                except:
                    pass

        # HAS_CITY_PREFERENCE (Preference->City)
        for _, row in preferences_df.iterrows():
            pref_id = row.get("preference_id")
            top_cities_str = row.get("top_cities")
            if pd.notnull(pref_id) and isinstance(top_cities_str, str) and top_cities_str.strip():
                try:
                    city_names = ast.literal_eval(top_cities_str)
                    for cname in city_names:
                        session.execute_write(
                            create_relationship,
                            "Preference", "preference_id", pref_id,
                            "HAS_CITY_PREFERENCE",
                            "City", "City", cname
                        )
                except:
                    pass

        # IS_READY_TO_APPLY_VISA (Preference->Passport) if preference["visa_preference"] == passport["Requirement"]
        prefs = preferences_df.to_dict("records")
        pports = passports_df.to_dict("records")
        for pref_row in prefs:
            pref_id = pref_row["preference_id"]
            v_pref = pref_row.get("visa_preference")
            if pd.notnull(pref_id) and pd.notnull(v_pref):
                for pport_row in pports:
                    pport_id = pport_row["passport_id"]
                    req = pport_row.get("Requirement")
                    if pd.notnull(pport_id) and pd.notnull(req):
                        if v_pref.strip() == req.strip():
                            session.execute_write(
                                create_relationship,
                                "Preference", "preference_id", pref_id,
                                "IS_READY_TO_APPLY_VISA",
                                "Passport", "passport_id", pport_id
                            )

        # REQUIRED_VISA_LIKE (Passport->History) if passport["Origin"] == history["issued_passport"]
        for pport_row in pports:
            pport_id = pport_row["passport_id"]
            origin = pport_row.get("Origin")
            if pd.notnull(pport_id) and pd.notnull(origin):
                for _, hist_row in histories_df.iterrows():
                    hist_id = hist_row["history_id"]
                    issued_p = hist_row.get("issued_passport")
                    if pd.notnull(hist_id) and pd.notnull(issued_p):
                        if origin.strip() == issued_p.strip():
                            session.execute_write(
                                create_relationship,
                                "Passport", "passport_id", pport_id,
                                "REQUIRED_VISA_LIKE",
                                "History", "history_id", hist_id
                            )

    print("Graph database build complete! All relationships merged.")

# Build the graph (comment out after first run if you want)
build_graph()

# ----------------------------
# 4. Build Representations for Retrieval
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
        "city_id": "City ID",
        "City": "City",
        "Country": "Country"
    }
    return build_representation(node._properties, fields)

def represent_flight(node):
    fields = {
        "flight_id": "Flight ID",
        "Airline": "Airline",
        "Total Fare (EUR)": "Total Fare"
    }
    return build_representation(node._properties, fields)

def represent_hotel(node):
    fields = {
        "hotel_id": "Hotel ID",
        "name": "Hotel Name",
        "price": "Price"
    }
    return build_representation(node._properties, fields)

def represent_restaurant(node):
    fields = {
        "restaurant_id": "Restaurant ID",
        "Restaurant Name": "Name",
        "Cuisines": "Cuisines"
    }
    return build_representation(node._properties, fields)

def represent_preference(node):
    fields = {
        "preference_id": "Preference ID",
        "visa_preference": "Visa Preference"
    }
    return build_representation(node._properties, fields)

def represent_user(node):
    fields = {
        "User_ID": "User ID",
        "Username": "Username"
    }
    return build_representation(node._properties, fields)

def represent_passport(node):
    fields = {
        "passport_id": "Passport ID",
        "Origin": "Origin",
        "Requirement": "Requirement"
    }
    return build_representation(node._properties, fields)

def represent_history(node):
    fields = {
        "history_id": "History ID",
        "city": "City",
        "country": "Country"
    }
    return build_representation(node._properties, fields)

# Gather all node data
with driver.session() as session:
    all_nodes = []
    for label, func in [
        ("City", represent_city),
        ("Flight", represent_flight),
        ("Hotel", represent_hotel),
        ("Restaurant", represent_restaurant),
        ("Preference", represent_preference),
        ("User", represent_user),
        ("Passport", represent_passport),
        ("History", represent_history)
    ]:
        result = session.run(f"MATCH (n:{label}) RETURN n")
        for record in result:
            node = record["n"]
            rep = func(node)
            if rep:
                all_nodes.append(rep)

representations = list(set(all_nodes))
print(f"Total representations for retrieval: {len(representations)}")

# ----------------------------
# 5. Compute Embeddings
# ----------------------------
print("Computing embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(representations, convert_to_tensor=True)

# ----------------------------
# 6. Retrieval
# ----------------------------
def retrieve_documents(query, top_k=5, similarity_threshold=0.5):
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    cos_scores = cosine_similarity(query_embedding.cpu().numpy(), doc_embeddings.cpu().numpy())[0]
    sorted_indices = np.argsort(cos_scores)[::-1]
    retrieved_docs = []
    for idx in sorted_indices:
        if cos_scores[idx] >= similarity_threshold:
            retrieved_docs.append(representations[idx])
        if len(retrieved_docs) >= top_k:
            break
    return retrieved_docs

# ----------------------------
# 7. LLM for Final Answer Generation
# ----------------------------
generator = pipeline(
    "text-generation",
    model="gpt2",
    do_sample=False,
    temperature=0.0,
    max_new_tokens=50
)

def generate_final_answer(query):
    retrieved_docs = retrieve_documents(query, top_k=5, similarity_threshold=0.5)
    if not retrieved_docs:
        return "No data found in the references."
    
    # Removed the strict instruction; now just providing data as context
    prompt = (
        "Here is some travel data that might help you:\n"
        + "\n".join(retrieved_docs)
        + "\n\nQuestion: "
        + query
        + "\nAnswer:"
    )
    result = generator(prompt, num_return_sequences=1) 
    generated_text = result[0]["generated_text"]
    # Extract the answer from the text
    answer = generated_text.replace(prompt, "").strip().split("\n")[0].strip()
    return answer

# ----------------------------
# 8. Flask Web Interface
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

from flask import Flask, request, render_template_string

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        question = request.form["question"]
        answer = generate_final_answer(question)
    return render_template_string(HTML_TEMPLATE, answer=answer)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
