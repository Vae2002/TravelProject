import pandas as pd
from neo4j import GraphDatabase

# -------------------------
# Configuration
# -------------------------
# Update these variables as needed
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "eunice2002"

# CSV file paths (adjust the paths as necessary)
USERS_CSV = "adjusted_datasets/users.csv"
FLIGHTS_CSV = "adjusted_datasets/adjusted_flights.csv"
CITIES_CSV = "adjusted_datasets/adjusted_cities.csv"
HOTELS_CSV = "adjusted_datasets/adjusted_hotels.csv"
RESTAURANTS_CSV = "adjusted_datasets/adjusted_restaurants.csv"
PREFERENCES_CSV = "adjusted_datasets/preferences.csv"

# -------------------------
# Neo4j Driver
# -------------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -------------------------
# Functions to create nodes
# -------------------------
def create_user_node(tx, user_props):
    query = """
    MERGE (u:User {user_id: $user_id})
    SET u += $props
    """
    tx.run(query, user_id=user_props['user_id'], props=user_props)

def create_flight_node(tx, flight_props):
    query = """
    MERGE (f:Flight {flight_id: $flight_id})
    SET f += $props
    """
    tx.run(query, flight_id=flight_props['flight_id'], props=flight_props)

def create_city_node(tx, city_props):
    query = """
    MERGE (c:City {city_id: $city_id})
    SET c += $props
    """
    tx.run(query, city_id=city_props['city_id'], props=city_props)

def create_hotel_node(tx, hotel_props):
    query = """
    MERGE (h:Hotel {hotel_id: $hotel_id})
    SET h += $props
    """
    tx.run(query, hotel_id=hotel_props['hotel_id'], props=hotel_props)

def create_restaurant_node(tx, restaurant_props):
    query = """
    MERGE (r:Restaurant {restaurant_id: $restaurant_id})
    SET r += $props
    """
    tx.run(query, restaurant_id=restaurant_props['restaurant_id'], props=restaurant_props)

def create_preference_node(tx, pref_props):
    query = """
    MERGE (p:Preference {preference_id: $preference_id})
    SET p += $props
    """
    tx.run(query, preference_id=pref_props['preference_id'], props=pref_props)

# -------------------------
# Functions to create relationships
# -------------------------
def create_relationship(tx, label_from, key_from, value_from, 
                        rel_type, label_to, key_to, value_to):
    # Generic relationship creator using MERGE on both nodes
    query = f"""
    MATCH (a:{label_from} {{{key_from}: $value_from}})
    MATCH (b:{label_to} {{{key_to}: $value_to}})
    MERGE (a)-[r:{rel_type}]->(b)
    """
    tx.run(query, value_from=value_from, value_to=value_to)

# -------------------------
# Main function to build the graph
# -------------------------
def build_graph():
    # Load CSV files
    users_df = pd.read_csv(USERS_CSV)
    flights_df = pd.read_csv(FLIGHTS_CSV)
    cities_df = pd.read_csv(CITIES_CSV)
    hotels_df = pd.read_csv(HOTELS_CSV)
    restaurants_df = pd.read_csv(RESTAURANTS_CSV)
    prefs_df = pd.read_csv(PREFERENCES_CSV)
    
    with driver.session() as session:
        # Create User nodes
        for _, row in users_df.iterrows():
            user_props = row.to_dict()
            session.write_transaction(create_user_node, user_props)
        
        # Create Flight nodes
        for _, row in flights_df.iterrows():
            flight_props = row.to_dict()
            session.write_transaction(create_flight_node, flight_props)
        
        # Create City nodes
        for _, row in cities_df.iterrows():
            city_props = row.to_dict()
            session.write_transaction(create_city_node, city_props)
        
        # Create Hotel nodes
        for _, row in hotels_df.iterrows():
            hotel_props = row.to_dict()
            session.write_transaction(create_hotel_node, hotel_props)
        
        # Create Restaurant nodes
        for _, row in restaurants_df.iterrows():
            restaurant_props = row.to_dict()
            session.write_transaction(create_restaurant_node, restaurant_props)
        
        # Create Preference nodes
        for _, row in prefs_df.iterrows():
            pref_props = row.to_dict()
            session.write_transaction(create_preference_node, pref_props)
        
        # -------------------------
        # Create Relationships
        # -------------------------
        # Users BOOKED_FLIGHT Flights
        for _, row in flights_df.iterrows():
            # Assumes flights_df has 'user_id' to indicate the user who booked
            user_id = row.get('user_id')
            flight_id = row.get('flight_id')
            if pd.notnull(user_id) and pd.notnull(flight_id):
                session.write_transaction(create_relationship,
                                            "User", "user_id", user_id,
                                            "BOOKED_FLIGHT",
                                            "Flight", "flight_id", flight_id)
        
        # Flights HAS_FLIGHT Cities
        for _, row in flights_df.iterrows():
            flight_id = row.get('flight_id')
            city_id = row.get('city_id')  # destination city from flight record
            if pd.notnull(flight_id) and pd.notnull(city_id):
                session.write_transaction(create_relationship,
                                            "Flight", "flight_id", flight_id,
                                            "HAS_FLIGHT",
                                            "City", "city_id", city_id)
        
        # Users VISITED Cities
        # Here we assume that a flight implies the user visited that city.
        # If you have a separate dataset for visited cities, adjust accordingly.
        for _, row in flights_df.iterrows():
            user_id = row.get('user_id')
            city_id = row.get('city_id')
            if pd.notnull(user_id) and pd.notnull(city_id):
                session.write_transaction(create_relationship,
                                            "User", "user_id", user_id,
                                            "VISITED",
                                            "City", "city_id", city_id)
        
        # Users STAYED_AT Hotels
        for _, row in hotels_df.iterrows():
            # Assumes hotels_df has 'user_id' indicating the user who stayed
            user_id = row.get('user_id')
            hotel_id = row.get('hotel_id')
            if pd.notnull(user_id) and pd.notnull(hotel_id):
                session.write_transaction(create_relationship,
                                            "User", "user_id", user_id,
                                            "STAYED_AT",
                                            "Hotel", "hotel_id", hotel_id)
        
        # Users DINED_AT Restaurants
        for _, row in restaurants_df.iterrows():
            user_id = row.get('user_id')
            restaurant_id = row.get('restaurant_id')
            if pd.notnull(user_id) and pd.notnull(restaurant_id):
                session.write_transaction(create_relationship,
                                            "User", "user_id", user_id,
                                            "DINED_AT",
                                            "Restaurant", "restaurant_id", restaurant_id)
        
        # Users WANTS_TO_VISIT Cities
        # This example assumes users_df has a column "wants_to_visit_city_id".
        # If this information is stored elsewhere, adjust accordingly.
        if "wants_to_visit_city_id" in users_df.columns:
            for _, row in users_df.iterrows():
                user_id = row.get("user_id")
                city_id = row.get("wants_to_visit_city_id")
                if pd.notnull(user_id) and pd.notnull(city_id):
                    session.write_transaction(create_relationship,
                                                "User", "user_id", user_id,
                                                "WANTS_TO_VISIT",
                                                "City", "city_id", city_id)
        
        # Users HAS_PREFERENCE Preferences
        for _, row in prefs_df.iterrows():
            user_id = row.get('user_id')
            pref_id = row.get('preference_id')
            if pd.notnull(user_id) and pd.notnull(pref_id):
                session.write_transaction(create_relationship,
                                            "User", "user_id", user_id,
                                            "HAS_PREFERENCE",
                                            "Preference", "preference_id", pref_id)

if __name__ == "__main__":
    build_graph()
    driver.close()
    print("Graph database build complete!")
