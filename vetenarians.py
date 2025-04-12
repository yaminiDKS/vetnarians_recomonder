import streamlit as st
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (expects GEMINI_API_KEY in .env)
load_dotenv()
genai.configure(api_key="AIzaSyALkJMLyvzHfFYGNj4TILbNseqS5Y_0HgA")

# Convert a vet row to a text description
def row_to_text(row, city):
    return (
        f"City: {city}, "
        f"Name: {row.get('Name', '')}, "
        f"Address: {row.get('Address', '')}, "
        f"Phone: {row.get('Justdial Phone', '')}, "
        f"Business Phone: {row.get('Business Phone', '')}, "
        f"Hours of operations: {row.get('Hours of operations', '')}, "
        f"Rating: {row.get('Rating', '')}, "
        f"Reviews: {row.get('Reviews', '')}"
    )

# Get embeddings from Gemini API
def get_embedding(text, task_type="retrieval_document"):
    return genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type=task_type
    )['embedding']

# Load data and generate normalized embeddings (cached for efficiency)
@st.cache_data
def load_data_and_embeddings():
    data_dir = "data"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    all_texts = []
    for csv_file in csv_files:
        city_name = os.path.splitext(csv_file)[0].capitalize()
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        texts = df.apply(lambda row: row_to_text(row, city_name), axis=1).tolist()
        all_texts.extend(texts)
    embeddings = np.array([get_embedding(text) for text in all_texts]).astype("float32")
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    return all_texts, normalized_embeddings

# Query veterinarians based on user input
def query_vets(normalized_embeddings, all_texts, user_query, top_k=3):
    query_embedding = get_embedding(user_query, task_type="retrieval_query")
    query_embedding = np.array(query_embedding).astype("float32")
    # Normalize query embedding
    query_norm = np.linalg.norm(query_embedding)
    normalized_query = query_embedding / query_norm
    # Compute cosine similarity scores
    similarity_scores = np.dot(normalized_embeddings, normalized_query)
    # Get indices of top K most similar entries
    top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]
    relevant_texts = [all_texts[i] for i in top_k_indices]
    # Prepare prompt for Gemini response
    context = "\n".join(relevant_texts)
    prompt = f"""You are an assistant that helps users find veterinarians.
Below is a list of veterinarian entries most relevant to the user's request.

Veterinarian Info:
{context}

User Query:
{user_query}

Please respond in a helpful tone, summarizing the top suggestions clearly.

Answer:"""
    gemini_model = genai.GenerativeModel('models/gemini-2.0-flash-thinking-exp-01-21')
    response = gemini_model.generate_content(prompt)
    return {
        "top_vets_raw": relevant_texts,
        "generated_response": response.text
    }

# Load data once when the app starts
st.info("Loading veterinarian data and generating embeddings...")
all_texts, normalized_embeddings = load_data_and_embeddings()
st.info("Data loaded successfully.")

# Streamlit UI
st.title("Vet Finder")
st.write("Enter a query to find veterinarians, such as 'Top vets in Chennai' or 'Vets with high ratings in Madurai'.")

query = st.text_input("Enter your query")
top_k = st.slider("Number of vets to display", min_value=1, max_value=10, value=3)

if st.button("Search"):
    if query:
        with st.spinner("Searching for veterinarians..."):
            result = query_vets(normalized_embeddings, all_texts, query, top_k)
        st.write("### 🔍 Gemini Summary")
        st.write(result["generated_response"])
        st.write("### 📋 Top Veterinarian Entries")
        for i, text in enumerate(result["top_vets_raw"], 1):
            st.write(f"{i}. {text}")
    else:
        st.warning("Please enter a query to search for veterinarians.")