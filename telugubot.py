import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from googletrans import Translator  # For translation

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: #f8f5e6;
        background-image: radial-gradient(#d4d0c4 1px, transparent 1px);
        background-size: 20px 20px;
    }
    .chat-font {
        font-family: 'Times New Roman', serif;
        color: #2c5f2d;
    }
    .user-msg {
        background: #ffffff !important;
        border-radius: 15px !important;
        border: 2px solid #2c5f2d !important;
    }
    .bot-msg {
        background: #fff9e6 !important;
        border-radius: 15px !important;
        border: 2px solid #ffd700 !important;
    }
    .stChatInput {
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Configure Google Gemini
genai.configure(api_key="AIzaSyA2XYTLxP5QV1BpvYbJE1OE9aD3cyiihc0")  # Replace with your Gemini API key
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model
translator = Translator()  # Initialize translator

# Load data and create FAISS index
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('telugu_dataset.csv')  # Replace with your dataset file name
        df['context'] = df.apply(
            lambda row: f"Question: {row['question']}\nAnswer: {row['answer_text']}", 
            axis=1
        )
        embeddings = embedder.encode(df['context'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])  # FAISS index for similarity search
        index.add(np.array(embeddings).astype('float32'))
        return df, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

# Load dataset and FAISS index
df, faiss_index = load_data()

# App Header
st.markdown('<h1 class="chat-font">Telugu Question Answering System</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask questions in English, get answers in Telugu and English!</h3>', unsafe_allow_html=True)
st.markdown("---")

# Function to find the closest matching question using FAISS
def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=1)  # Top 1 match
    if I.size > 0:
        return df.iloc[I[0][0]]['answer_text']  # Return the closest Telugu answer
    return None

# Function to translate Telugu to English
def translate_to_english(text):
    try:
        translation = translator.translate(text, src='te', dest='en')
        return translation.text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return text  # Return original text if translation fails

# Function to generate a refined answer using Gemini
def generate_refined_answer(query, telugu_answer):
    prompt = f"""You are a helpful and knowledgeable assistant. Refine the following answer in both English and Telugu:
    Question: {query}
    Telugu Answer: {telugu_answer}
    - Provide a detailed and accurate response in both languages.
    - Ensure the Telugu response is grammatically correct and culturally appropriate.
    """
    response = gemini.generate_content(prompt)
    return response.text

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="🙋" if message["role"] == "user" else "🕉️"):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question in English..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Finding the best answer..."):
        try:
            # Find the closest Telugu answer
            telugu_answer = find_closest_question(prompt, faiss_index, df)
            if telugu_answer:
                # Generate a refined answer using Gemini
                refined_answer = generate_refined_answer(prompt, telugu_answer)
                response = f"**Refined Answer**:\n{refined_answer}"
            else:
                response = "Sorry, I couldn't find an answer to your question."
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
