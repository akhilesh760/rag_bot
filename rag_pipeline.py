import os

from dotenv import load_dotenv
from pinecone import Pinecone

from sentence_transformers import SentenceTransformer

from groq import Groq

# Load local `.env` for development (no-op in prod if env vars already set)
load_dotenv()

# -------------------------------
# API KEYS
# -------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError(
        "Missing PINECONE_API_KEY. Set it in your environment or in a local .env file."
    )

if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Set it in your environment or in a local .env file."
    )

INDEX_NAME = "vsoft-rag"

# -------------------------------
# Similarity Threshold
# -------------------------------

THRESHOLD = 0.5

# -------------------------------
# Load Embedding Model
# -------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")

# -------------------------------
# Connect Pinecone
# -------------------------------

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(INDEX_NAME)

print("Connected to Pinecone")

# -------------------------------
# Connect Groq
# -------------------------------

client = Groq(
    api_key=GROQ_API_KEY
)

print("Connected to Groq")

# -------------------------------
# Ask Question Function
# -------------------------------

def ask_question(query):

    print(f"\nQuestion: {query}")

    # --------------------------------
    # Convert Query to Embedding
    # --------------------------------

    query_embedding = model.encode(query).tolist()

    # --------------------------------
    # Search Pinecone
    # --------------------------------

    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )

    matches = results["matches"]

    # --------------------------------
    # If no matches found
    # --------------------------------

    if len(matches) == 0:

        prompt = f"""
        Answer the following question.

        Question:
        {query}

        Answer:
        """

    else:

        # --------------------------------
        # Top Similarity Score
        # --------------------------------

        top_score = matches[0]["score"]

        print(f"\nTop Similarity Score: {top_score}")

        # =========================================
        # CASE 1 -> Use Pinecone Knowledge
        # =========================================

        if top_score > THRESHOLD:

            print("\nUsing Pinecone Knowledge Base")

            retrieved_chunks = []

            for match in matches:

                text = match["metadata"].get("text", "")

                retrieved_chunks.append(text)

            context = "\n\n".join(retrieved_chunks)

            print("\nRetrieved Context:\n")
            print(context)

            prompt = f"""
            You are a helpful AI assistant.

            Answer the user's question ONLY from the context below.

            If the answer is not present in the context,
            say:
            "I could not find the answer in the knowledge base."

            Context:
            {context}

            Question:
            {query}

            Answer:
            """

        # =========================================
        # CASE 2 -> General LLM Knowledge
        # =========================================

        else:

            print("\nUsing General LLM Knowledge")

            prompt = f"""
            You are a helpful AI assistant.

            Answer the following question naturally.

            Question:
            {query}

            Answer:
            """

    # --------------------------------
    # Generate Answer from Groq
    # --------------------------------

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    return answer