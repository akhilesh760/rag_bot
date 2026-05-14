import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# LangChain Imports
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# -------------------------------
# Load Environment Variables
# -------------------------------

load_dotenv()

# -------------------------------
# API KEYS
# -------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError(
        "Missing PINECONE_API_KEY. Set it in environment or .env file."
    )

if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Set it in environment or .env file."
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

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

print("Connected to Pinecone")

# -------------------------------
# Connect Groq
# -------------------------------

client = Groq(api_key=GROQ_API_KEY)

print("Connected to Groq")

# =====================================================
# LANGCHAIN MEMORY
# =====================================================

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=False
)

print("Conversation memory initialized")

# =====================================================
# PROMPT TEMPLATES
# =====================================================

# Prompt for RAG responses
rag_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
You are a helpful AI assistant.

Previous Conversation:
{chat_history}

Use ONLY the context below to answer the user question.

If the answer is not available in the context,
say:
"I could not find the answer in the knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""
)

# Prompt for general chatbot responses
normal_prompt = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""
You are a helpful AI assistant.

Previous Conversation:
{chat_history}

Answer the following question naturally and clearly.

Question:
{question}

Answer:
"""
)

print("Prompt templates initialized")

# =====================================================
# HELPER FUNCTION FOR GROQ RESPONSE
# =====================================================


def generate_response(prompt):

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

    return response.choices[0].message.content


# =====================================================
# MAIN ASK QUESTION FUNCTION
# =====================================================


def ask_question(query):

    print(f"\nQuestion: {query}")

    # --------------------------------
    # Load Previous Conversation
    # --------------------------------

    chat_history = memory.load_memory_variables({})
    history = chat_history.get("chat_history", "")

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

    # =====================================================
    # CASE 1 -> NO MATCHES FOUND
    # =====================================================

    if len(matches) == 0:

        print("\nNo Pinecone matches found")

        final_prompt = normal_prompt.format(
            question=query,
            chat_history=history
        )

    else:

        top_score = matches[0]["score"]

        print(f"\nTop Similarity Score: {top_score}")

        # =====================================================
        # CASE 2 -> USE RAG KNOWLEDGE BASE
        # =====================================================

        if top_score > THRESHOLD:

            print("\nUsing Pinecone Knowledge Base")

            retrieved_chunks = []

            for match in matches:

                text = match["metadata"].get("text", "")

                if text:
                    retrieved_chunks.append(text)

            context = "\n\n".join(retrieved_chunks)

            print("\nRetrieved Context:\n")
            print(context)

            final_prompt = rag_prompt.format(
                context=context,
                question=query,
                chat_history=history
            )

        # =====================================================
        # CASE 3 -> GENERAL LLM RESPONSE
        # =====================================================

        else:

            print("\nUsing General LLM Knowledge")

            final_prompt = normal_prompt.format(
                question=query,
                chat_history=history
            )

    # --------------------------------
    # Generate Final Answer
    # --------------------------------

    answer = generate_response(final_prompt)

    # --------------------------------
    # Save Conversation in Memory
    # --------------------------------

    memory.save_context(
        {"input": query},
        {"output": answer}
    )

    return answer


# =====================================================
# TESTING
# =====================================================

if __name__ == "__main__":

    while True:

        user_query = input("\nAsk Question: ")

        if user_query.lower() in ["exit", "quit"]:
            break

        response = ask_question(user_query)

        print("\nAI Response:\n")
        print(response)
