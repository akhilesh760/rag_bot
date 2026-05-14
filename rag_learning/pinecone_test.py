from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(
    api_key="pcsk_69d2BQ_UQZUQP6XwajN6ngeKszjYPmBP4t7fetYFRNGbfZetTiG4fPxYXBEBS6Y5xZ57km "
)

# Connect to index
index = pc.Index("rag-chatbot")

print("Connected successfully")