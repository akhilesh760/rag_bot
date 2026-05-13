import os

from dotenv import load_dotenv
from pinecone import Pinecone

# -------------------------------
# Your Pinecone API Key
# -------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise RuntimeError(
        "Missing PINECONE_API_KEY. Set it in your environment or in a local .env file."
    )

try:

    # Connect Pinecone
    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    print("✅ Pinecone Connected Successfully")

    # List indexes
    indexes = pc.list_indexes()

    print("\nAvailable Indexes:")
    print(indexes)

except Exception as e:

    print("❌ Error Connecting to Pinecone")
    print(e)