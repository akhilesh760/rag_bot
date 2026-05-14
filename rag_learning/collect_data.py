import requests
from bs4 import BeautifulSoup
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

#Function for creating embeddings
def create_embeddings(chunks):

    embeddings = model.encode(chunks)
    return embeddings

#Function to clean text

def clean_text(text):

    # Split text into lines
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Remove leading/trailing spaces
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip very short noisy lines
        if len(line) < 3:
            continue
        cleaned_lines.append(line)
    # Join lines into single text
    text = " ".join(cleaned_lines)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text

#Function for chunking 

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


#Function for data collection

def collect_website_data(url):
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content,"html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    raw_text = soup.get_text(separator="\n")
    # Clean text
    cleaned_text = clean_text(raw_text)
    # Create chunks
    chunks = chunk_text(cleaned_text)
    embeddings = create_embeddings(chunks)

    print("\n========== TOTAL CHUNKS ==========\n")
    print(len(chunks))

    print("\n========== FIRST 3 CHUNKS ==========\n")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n")
        print(chunk)

    print("\n========== EMBEDDING INFO ==========\n")
    print("Total embeddings:", len(embeddings))
    print("\nEmbedding dimension size:")
    print(len(embeddings[0]))
    print("\nFirst 10 values of first embedding:\n")
    print(embeddings[0][:10])
    return chunks, embeddings

if __name__ == "__main__":

    chunks, embeddings = collect_website_data(
        "https://www.vsoftconsulting.com"
    )

    print("\n========== TOTAL CHUNKS ==========\n")
    print(len(chunks))

    print("\n========== EMBEDDING INFO ==========\n")
    print("Total embeddings:", len(embeddings))