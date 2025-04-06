import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Load structured JSON data
with open("blog_data/insurance_pages.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Convert to LangChain Documents with metadata
documents = []
for entry in data:
    doc = Document(
        page_content=entry["text"],
        metadata={
            "url": entry["url"],
            "title": entry.get("title", "Untitled"),
            "source": "Ditto Blog"
        }
    )
    documents.append(doc)

# Step 3: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Step 4: Create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 5: Create and save vectorstore
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("vectorstore/insurance_faiss_index")

print("✅ FAISS index created and saved with metadata support!")
