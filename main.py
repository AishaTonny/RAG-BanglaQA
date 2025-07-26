from transformers import pipeline
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ✅ Step 1: Load multilingual QA model
print(" Loading QA model...")
qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-multi-cased-finetuned-xquadv1"
)

# ✅ Step 2: Extract text from PDF
print(" Extracting text from PDF...")
pdf_path = r"D:\MyProjects\RAG-BanglaQA\HSC26-Bangla1st-Paper.pdf"
reader = PdfReader(pdf_path)
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# ✅ Step 3: Split text into chunks
print(" Splitting into chunks...")
chunk_size = 300
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ✅ Step 4: Embed chunks and create vector index
print(" Creating vector index...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = embedding_model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ✅ Step 5: Accept user question and generate answer
while True:
    query = input("\n Enter your question (or type 'exit'): ")
    if query.lower() == 'exit':
        break

    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    context = " ".join([chunks[i] for i in I[0]])

    result = qa_pipeline(question=query, context=context)
    print("✅ Answer:", result['answer'])
