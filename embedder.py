import json
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

class Embedder:
    def __init__(self, input_file: str = os.getenv("FORMATTED_DATA_FILE", "formatted_data.json"), output_dir: str = "faiss_index"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def load_formatted_data(self)->Dict:
        """Load formatted JSON data."""
        try:
            with open(self.input_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'data': []}

    def create_vector_store(self):
        """Create and save FAISS vector store from formatted data."""
        formatted_data = self.load_formatted_data()
        documents = []
        
        for item in formatted_data['data']:
            for section in item['related_sections']:
                documents.append(Document(
                    page_content=section['text'],
                    metadata={'source_url': item['url'], 'link_url': section['url'], 'title': section['title']}
                ))
        
        if not documents:
            print("Warning: No documents to embed.")
            return
        
        # Create FAISS index
        vector_store = FAISS.from_documents(documents, self.embeddings)
        os.makedirs(self.output_dir, exist_ok=True)
        vector_store.save_local(self.output_dir)
        print(f"FAISS index saved to {self.output_dir}")

if __name__ == "__main__":
    embedder = Embedder()
    embedder.create_vector_store()
    