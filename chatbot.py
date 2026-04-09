import json
from typing import List, Dict
from transformers import pipeline
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

class ChatbotAgent:
    def __init__(self, json_file: str = os.getenv("FORMATTED_DATA_FILE", "formatted_data.json"), faiss_dir: str = "faiss_index"):
        self.json_file = json_file
        self.faiss_dir = faiss_dir
        self.conversation_history = []
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Load FAISS index
        try:
            self.vector_store = FAISS.load_local(self.faiss_dir, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS index: {str(e)}")
            self.vector_store = None
        # Initialize Hugging Face chat model
        try:
            self.chat_model = pipeline(
                "text-generation",
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                model_kwargs={"load_in_4bit": True} if os.getenv("USE_4BIT", "false").lower() == "true" else {},
                device=-1  # Use CPU; set to 0 for GPU
            )
        except Exception as e:
            print(f"Error loading Mixtral, falling back to distilgpt2: {str(e)}")
            self.chat_model = pipeline("text-generation", model="distilgpt2", device=-1)

    def load_data(self) -> Dict:
        """Load formatted JSON data."""
        try:
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'data': []}

    def retrieve_documents(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant documents using FAISS."""
        if not self.vector_store:
            return []
        docs = self.vector_store.similarity_search(query, k=k)
        return [
            {
                'text': doc.page_content,
                'source_url': doc.metadata['source_url'],
                'link_url': doc.metadata['link_url'],
                'title': doc.metadata['title']
            } for doc in docs
        ]

    def generate_response(self, query: str) -> str:
        """Generate a response using RAG with the chat model."""
        self.conversation_history.append({'query': query, 'response': ''})
        
        # Retrieve relevant documents
        docs = self.retrieve_documents(query)
        if not docs:
            return "Sorry, I could not find relevant information."

        # Prepare context
        context = "Related information:\n"
        for doc in docs:
            context += f"- {doc['title']} ({doc['link_url']}) from {doc['source_url']}\n"
        
        # Include recent conversation history
        history = "\n".join([f"Q: {h['query']} A: {h['response']}" for h in self.conversation_history[-2:-1]])
        prompt = (
            f"<s>[INST] You are a helpful chatbot. Answer the query based on the provided context and conversation history. "
            f"Keep the response concise, conversational, and relevant. If the context is insufficient, provide a general answer "
            f"and include relevant links.\n\n"
            f"History:\n{history}\n\nQuery: {query}\nContext:\n{context}\n[/INST]"
        )

        # Generate response
        try:
            response = self.chat_model(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                truncation=True
            )[0]['generated_text']
            response = response.replace(prompt, '').strip()
            if not response:
                response = f"I don't have detailed info on '{query}', but here are related links:\n{context}"
            self.conversation_history[-1]['response'] = response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            response = f"I encountered an issue, but here are related links:\n{context}"
            self.conversation_history[-1]['response'] = response
        
        return response

    def get_followup_suggestions(self) -> List[str]:
        """Generate follow-up suggestions."""
        if not self.conversation_history or not self.vector_store:
            return []
        docs = self.retrieve_documents(self.conversation_history[-1]['query'])
        return [doc['link_url'] for doc in docs[:3]]

def main():
    chatbot = ChatbotAgent()
    print("Welcome to the Client Support Chatbot! Type 'exit' to quit.")
    while True:
        query = input("Your question: ")
        if query.lower() == 'exit':
            break
        
        response = chatbot.generate_response(query)
        print("\nResponse:")
        print(response)
        
        suggestions = chatbot.get_followup_suggestions()
        if suggestions:
            print("\nYou might also be interested in:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")

if __name__ == "__main__":
    main()