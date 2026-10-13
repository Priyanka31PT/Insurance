# Enterprise Insurance Claim & Policy Intelligence Hub

An advanced RAG (Retrieval-Augmented Generation) pipeline built with **LangGraph** and **Flask** designed to handle mixed insurance datasets (structured CSVs and unstructured text files).

## 🚀 Key Architectural Features
- **Deterministic Query Routing:** Automatically separates factual/structured requests from abstract queries using regex and pattern tracking.
- **Zero-Token Factual Pathway:** Direct ID-lookups bypass the LLM completely, running directly out of fast memory lookups to eliminate API costs.
- **On-the-Fly Analytical RAG Node:** Runs vector search queries and embedding computations dynamically using a local `BAAI/bge-base-en-v1.5` encoder.
- **Security Guardrails:** Includes native input validation to block prompt injections and jailbreaking attempts.
- **Automated Pipeline Evaluation:** Integrated with the **Ragas** framework to score system outputs based on faithfulness and answer relevance.

## 🛠️ Local Setup Instructions
1. Clone this repository to your machine.
2. Initialize your Python virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install flask flask-cors langgraph openai sentence-transformers pandas python-dotenv ragas requests
   ```
4. Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_actual_key_here
   ```
5. Place your insurance data files inside a folder named `cleaned/`.
6. Start the Flask server: `python app.py` and open `index.html` to access the visual workspace dashboard.
