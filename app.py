# D:\projects\insurance\app.py
from dotenv import load_dotenv
load_dotenv() 

import os
import re
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
CLEANED_FOLDER = BASE_DIR / "cleaned"

# 1. INITIALIZE MEMORY ARRAYS AT STARTUP (~2 Seconds)
print("Loading raw insurance assets directly into memory storage...")
csv_records = {}   # Holds structural row data mapped by ID: {"CLM300000": "row text"}
policy_corpus = [] # Holds unstructured text chunks for RAG

# Load CSV files line by line into a fast lookup dictionary
for csv_file in CLEANED_FOLDER.glob("*.csv"):
    try:
        df = pd.read_csv(csv_file)
        id_column = next((col for col in df.columns if 'id' in col.lower()), None)
        if id_column:
            for _, row in df.iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                csv_records[str(row[id_column]).upper().strip()] = row_str
    except Exception as e:
        print(f"Skipping CSV {csv_file.name}: {e}")

# Load TXT files into paragraph blocks for RAG
for txt_file in CLEANED_FOLDER.glob("*.txt"):
    try:
        text_content = txt_file.read_text(encoding="utf-8")
        # Split into logical paragraphs to act as pseudo-chunks
        paragraphs = [p.strip() for p in text_content.split("\n\n") if len(p.strip()) > 30]
        policy_corpus.extend(paragraphs)
    except Exception as e:
        print(f"Skipping TXT {txt_file.name}: {e}")

print(f"Memory Sync Complete: Loaded {len(csv_records)} CSV rows & {len(policy_corpus)} Policy text blocks.")

# Load models locally on-the-fly
print("Loading lightweight local BGE model...")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CLAIM_ID_PATTERN = r"CLM\d+"
POLICY_ID_PATTERN = r"POL\d+"

# 2. LANGGRAPH STATE DEFINITION
class AgentState(TypedDict):
    query: str
    target_id: Optional[str]
    route: Optional[str]
    context: Optional[str]
    response: Optional[str]
    guardrail_status: Optional[str]

# 3. ROUTING & EXECUTION NODES
def input_guardrail_node(state: AgentState) -> AgentState:
    query = state["query"].lower()
    malicious_keywords = ["ignore previous instructions", "system prompt", "override role"]
    if any(keyword in query for keyword in malicious_keywords):
        return {**state, "guardrail_status": "BLOCKED", "response": "Guardrail Alert: Malicious prompt attempt blocked."}
    return {**state, "guardrail_status": "PASSED"}

def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    claim_match = re.search(CLAIM_ID_PATTERN, query, re.IGNORECASE)
    policy_match = re.search(POLICY_ID_PATTERN, query, re.IGNORECASE)
    
    target_id = None
    if claim_match: target_id = claim_match.group(0).upper()
    elif policy_match: target_id = policy_match.group(0).upper()
        
    # Route Condition
    if target_id and any(kw in query.lower() for kw in ["status", "date", "amount"]):
        return {**state, "target_id": target_id, "route": "factual"}
    return {**state, "target_id": target_id, "route": "analytical"}

def factual_execution_node(state: AgentState) -> AgentState:
    """ROUTE 1: Fast exact memory dictionary lookup. Bypasses LLM completely."""
    target_id = state["target_id"]
    record = csv_records.get(target_id)
    
    if record:
        return {**state, "response": f"Direct Record Extracted (No LLM): {record}"}
    return {**state, "response": f"System Alert: ID {target_id} recognized, but not found in current records."}

def analytical_execution_node(state: AgentState) -> AgentState:
    """ROUTE 2: Computes embeddings ON-THE-FLY only for the text needed during query runtime."""
    query = state["query"]
    
    if not policy_corpus:
        return {**state, "context": "", "response": "No document assets found in memory."}
        
    # Generate vector embedding for just the incoming question and current corpus
    query_vector = embedding_model.encode(query, convert_to_tensor=True)
    corpus_vectors = embedding_model.encode(policy_corpus, convert_to_tensor=True)
    
    # Compute scores instantly in memory
    cosine_scores = util.cos_sim(query_vector, corpus_vectors)[0]
    top_results = cosine_scores.argsort(descending=True)[:3] # Grab top 3 matches
    
    retrieved_chunks = [policy_corpus[idx] for idx in top_results]
    context_text = "\n\n".join(retrieved_chunks)
    
    # Pass to LLM for final generation
    llm_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional insurance assistant. Synthesize the provided context chunks accurately."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
    )
    return {**state, "context": context_text, "response": llm_response.choices.message.content}

# 4. GRAPH PIPELINE ASSEMBLY
workflow = StateGraph(AgentState)
workflow.add_node("input_guardrail", input_guardrail_node)
workflow.add_node("router", router_node)
workflow.add_node("factual_backend", factual_execution_node)
workflow.add_node("analytical_backend", analytical_execution_node)

workflow.set_entry_point("input_guardrail")
workflow.add_conditional_edges("input_guardrail", lambda state: "END" if state["guardrail_status"] == "BLOCKED" else "router", {"END": END, "router": "router"})
workflow.add_conditional_edges("router", lambda state: state["route"], {"factual": "factual_backend", "analytical": "analytical_backend"})
workflow.add_edge("factual_backend", END)
workflow.add_edge("analytical_backend", END)
graph_engine = workflow.compile()

@app.route('/api/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    initial_state = {"query": data['query'], "target_id": None, "route": None, "context": None, "response": None, "guardrail_status": None}
    final_output = graph_engine.invoke(initial_state)
    return jsonify({
        "query": final_output["query"],
        "route_taken": final_output["route"],
        "guardrail_status": final_output["guardrail_status"],
        "response": final_output["response"]
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
