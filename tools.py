# tools.py
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from typing import Literal

# --- Configuration (Replace with actual details) ---
ES_URL = "http://localhost:9200"
ES_INDEX = "it_runbooks"

# Initialize Embeddings and Vector Store for RAG
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
try:
    es_vectorstore = ElasticsearchStore(
        es_url=ES_URL,
        index_name=ES_INDEX,
        embedding=embeddings,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy()
    )
except Exception as e:
    print(f"Warning: Could not connect to Elasticsearch. RAG will not function: {e}")
    # Create a mock store if ES is unavailable
    es_vectorstore = None

# --- RAG Tool ---
@tool
def search_knowledge_base(query: str) -> str:
    """Use this tool to search the internal knowledge base (KB) for diagnostic
    information, runbooks, and module documentation to assist the conversation."""
    if es_vectorstore:
        # Retrieve top 3 relevant documents
        docs = es_vectorstore.similarity_search(query, k=3)
        return "\n\n".join([f"Source: {d.metadata.get('source', 'KB')}\nContent: {d.page_content}" for d in docs])
    return "Knowledge base search is currently unavailable. Proceeding with triage."


# --- Action Tools (Simulating MCP/API calls) ---

@tool
def find_recent_changes(module_name: str) -> dict:
    """Searches GitLab for the last 5 commits on the 'main' branch of the given module.
    Returns details including change ID, date, and author. Essential for warranty check."""
    # Simulates a call to the GitLab MCP Server
    if module_name.lower() == "billing_service":
        return {"found_changes": True, "commit_date": "2025-11-28", "author": "Vendor_TCS", "id": "COMMIT-A123"}
    return {"found_changes": False}

@tool
def create_remedy_ticket(summary: str, description: str, assignment_group: str, priority: Literal["P1", "P2"]) -> str:
    """Creates a production incident in Remedy and assigns it to the specified group."""
    # Simulates a call to the Remedy MCP Server
    return f"Remedy Ticket R-99001 created. Assigned to: {assignment_group}. Priority: {priority}."

# --- The list of all tools the Gemini LLM can use ---
ALL_TOOLS = [search_knowledge_base, find_recent_changes, create_remedy_ticket]
