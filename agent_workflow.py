# agent_workflow.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from tools import ALL_TOOLS, search_knowledge_base, find_recent_changes, create_remedy_ticket
import operator

# --- 1. Define the Agent State ---
# This dictionary defines the information passed between nodes in the graph
class AgentState(TypedDict):
    input: str
    chat_history: List[str]
    agent_output: str
    module_name: str # Triage result from the LLM
    gitlab_result: dict # Result from the find_recent_changes tool
    final_response: str
    
# --- 2. Initialize the Gemini LLM and Agent ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

# The core tool-calling agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Intelligent Triage, Warranty, and Vendor Management Agent. Your first task is to determine the 'module_name' the user is having a problem with. If you need external info, use the search_knowledge_base tool. Once the module_name is identified, stop talking and transition to the next step."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent_executor = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

# --- 3. Define Graph Nodes (Steps) ---

def run_agent(state: AgentState):
    """Executes the core conversational and tool-calling agent."""
    result = agent_executor.invoke({"input": state["input"], "agent_scratchpad": state["agent_scratchpad"]})
    
    # Simple check to extract the module name if the LLM provided a structured answer 
    # (A real implementation would use a structured output parser)
    if "module_name" in result['output']:
        state["module_name"] = result['output'].split(":")[1].strip()
        state["agent_output"] = result['output']
        return state

    state["agent_output"] = result['output']
    return state

def gitlab_warranty_check(state: AgentState):
    """Checks GitLab for changes using the determined module name."""
    if not state.get("module_name"):
        return {"final_response": "Error: Module name was not identified during triage."}

    # Call the tool defined in tools.py
    print(f"\n--- Checking GitLab for module: {state['module_name']} ---")
    gitlab_result = find_recent_changes.invoke({"module_name": state["module_name"]})
    
    state["gitlab_result"] = gitlab_result
    return state

def assign_ticket(state: AgentState):
    """Determines assignment based on warranty/changes and creates the Remedy ticket."""
    gitlab_result = state["gitlab_result"]
    
    if gitlab_result.get("found_changes"):
        # Real logic would compare commit_date to incident_date and a warranty period (e.g., 30 days)
        # For simplicity, we assume changes *found* means a warranty check is possible.
        is_in_warranty = gitlab_result.get("author", "").startswith("Vendor_") # Simplistic rule
        
        if is_in_warranty:
            # Assign to vendor (e.g., TCS, Infosys, Incedo)
            assignment_group = f"Vendor_{gitlab_result['author'].split('_')[1]}"
            ticket_type = "Warranty Ticket"
        else:
            # Assign to internal system architect
            assignment_group = f"SA_Architect_{state['module_name']}"
            ticket_type = "Production Incident"
    else:
        # No recent changes found, assign to L2 support by default
        assignment_group = "L2_Support_Triage"
        ticket_type = "Production Incident"

    # Create the Remedy ticket via the tool
    remedy_response = create_remedy_ticket.invoke({
        "summary": f"{state['module_name']} Production Issue - {ticket_type}",
        "description": state["input"],
        "assignment_group": assignment_group,
        "priority": "P1" 
    })
    
    final_message = (
        f"✅ Triage complete. Identified module: **{state['module_name']}**.\n"
        f"📝 {remedy_response}\n"
        f"➡️ Assigned based on **Warranty Check** ({ticket_type})."
    )
    return {"final_response": final_message}

# --- 4. Define Graph Edges (Transitions) ---

def decide_next_step(state: AgentState):
    """Decision node to determine if triage needs RAG, more conversation, or can proceed to action."""
    if state.get("module_name"):
        # Module identified, proceed to checking code changes
        return "gitlab_check"
    
    # Check if the agent called a tool (e.g., search_knowledge_base)
    # This logic is simplified; a real agent uses structured output for tools/actions
    if "search_knowledge_base" in state.get("agent_output", ""):
        # If RAG was used, loop back to the agent to synthesize the answer
        return "run_agent" 
        
    # Default: Continue conversation until module is identified
    return "run_agent"

# --- 5. Build the LangGraph ---
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("run_agent", run_agent)
workflow.add_node("gitlab_check", gitlab_warranty_check)
workflow.add_node("assign_ticket", assign_ticket)

# Define the entry point and conditional edge
workflow.set_entry_point("run_agent")
workflow.add_conditional_edges(
    "run_agent",
    decide_next_step,
    {
        "gitlab_check": "gitlab_check",  # If module is identified
        "run_agent": "run_agent"        # If more conversation/RAG is needed
    }
)

# Define sequential edges
workflow.add_edge("gitlab_check", "assign_ticket")
workflow.add_edge("assign_ticket", END)

# Compile the graph
app = workflow.compile()
