# main.py
from agent_workflow import app
import os
import pprint

# Set up environment variables (Replace with your actual keys)
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

# --- Simulate User Input (P1 Production Issue) ---
slack_user_message = (
    "The checkout service is giving 500 errors when processing payments. "
    "I need a P1 ticket opened and I think it's related to the recent configuration change."
)

print("--- Incoming Slack Message ---")
print(f"User: {slack_user_message}\n")

# Run the LangGraph workflow
# The input is the initial Slack message
final_state = app.invoke(
    {"input": slack_user_message, "chat_history": [], "agent_scratchpad": []},
    config={"recursion_limit": 10} # Allow graph to loop a few times for triage
)

print("\n--- Final Agent Response (To be posted back to Slack) ---")
print(final_state["final_response"])

# Expected Output (simulated):
# ✅ Triage complete. Identified module: **billing_service**.
# 📝 Remedy Ticket R-99001 created. Assigned to: Vendor_TCS. Priority: P1.
# ➡️ Assigned based on **Warranty Check** (Warranty Ticket).
