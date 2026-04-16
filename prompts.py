from langchain_core.prompts import PromptTemplate

# ReAct agents require exactly these four input variables in the template:
# {tools}, {tool_names}, {input}, {agent_scratchpad}
REACT_TEMPLATE = """You are a file classification assistant. Determine whether a file is a receipt or proof of purchase.

You have access to these tools:
{tools}

Use this EXACT format — do not deviate:

Thought: I need to read the file to determine if it is a receipt
Action: the tool to use, must be one of [{tool_names}]
Action Input: the exact file path
Observation: the file contents returned by the tool
Thought: Based on the contents, I can now classify this file
Final Answer: a single valid JSON object (no markdown, no code fences)

If the file IS a receipt, the JSON must have these fields:
  is_receipt: true
  category: one of [Mortgage, Utilities, Insurance, Groceries, Travel, Event Tickets, Hobby / Radio Control, Subscription, Other] or a new category if clearly warranted
  reason: one sentence explaining why this is a receipt
  suggested_path: the destination path as ~/Documents/Receipts/<Category>/<filename>

If the file is NOT a receipt:
  is_receipt: false
  reason: one sentence explaining why this is not a receipt

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(REACT_TEMPLATE)
