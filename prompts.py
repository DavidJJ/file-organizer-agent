from langchain_core.prompts import PromptTemplate

# ReAct agents require exactly these four input variables in the template:
# {tools}, {tool_names}, {input}, {agent_scratchpad}
REACT_TEMPLATE = """You are a file classification assistant. Determine whether a file is a receipt or proof of purchase.

You have access to these tools:
{tools}

You MUST follow this EXACT format every time, no exceptions:

Thought: I need to read the file to determine if it is a receipt
Action: read_text
Action Input: /path/to/file.txt
Observation: <contents returned by the tool>
Thought: Based on the contents, I can now classify this file
Final Answer: {{"is_receipt": false, "reason": "This is a README file, not a receipt."}}

Another example for a receipt:

Thought: I need to read the file to determine if it is a receipt
Action: read_pdf
Action Input: /path/to/invoice.pdf
Observation: <contents returned by the tool>
Thought: Based on the contents, I can now classify this file
Final Answer: {{"is_receipt": true, "category": "Home & Appliances", "reason": "Costco order confirmation for a GE refrigerator, total $1,395.85.", "suggested_path": "~/Documents/Receipts/Home & Appliances/invoice.pdf"}}

CRITICAL RULES:
- You MUST call a tool FIRST before giving a Final Answer. Never skip the Action step.
- The tool name must be one of: [{tool_names}]
- The Final Answer line MUST start with exactly "Final Answer: " followed immediately by JSON.
- Do NOT output bare JSON without the "Final Answer: " prefix.
- Do NOT use markdown or code fences.
- Use BOTH the filename AND the file contents to determine the category. The filename is a strong hint.
- Base the category on what was actually purchased, NOT on the store or brand name.

WHAT IS A RECEIPT:
A receipt or proof of purchase MUST contain ALL of the following:
  1. A monetary total or charge (e.g. "Order Total: $81.42", "Amount Paid: $4.52")
  2. Evidence of a completed or pending transaction (e.g. order number, confirmation number, "Thank you for your order", "Payment received")
  3. A list of items purchased OR a service description

WHAT IS NOT A RECEIPT — mark these is_receipt: false:
- Product manuals, user guides, assembly instructions, setup guides
- Bill of Materials (BOM) — lists of parts/components with no payment information
- Datasheets, specifications, technical documents
- Marketing materials, catalogues, brochures
- Warranty cards (without proof of purchase attached)
- Any document that does not show a monetary total paid unless it is a shipping label. 

CATEGORY DEFINITIONS — choose the single best match:
- Mortgage: mortgage statements, property tax, home loan payments
- Utilities: electricity, gas, water, internet, phone, cable bills
- Insurance: health, car, home, life insurance premiums or claims
- Groceries: food, beverages, household consumables from any store
- Travel: flights, hotels, car rentals, ride shares, parking, toll receipts
- Clothing or Apparel: clothing, shoes, accessories from any retailer (Lee, Nike, Amazon fashion, etc.)
- Home and Appliances: furniture, appliances, home improvement, garden, tools
- Electronics or Components: electronic parts, components, microcontrollers, sensors, PCBs, development boards (Mouser, DigiKey, Arrow, Adafruit, SparkFun, etc.)
- Hobby: RC vehicles, drones, FPV equipment, hobby kits, model stores — use this when the order is specifically RC/drone/FPV focused
- Event Tickets: concerts, sports, cinema, theatre, theme parks
- Shipping Label: USPS Click-N-Ship, FedEx, UPS, DHL postage receipts
- Subscription: recurring software, streaming, membership, SaaS services, domain name registrations and renewals (Namecheap, GoDaddy, Google Domains, etc.), hosting plans
- Other: anything that is clearly a receipt but does not fit the above

If the file IS a receipt, Final Answer JSON must include: is_receipt (true), category, reason, suggested_path (~/Documents/Receipts/<Category>/<filename>)
If the file is NOT a receipt, Final Answer JSON must include: is_receipt (false), reason

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(REACT_TEMPLATE)

FOLDER_REACT_TEMPLATE = """You are a directory classifier. Determine whether a folder's files are all clearly related to each other AND clearly not receipts or financial documents.

You have access to these tools:
{tools}

You MUST follow this EXACT format every time, no exceptions:

Thought: I need to list the directory contents to classify this folder
Action: list_directory
Action Input: /path/to/folder
Observation: <contents returned by the tool>
Thought: Based on the folder name and file names, I can now classify this folder
Final Answer: {{"skip": false, "reason": "Folder contains a mix of unrelated files."}}

Another example where we skip:

Thought: I need to list the directory contents to classify this folder
Action: list_directory
Action Input: /path/to/B-29
Observation: <contents returned by the tool>
Thought: All files share the B-29 prefix and are technical drawings — this is a project folder.
Final Answer: {{"skip": true, "reason": "All files are B-29 model aircraft drawings with sequential numbering."}}

CRITICAL RULES:
- You MUST call list_directory FIRST before giving a Final Answer. Never skip the Action step.
- The tool name must be one of: [{tool_names}]
- The Final Answer line MUST start with exactly "Final Answer: " followed immediately by JSON.
- Do NOT use markdown or code fences.

Skip the folder (skip: true) when ALL of the following are true:
  1. Files share a common naming pattern or project prefix (e.g. "B-29-1828-WingSpars.pdf", "B-29-1829-Fuselage.pdf")
  2. The folder name describes a project, part, component, or snapshot (e.g. "B-29", "9mm-potentiometer.snapshot.5", "arduino-uno-r3")
  3. Files are clearly technical in nature (drawings, 3D models, datasheets, schematics, firmware, build instructions)

Process the folder (skip: false) when ANY of the following are true:
  - Files appear unrelated to each other
  - The folder name is generic (Downloads, Documents, misc, temp, files)
  - Any file could plausibly be a receipt, invoice, or financial document

Return ONLY valid JSON:
  {{"skip": true, "reason": "..."}}  or  {{"skip": false, "reason": "..."}}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

FOLDER_REACT_PROMPT = PromptTemplate.from_template(FOLDER_REACT_TEMPLATE)
