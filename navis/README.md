# Navis Agents

## Index

What is Navis Agents?
Navis Agents Features
Navis Agents Architecture
Navis Agents Workflows
Navis Agents WebUI usage


## What is Navis Agents?

Navis Agents is a Compute over data platform for Filecoin storage using Akave O3 which enables AI agent storing agent metadata and storing open source models such as Google Gemma 3.
Navis Agents enables verifiability via storing all agent data and interactions Filecoin storage as well as enabling compute over data workflows with Filecoin data with several AI models from Google alongside open source models.

## Navis Agents workflows

Check Web Application screenshots in repository images folder for Navis Agent creation and interaction workflows

1. User creates agent → Agent Metadata and open source model for agents stored in Akave O3
2. Agent receives request → Accesses Filecoin CIDs as context (optional)
3. AI model processes request → Returns response
4. Logs stored → Decentralized on Filecoin

## Navis Agents features

### Agent Creation
- **Multi-Model Selection**
  - Google Gemini Pro 2.5 for complex reasoning and Google Search
  - Open source models such as Gemma 3 uploaded to Filecoin via Akave O3

### Data Management  
- **Decentralized Storage via Filecoin**
  - Store agent metadata permanently
  - Reference external datasets via CIDs
  - Maintain complete data ownership and verifiability of agent interactions

### Agent Interaction
- **Web-Based Chat Interface**
  - Real-time responses
  - Filecoin data context-aware conversations

## Navis Agents Architecture 

┌─────────────┐
│    Agent    │
│  Metadata   │ ──────► Akave O3
└─────────────┘
       │
       ▼
┌─────────────┐
│   Agent     │
│   Context   │ ──────► Filecoin Content Identifiers
│   Files     │
└─────────────┘
       │
       ▼
┌─────────────┐
│  Navis Agent│ ──────► Google Gemini Pro 2.5 /Open Source Models via Navis AI runtime
│  AI Models  │
└─────────────┘

┌──────────────────────────────────────────────────┐
│                  Frontend                        │
│  ┌────────────┬────────────┬────────────────┐    │
│  │  Agent     │    Chat    │     Create     │    │
│  │  List      │  Interface │     Agent      │    │
│  └────────────┴────────────┴────────────────┘    │
└──────────────────────────────────────────────────┘
                         │
                    REST API
                         │
┌──────────────────────────────────────────────────┐
│                Backend API                       │
│  ┌────────────────────────────────────────────┐  │
│  │          NavisAgentStore                   │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │  Create  │  │  Update  │  │  Delete  │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  │  │
│  └────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────┐  │
│  │           NavisAgent                       │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │AI Provider (Google, Navis AI runtime)│  │  │
│  │  └──────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼────────────────┐
        │                         │                │
┌───────▼───────────────┐ ┌───────▼───────┐ ┌──────▼───────┐
│       Storage         │ │  AI Models    │ │   Filecoin   │
│(Filecoin via Akave O3)│ │(Gemini,Gemma3)│ │    (CIDs)    │
└───────────────────────┘ └───────────────┘ └──────────────┘

## Navis Agents workflows

Check Web Application screenshots in images folder for Navis Agent creation and interaction workflows

1. User creates agent → Agent Metadata and open source model for agents stored on Akave O3
2. Agent receives request → Accesses Filecoin CIDs as context (optional)
3. AI model processes request → Returns response
4. Logs stored → Decentralized on Filecoin

## Navis Agents Web Application usage

1. **Access Navis Web Application to create an Agent**
   - Click "Create Navis Agent"
   - Name: "Research Assistant"
   - Model: "Gemini Pro 2.5"
   - Profile: "Agent focused on Artificial Intelligence research" 

- Add Filecoin data to your agent on creation. AI research papers from arxiv.org in this case.
- Customize agent behavior 
- Explore advanced features 

2. **Start Conversation with Navis Agent**
   - Select your agent from the dropdown
   - Ask: "Tell me about latest research papers about context engineering"
   - Your agent response using Filecoin data as context alongisde current data via Google Search
   Example Navis Agent Use Case:
    - Upload research papers about context engineering as knowledge base (TXT, 25MB)
    - CID(s): `bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi`
    - Navis Agent can now answer both reading Filecoin CIDs plus Google Search: "Tell me about latest research papers about context engineering"

## Navis Agent Features

### Navis Agent Creation
- **Multi-Model Selection**
  - Google Gemini Pro 2.5 for complex reasoning
  - Gemma 3 and other open source models (coming soon)

### Navis Agent Data Management  
- **Decentralized Storage via Filecoin**
  - Store agent metadata and models
  - Reference external datasets via CIDs

### Navis Agent Agent Interaction
- **Web-Based Chat Interface**
  - Real-time responses with Google Search
  - Context-aware conversations
  - Multi-turn dialogue support
  
  Example request: "Tell me about latest research papers about context engineering"

## Navis Agents Web Application

Navis Agents provides Web Application to create and interact with agents. You can see WebUI workflows step by step with screenshots of Navis Web Application in images folder.

1. Creating Navis Agent

How to create a new Navis Agent with Filecoin data as context.

Navigate to Create Agent Tab

Click on the "Create Navis Agent" tab at the top

Fill in Agent details and create Navis Agent

Click the "Create Navis Agent" button
You'll see notification: "✓ Loaded 1 agents" and the Navis agent will appear in agents list

Working with Filecoin Storage
Adding Filecoin Context to Agents
Filecoin Content Identifiers allow Navis Agents to reference decentralized data:

Obtain Filecoin Content Identifiers from your stored data on Filecoin
Add Filecoin Content Identifiers during agent creation
Multiple Filecoin Content Identifiers can be added, one per line
Agent will have access to this data as context during conversations

2. Managing Navis Agents

After creating Navis agents, navigate to the "Navis Agents" tab to see your agents list.

Each agent displays:

Agent name
Unique agent ID
Selected AI model
Number of Filecoin files attached
Agent profile description
Action buttons (Edit, Chat, Delete)

Deleting a Navis Agent

Click the "Delete" button on the agent card
Confirm deletion in the popup dialog
Click "OK" to permanently remove the agent

3. Interfacing with Navis Agents 

Navigate to the Chat tab
Select an agent from the dropdown menu

Navis agent's details will be displayed:
Agent name and model
Agent ID
Agent profile

Interfacing with Navis Agent

Type your request in the text input field
Click "Send" or press Enter

Navis Agent will process request and respond with context-aware responses based on Filecoin data using Filecoin context identifiers specified at creation with Google Search for response grounding