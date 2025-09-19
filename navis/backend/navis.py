"""
Navis Agents
A system for managing AI agents with Filecoin storage using Akave
"""

# =============================================================================
# IMPORTS
# =============================================================================

import boto3
from botocore.config import Config
import os
import uuid
import json
import threading
import traceback
import logging
import time
import random
import string
import requests
from typing import Dict, Optional, List, Tuple, Any, ClassVar
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Google Gemini Pro 2.5 imports 
from google import genai
from google.genai import types

# Navis AI runtime import
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Environment variables
# =============================================================================

# Google AI provider API Keys
GOOGLE_API_KEY = "AIza_xxx"

# Akave O3 API Keys
AKAVE_O3_PUBLIC_KEY = "O3_xxx"
AKAVE_O3_SECRET_KEY = "0Pe_xxx"

# Global agent storage instance - will be initialized before use
agent_storage = None

# AI providers dictionary
ai_providers = {
    "google": None,      # Google API
    "navis": {}          # Navis runtime for open source models such as Gemma3
}

# Initialize Akave storage
s3 = boto3.client(
    's3',
    aws_access_key_id='AKAVE_O3_PUBLIC_KEY',
    aws_secret_access_key='AKAVE_O3_SECRET_KEY',
    region_name='us-east-1',
    endpoint_url='https://o3-rc3.akave.xyz',
    config=Config(request_checksum_calculation="when_required", response_checksum_validation="when_required")
)
logger.info("✅ Akave O3 client initialized")


# =============================================================================
# FUNCTIONS FOR AI PROVIDERS SETUP AND MODEL DOWNLOAD FROM AKAVE
# =============================================================================

def download_folder_from_akave_o3(bucket_name, prefix='', local_dir='gemma'):
    """Download a folder from S3 to local directory"""
    os.makedirs(local_dir, exist_ok=True)
    # List objects in S3 folder
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    downloaded_files = []
    for page in pages:
        for obj in page.get('Contents', []):
            target = obj['Key']
            if target.endswith('/'):
                continue  # Skip directories
            local_file_path = os.path.join(local_dir, os.path.relpath(target, prefix))
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            logger.info(f"Downloading {target} to {local_file_path}")
            s3.download_file(bucket_name, target, local_file_path)
            downloaded_files.append(local_file_path)
    return downloaded_files


def setup_ai_providers():
    # Initialize AI providers including Google Gemini and Navis runtime
    
    # Configure the Google client
    google_client = genai.Client(api_key=GOOGLE_API_KEY)

    # Define the grounding tool
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    # Store both client and config
    ai_providers["google"] = {
        "client": google_client,
        "grounding_tool": grounding_tool
    }
    logger.info("✅ Loaded Google provider with Google Search grounding support!")
    
    # Initialize Navis AI runtime with Gemma3-270m model
    try:
        logger.info("Setting up Navis Agents AI runtime...")
        
        # Download Gemma model from Filecoin
        download_folder_from_akave_o3("navis", "gemma270m", "gemma_model")
        model_dir = 'gemma_model/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1'
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
        
        # Store in ai_providers
        ai_providers["navis"]["model"] = model
        ai_providers["navis"]["tokenizer"] = tokenizer
        
        logger.info("✅ Loaded Navis AI runtime with Gemma3-270m model!")
        
    except Exception as e:
        logger.warning(f"Failed to setup Navis runtime: {e}")
        logger.warning("Navis AI runtime will not be available")


def get_ai_provider(provider_name: str):
    """Get a specific AI provider"""
    return ai_providers.get(provider_name)


# =============================================================================
# NAVIS AGENT CLASS WITH FILECOIN CONTEXT
# =============================================================================

class NavisAgent(BaseModel):
    """
    Navis Agents Class with Filecoin context support
    Downloads and stores Filecoin files as context for LLM requests
    """

    SUPPORTED_MODELS: ClassVar[set[str]] = {
        'gemini-2.5-pro', # Google Gemini model
        'gemma-270m' # Navis AI Runtime Gemma model
    }

    # IPFS/Filecoin gateways to try (in order of preference)
    FILECOIN_GATEWAYS: ClassVar[List[str]] = [
        "https://ipfs.io/ipfs/",
        "https://gateway.pinata.cloud/ipfs/", 
        "https://cloudflare-ipfs.com/ipfs/",
        "https://dweb.link/ipfs/",
        "https://ipfs.filebase.io/ipfs/"
    ]

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    model: str
    profile: str
    description: str
    filecoin_cids: Optional[List[str]] = None

    model_config = {"extra": "allow"}

    def __init__(self, **data):
        super().__init__(**data)
        self.filecoin_context = ""
        # Download Filecoin files on initialization if CIDs are provided
        if self.filecoin_cids:
            self._download_filecoin_context()
    
    def _download_filecoin_context(self):
        """Download all Filecoin files and build context string"""
        if not self.filecoin_cids:
            self.filecoin_context = ""
            return

        logger.info(f"NavisAgent {self.name}: Retrieving {len(self.filecoin_cids)} Filecoin files...")
        
        context_parts = ["=== START CONTEXT ===\n"]
        
        for i, cid in enumerate(self.filecoin_cids, 1):
            try:
                file_content = self._download_filecoin_file(cid)
                if file_content:
                    context_parts.append(f"\n--- File {i}: CID {cid[:16]}... ---\n")
                    context_parts.append(file_content)
                    context_parts.append("\n")
                    logger.info(f"Downloaded CID {cid[:12]}... ({len(file_content)} chars)")
                else:
                    logger.warning(f"Failed to download CID {cid}")
                    context_parts.append(f"\n--- File {i}: CID {cid[:16]}... (DOWNLOAD FAILED) ---\n")
                    
            except Exception as e:
                logger.error(f"Error downloading CID {cid}: {str(e)}")
                context_parts.append(f"\n--- File {i}: CID {cid[:16]}... (ERROR: {str(e)}) ---\n")

        context_parts.append("=== END CONTEXT ===\n")
        self.filecoin_context = "".join(context_parts)
        
        logger.info(f"Built Filecoin context: {len(self.filecoin_context)} characters")

    def _download_filecoin_file(self, cid: str) -> Optional[str]:
        """Download file from Filecoin using multiple gateway attempts"""
        for gateway in self.FILECOIN_GATEWAYS:
            try:
                url = f"{gateway}{cid}"
                logger.debug(f"Attempting to download {cid[:12]}... from {gateway}")
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Try to decode as text
                try:
                    content = response.content.decode('utf-8')
                    return content
                except UnicodeDecodeError:
                    # Handle binary files
                    logger.warning(f"CID {cid} contains binary data")
                    return f"[BINARY FILE - CID: {cid}, Size: {len(response.content)} bytes]"
                    
            except Exception as e:
                logger.debug(f"Failed to download from {gateway}: {str(e)}")
                continue
                
        logger.error(f"Failed to download CID {cid} from all gateways")
        return None

    def _get_provider(self) -> str:
        """AI provider detection based on model name."""
        if "gemini" in self.model:
            return "google"
        elif "gemma-270m" in self.model:
            return "navis"
        else:
            return "google" # Set to Google Gemini model as default ✓ 

    def generate_response(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate response with Filecoin context included"""
        if not prompt.strip():
            return "Error: Empty prompt"

        provider = self._get_provider()
        client = ai_providers.get(provider)
        if not client:
            return f"Error: {provider} provider not configured"
        
        # Build system message with Filecoin context
        system_content = self.profile
        if hasattr(self, 'filecoin_context') and self.filecoin_context:
            system_content = f"{self.profile}\n\nContext:\n{self.filecoin_context}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]

        if provider == "google":
            # Handle Google provider with new genai client
            from google.genai import types

            google_client = client["client"]
            grounding_tool = client["grounding_tool"]

            # Check if the prompt seems to need web search
            search_keywords = ['search', 'research', 'find', 'browse', 'latest', 'current', 'recent', 'news', 'today']
            needs_search = any(keyword in prompt.lower() for keyword in search_keywords)

            # Combine system message and user message for Google
            full_prompt = f"{system_content}\n\nUser: {prompt}"

            if needs_search:
                # Configure with Google Search grounding
                config = types.GenerateContentConfig(
                    tools=[grounding_tool],
                    temperature=0.7,
                    max_output_tokens=min(max_tokens, 8192)
                )
            else:
                # Configure without grounding
                config = types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=min(max_tokens, 8192)
                )

            # Make the request
            response = google_client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=config
            )
            logger.info(f"Google API response: {response.text[:100]}...") 

            return response.text or "No response generated"

        elif provider == "navis":
            return self._generate_navis_ai_response(client, messages, max_tokens)
        else:
            return "No response generated"
    
    def _generate_navis_ai_response(self, client: Dict[str, Any], messages: List[Dict[str, str]], max_tokens: int) -> str:
        """Generate response using Navis AI runtime with Gemma model"""
        try:
            tokenizer = client["tokenizer"]
            model = client["model"]

            # Build prompt from messages including context
            system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
            user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            
            # Format prompt for Gemma model
            prompt = f"{system_msg}\nUser: {user_msg}\nAssistant:"
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs, 
                max_new_tokens=min(max_tokens, 2048),
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated text
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from the response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response or "No response generated"

        except Exception as e:
            logger.error(f"Navis AI runtime error: {str(e)}")
            return f"Navis AI runtime error: {str(e)}"


# =============================================================================
# NAVIS AGENT STORE CLASS 
# =============================================================================

class NavisAgentStore:
    """Thread-safe Filecoin storage for Navis Agents"""
    def __init__(self, s3_client=None, bucket: str = 'navis', key: str = 'agents.json'):
        self.s3 = s3_client
        self.bucket = bucket
        self.key = key
        self.lock = threading.RLock()
        self.agents: Dict[str, NavisAgent] = {}
        self.load_agents()

    def _validate_agent_data(self, agent_data: Dict[str, Any]) -> bool:
        """Validate agent data inline"""
        required_fields = ['name', 'model', 'profile', 'description']
        if not isinstance(agent_data, dict):
            return False

        for field in required_fields:
            if field not in agent_data or not isinstance(agent_data[field], str) or not agent_data[field].strip():
                return False

        if 'filecoin_cids' in agent_data:
            cids = agent_data['filecoin_cids']
            if not isinstance(cids, list) or not all(isinstance(cid, str) for cid in cids):
                return False

        return True

    def load_agents(self) -> bool:
        """Load all agents from S3 storage or local file"""
        with self.lock:
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=self.key)
                agents_data = json.loads(response['Body'].read().decode('utf-8'))

                # Convert to NavisAgent objects
                self.agents = {}
                for agent_data in agents_data:
                    agent = NavisAgent(**agent_data)
                    self.agents[agent.id] = agent

                logger.info(f"✅ Loaded {len(self.agents)} agents")
                return True
                
            except Exception as e:
                logger.warning(f"Starting with empty agent list: {e}")
                self.agents = {}
                return self.save_agents()

    def save_agents(self) -> bool:
        """Save all agents to S3 storage or local file"""
        with self.lock:
            try:
                # Convert NavisAgent objects to dicts
                agents_list = [agent.model_dump() for agent in self.agents.values()]
                                
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=self.key,
                    Body=json.dumps(agents_list, indent=2),
                    ContentType='application/json'
                )
                
                return True
            except Exception as e:
                logger.error(f"Error saving agents: {str(e)}")
                return False

    def get_agent(self, agent_id: str) -> Optional[NavisAgent]:
        """Get an agent by ID"""
        if not agent_id:
            return None
        with self.lock:
            return self.agents.get(agent_id)

    def list_agents(self) -> List[NavisAgent]:
        """List all agents"""
        with self.lock:
            return list(self.agents.values())

    def create_agent(self, name: str, model: str, profile: str,
                    description: str, filecoin_cids: Optional[List[str]] = None) -> NavisAgent:
        """Create a new agent and add it to the store"""
        agent_data = {
            'name': name,
            'model': model,
            'profile': profile,
            'description': description,
            'filecoin_cids': filecoin_cids or []
        }

        if not self._validate_agent_data(agent_data):
            raise ValueError("Invalid agent data")

        agent = NavisAgent(**agent_data)

        with self.lock:
            self.agents[agent.id] = agent
            if self.save_agents():
                return agent
            else:
                del self.agents[agent.id]
                raise ValueError("Failed to save agent")

    def update_agent(self, agent_id: str, **kwargs) -> bool:
        """Update specific fields of an existing agent"""
        if not agent_id:
            return False

        with self.lock:
            if agent_id not in self.agents:
                return False

            # Get current agent data as dict
            current_agent = self.agents[agent_id]
            updated_data = current_agent.model_dump()

            # Update with new values
            for key, value in kwargs.items():
                if key != 'id' and hasattr(current_agent, key):
                    updated_data[key] = value

            if not self._validate_agent_data(updated_data):
                return False

            # Create new agent with updated data
            original_agent = self.agents[agent_id]
            self.agents[agent_id] = NavisAgent(**updated_data)

            if self.save_agents():
                return True
            else:
                self.agents[agent_id] = original_agent
                return False

    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent by ID"""
        if not agent_id:
            return False

        with self.lock:
            if agent_id not in self.agents:
                return False

            removed_agent = self.agents[agent_id]
            del self.agents[agent_id]

            if self.save_agents():
                return True
            else:
                self.agents[agent_id] = removed_agent
                return False

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title="Navis Agents", version="1.0.0")

# Enable CORS for all origins (for quick setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for quick setup
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# FASTAPI ROUTES
# =============================================================================

@app.post('/api/v1/agents', status_code=201)
async def create_agent_route(request: Request):
    """Create a new agent. Agent ID assigned upon creation"""
    try:
        # Get JSON data
        agent_data = await request.json()

        if not agent_data:
            raise HTTPException(status_code=400, detail="Invalid JSON data")

        # Validate required fields
        required_fields = ['name', 'model', 'profile', 'description']
        for field in required_fields:
            if field not in agent_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        # Extract optional filecoin_cids
        filecoin_cids = agent_data.get('filecoin_cids')

        # Validate filecoin_cids if provided
        if filecoin_cids is not None and not isinstance(filecoin_cids, list):
            raise HTTPException(status_code=400, detail="filecoin_cids must be an array of strings")

        # Create the agent
        try:
            agent = agent_storage.create_agent(
                agent_data['name'],
                agent_data['model'],
                agent_data['profile'],
                agent_data['description'],
                filecoin_cids
            )

            logger.info(f"Navis Agent created successfully with ID: {agent.id}")
            return {"agent": agent.model_dump()}

        except ValueError as e:
            logger.error(f"Error creating agent: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating Navis Agent: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/api/v1/agents')
async def list_agents_route():
    """List all agents"""
    try:
        agents = agent_storage.list_agents()
        # Convert agents to dicts for JSON serialization
        agent_dicts = [agent.model_dump() for agent in agents]
        logger.debug(f"Found {len(agents)} Navis agents")
        return {"agents": agent_dicts}

    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/v1/agents/{agent_id}')
async def get_agent_route(agent_id: str):
    """Get an agent by ID"""
    try:
        agent = agent_storage.get_agent(agent_id)
        if agent is None:
            logger.warning(f"Agent with ID {agent_id} not found")
            raise HTTPException(status_code=404, detail="Agent not found")

        return {"agent": agent.model_dump()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.put('/api/v1/agents/{agent_id}')
async def update_agent_route(agent_id: str, request: Request):
    """Update an agent by ID"""
    try:
        # Get request data
        data = await request.json()
        if not data:
            raise HTTPException(status_code=400, detail="Invalid JSON data")

        # Get the existing agent
        agent = agent_storage.get_agent(agent_id)
        if agent is None:
            error_msg = "Agent not found"
            logger.warning(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)

        # Update agent using the corrected method signature
        success = agent_storage.update_agent(agent_id, **data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update agent")

        # Get the updated agent
        updated_agent = agent_storage.get_agent(agent_id)
        logger.info(f"Agent updated successfully with ID: {agent_id}")
        return {"agent": updated_agent.model_dump()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/v1/agents/{agent_id}')
async def delete_agent_route(agent_id: str):
    """Delete an agent by ID"""
    try:
        success = agent_storage.delete_agent(agent_id)
        if not success:
            logger.warning(f"Agent with ID {agent_id} not found")
            raise HTTPException(status_code=404, detail="Agent not found")

        return {"success": True}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/v1/agents/{agent_id}/chat')
async def agent_model_route(agent_id: str, request: Request):
    """Access Navis agent LLM model with prompt"""
    try:
        data = await request.json()
        prompt = data.get('prompt')
        max_tokens = data.get('max_tokens', 1024)

        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # Get the agent
        agent = agent_storage.get_agent(agent_id)
        if agent is None:
            logger.warning(f"Agent with ID {agent_id} not found")
            raise HTTPException(status_code=404, detail="Agent not found")

        # Generate response
        try:
            response = agent.generate_response(prompt, max_tokens)
            return {"result": response}

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error accessing agent model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running and healthy"""
    return {
        "status": "healthy",
        "version": "1.5.0",
        "service": "Navis AI Agents API"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Navis AI Agents API",
        "version": "1.0.0",
        "storage": "Akave O3",
        "ai_providers": sum(1 for p in ai_providers.values() if p),
        "endpoints": {
            "health": "/health",
            "agents": "/api/v1/agents",
            "chat": "/api/v1/agents/{agent_id}/chat"
        }
    }


# Initialize AI models(Google Gemini Pro 2.5 and Gemma 3 Open Source model)
setup_ai_providers()

# Akave O3 storage
agent_storage = NavisAgentStore(s3_client=s3, bucket='navis', key='agents.json')

uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)