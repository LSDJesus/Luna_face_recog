# RAGLight Agentic System - Founding Document
**Project Code Name**: COGNITIVE-CORE  
**Date**: October 1, 2025  
**Creator**: Brian (LSDJesus)  
**Purpose**: Multi-platform AI cognitive assistance with semantic memory

---

## 🎯 Project Vision

Build a **RAGLight-powered agentic RAG system** that provides intelligent, context-aware assistance across multiple platforms (VS Code, Chrome extensions, AI-Evolution) with persistent semantic memory and multi-modal understanding.

### Core Objectives
1. **Local RAG Server**: FastAPI-based service with RAGLight framework
2. **Multi-Platform Integration**: VS Code extension, Chrome extension, AI-Evolution connector
3. **Semantic Memory**: Persistent vector store with Chroma for conversation and code context
4. **Agentic Capabilities**: Tool-using agents for code analysis, web research, and task execution
5. **Qwen Integration**: Leverage existing Qwen2.5-VL models via Ollama

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  RAGLight Core Server                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   FastAPI    │  │   RAGLight   │  │    Chroma    │  │
│  │   Endpoints  │──│   Framework  │──│ Vector Store │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                  │                  │          │
│         └──────────────────┴──────────────────┘          │
│                         │                                │
│              ┌──────────┴──────────┐                     │
│              │   Ollama Backend    │                     │
│              │  (Qwen2.5-VL 7B)    │                     │
│              └─────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────────┐ ┌────▼─────┐ ┌────────▼──────┐
│  VS Code Ext   │ │  Chrome  │ │ AI-Evolution  │
│  (TypeScript)  │ │   Ext    │ │   Connector   │
└────────────────┘ └──────────┘ └───────────────┘
```

---

## 📋 Technology Stack

### Backend (RAGLight Server)
- **Framework**: RAGLight (lightweight RAG framework)
- **API**: FastAPI (Python async web framework)
- **Vector Store**: Chroma (embedded vector database)
- **LLM Backend**: Ollama (local model serving)
- **Models**: Qwen2.5-VL-7B-Instruct (already installed)
- **Embeddings**: Qwen2.5-VL MMProj or all-MiniLM-L6-v2
- **Database**: SQLite for metadata (optional)

### Client Integrations
- **VS Code Extension**: TypeScript, VS Code Extension API
- **Chrome Extension**: Manifest V3, JavaScript/TypeScript
- **AI-Evolution**: Python integration via REST API

### Development Tools
- **Python**: 3.10+ with venv
- **Node.js**: 18+ for VS Code/Chrome extensions
- **Docker**: Optional containerization
- **Git**: Version control

---

## 🚀 Phase 1: RAGLight Core Server (Week 1-2)

### Step 1: Project Setup
```bash
# Create new project directory
mkdir RAGLight-Cognitive-Core
cd RAGLight-Cognitive-Core

# Initialize git
git init
git remote add origin https://github.com/LSDJesus/RAGLight-Cognitive-Core.git

# Create Python environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install raglight fastapi uvicorn chromadb ollama pydantic python-dotenv
```

### Step 2: Project Structure
```
RAGLight-Cognitive-Core/
├── server/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic models for API
│   ├── raglight_engine.py   # RAGLight wrapper
│   ├── vector_store.py      # Chroma integration
│   └── agents/
│       ├── __init__.py
│       ├── code_agent.py    # Code analysis agent
│       ├── web_agent.py     # Web research agent
│       └── task_agent.py    # Task execution agent
├── clients/
│   ├── vscode-extension/    # VS Code extension
│   ├── chrome-extension/    # Chrome extension
│   └── ai-evolution/        # AI-Evolution connector
├── tests/
│   ├── test_server.py
│   └── test_agents.py
├── docs/
│   ├── API.md              # API documentation
│   └── SETUP.md            # Setup instructions
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE                 # AGPL-3.0
```

### Step 3: Core Implementation

#### `server/main.py` - FastAPI Server
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from .raglight_engine import RAGLightEngine
from .models import QueryRequest, QueryResponse, DocumentRequest

app = FastAPI(title="RAGLight Cognitive Core", version="1.0.0")
rag_engine = RAGLightEngine()

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint for RAG operations"""
    try:
        result = await rag_engine.query(
            query=request.query,
            context=request.context,
            mode=request.mode
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_document(request: DocumentRequest):
    """Ingest documents into vector store"""
    try:
        result = await rag_engine.ingest(
            content=request.content,
            metadata=request.metadata
        )
        return {"status": "success", "doc_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "engine": "raglight"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### `server/raglight_engine.py` - RAGLight Integration
```python
from raglight import RAGLight
from chromadb import Client
from chromadb.config import Settings
import ollama

class RAGLightEngine:
    def __init__(self):
        # Initialize Chroma
        self.chroma_client = Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./data/chroma"
        ))
        
        # Initialize RAGLight
        self.rag = RAGLight(
            llm_backend="ollama",
            model_name="qwen2.5-vl:7b",
            vector_store=self.chroma_client,
            embedding_model="all-MiniLM-L6-v2"
        )
    
    async def query(self, query: str, context: dict = None, mode: str = "default"):
        """Process query with RAG"""
        # Add context if provided
        if context:
            query = self._enrich_query(query, context)
        
        # Execute RAG query
        result = await self.rag.query(query, mode=mode)
        
        return {
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence
        }
    
    async def ingest(self, content: str, metadata: dict):
        """Ingest content into vector store"""
        doc_id = await self.rag.add_document(
            content=content,
            metadata=metadata
        )
        return doc_id
    
    def _enrich_query(self, query: str, context: dict) -> str:
        """Enrich query with contextual information"""
        enriched = f"{query}\n\nContext:\n"
        for key, value in context.items():
            enriched += f"- {key}: {value}\n"
        return enriched
```

#### `server/models.py` - Pydantic Models
```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    mode: str = "default"
    max_sources: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

class DocumentRequest(BaseModel):
    content: str
    metadata: Dict[str, Any]
```

---

## 🔧 Phase 2: VS Code Extension (Week 3-4)

### Setup
```bash
cd clients
npx yo code  # Yeoman generator for VS Code extensions
# Choose: New Extension (TypeScript)
# Name: raglight-assistant
```

### Key Features
1. **Context Menu**: Right-click code → "Ask RAGLight"
2. **Sidebar Panel**: Chat interface with RAG
3. **Code Analysis**: Automatic context extraction
4. **Inline Suggestions**: Code completion with RAG

### Extension Structure
```
vscode-extension/
├── src/
│   ├── extension.ts         # Main extension entry
│   ├── raglight-client.ts   # API client
│   ├── chat-panel.ts        # Chat UI provider
│   └── code-context.ts      # Context extraction
├── package.json
└── tsconfig.json
```

---

## 🌐 Phase 3: Chrome Extension (Week 5)

### Manifest V3 Setup
```json
{
  "manifest_version": 3,
  "name": "RAGLight Web Assistant",
  "version": "1.0.0",
  "permissions": ["activeTab", "storage"],
  "background": {
    "service_worker": "background.js"
  },
  "action": {
    "default_popup": "popup.html"
  }
}
```

### Features
1. **Page Context**: Extract webpage content for RAG
2. **Quick Queries**: Popup interface for questions
3. **Annotations**: Highlight and query text
4. **Research Mode**: Automatic fact-checking

---

## 🤖 Phase 4: Agentic Capabilities (Week 6-7)

### Agent Framework
```python
# server/agents/base_agent.py
class BaseAgent:
    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.tools = []
    
    async def execute(self, task: str) -> dict:
        """Execute agent task"""
        raise NotImplementedError

# server/agents/code_agent.py
class CodeAgent(BaseAgent):
    """Agent for code analysis and generation"""
    
    async def execute(self, task: str) -> dict:
        # Analyze code structure
        # Generate solutions
        # Provide explanations
        pass

# server/agents/web_agent.py
class WebAgent(BaseAgent):
    """Agent for web research"""
    
    async def execute(self, task: str) -> dict:
        # Search web
        # Extract information
        # Synthesize results
        pass
```

---

## 📊 Phase 5: Integration & Testing (Week 8)

### Testing Strategy
1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test API endpoints
3. **E2E Tests**: Test full workflows
4. **Performance Tests**: Measure latency and throughput

### Deployment
```bash
# Local development
uvicorn server.main:app --reload

# Production (Docker)
docker build -t raglight-core .
docker run -p 8000:8000 raglight-core
```

---

## 🎯 Success Metrics

### Technical Metrics
- **Response Time**: <500ms for queries
- **Accuracy**: >85% relevance in retrieved documents
- **Uptime**: 99%+ availability

### User Metrics
- **Daily Active Users**: 100+ (Month 3)
- **Queries/Day**: 1000+ (Month 6)
- **User Satisfaction**: 4.5/5 stars

---

## 📚 Key Resources

### RAGLight
- GitHub: https://github.com/raglight/raglight
- Docs: https://raglight.readthedocs.io

### FastAPI
- Docs: https://fastapi.tiangolo.com

### Chroma
- Docs: https://docs.trychroma.com

### Ollama
- Docs: https://ollama.ai/docs

### VS Code Extension API
- Docs: https://code.visualstudio.com/api

---

## 🚀 Quick Start Commands

```bash
# 1. Create project
mkdir RAGLight-Cognitive-Core && cd RAGLight-Cognitive-Core

# 2. Setup Python environment
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install raglight fastapi uvicorn chromadb ollama

# 3. Create basic structure
mkdir -p server/agents clients tests docs

# 4. Start Ollama (if not running)
ollama serve

# 5. Run server (once implemented)
uvicorn server.main:app --reload
```

---

## 🎓 Learning Path

1. **Week 1**: RAGLight basics + FastAPI fundamentals
2. **Week 2**: Chroma vector store + embedding strategies
3. **Week 3**: VS Code extension development
4. **Week 4**: Chrome extension patterns
5. **Week 5**: Agentic frameworks (LangChain/AutoGPT patterns)
6. **Week 6**: Multi-modal processing with Qwen
7. **Week 7**: Optimization and scaling
8. **Week 8**: Deployment and monitoring

---

## 💡 Next Steps for Fresh Chat

When starting your new chat, paste this and say:

> "I'm Brian, creator of the VAPOR-FACE project. I want to build the RAGLight Cognitive Core system outlined in this founding document. Let's start with Phase 1: setting up the RAGLight server with FastAPI. I have Ollama with Qwen2.5-VL-7B already running locally. Help me create the initial project structure and implement the core server components."

Then provide:
1. Your Python version
2. Your operating system
3. Your preferred code editor
4. Any specific requirements or constraints

---

## 📝 Notes

- **AGPL-3.0 License**: Same dual licensing strategy as VAPOR-FACE
- **Token Efficiency**: Fresh project = minimal context
- **Incremental Development**: Build and test each phase
- **Documentation**: Document as you build
- **Community**: Consider open-sourcing for community contributions

---

**Document Version**: 1.0  
**Last Updated**: October 1, 2025  
**Status**: Ready for Implementation

---

## Appendix A: Environment Variables

```bash
# .env.example
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5-vl:7b
CHROMA_PERSIST_DIR=./data/chroma
EMBEDDING_MODEL=all-MiniLM-L6-v2
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## Appendix B: Requirements.txt

```txt
raglight>=0.1.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
chromadb>=0.4.15
ollama>=0.1.0
pydantic>=2.4.0
python-dotenv>=1.0.0
aiohttp>=3.9.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.10.0
ruff>=0.1.0
```

---

**Ready to revolutionize your AI workflow!** 🚀