# MESS-AI Architecture

## Service Separation

The project now has a clean separation between backend API and ML pipeline:

```
mess-ai/
├── backend/           # FastAPI web service (port 8000)
│   ├── api/          # REST endpoints for web/frontend
│   └── services/     # Pipeline client communication
├── pipeline/         # ML processing service (port 8001)  
│   ├── api/          # Simple ML API wrapper
│   ├── extraction/   # MERT feature extraction
│   ├── probing/      # Layer discovery & validation
│   └── query/        # Recommendation engines
└── frontend/         # React UI (port 3000)
```

## Key Components

### Backend Service (Port 8000)
- **Purpose**: Web API, metadata, audio serving
- **Dependencies**: Lightweight, web-focused
- **Communication**: Calls pipeline service via HTTP

### Pipeline Service (Port 8001)  
- **Purpose**: ML processing, MERT models, recommendations
- **Dependencies**: PyTorch, transformers, scikit-learn
- **Endpoints**:
  - `POST /recommend` - Get track recommendations  
  - `POST /query` - Natural language queries
  - `GET /health` - Service health

### Validated ML System
- **Layer 0**: Spectral brightness (R² = 0.944) - Best for timbral similarity
- **Layer 1**: Timbral texture (R² = 0.922)
- **Layer 2**: Acoustic structure (R² = 0.933)

## Benefits

**Independent Scaling**: ML service runs on GPU instances, web service on CPU
**Clean Separation**: Different update cycles, dependencies, and responsibilities  
**Service Communication**: Simple HTTP API between services
**Development Focus**: ML engineers work on pipeline, web engineers on backend/frontend

## Usage

```bash
# Start pipeline service
cd pipeline && python api/main.py

# Start backend service  
cd backend && python -m api.main

# Start frontend
cd frontend && npm start
```

The pipeline service loads the empirically validated MERT layer mappings and provides intelligent music similarity search through both direct recommendations and natural language queries.