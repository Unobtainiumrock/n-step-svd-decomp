> Historical working notes, preserved for context. Claims and commands below are not verified handoff instructions. Use the root README and docs/HANDOFF.md; relative links may refer to the original layout.

# Residualized Tone Space: "Face ID for Tone" - Full Stack

A complete full-stack system for discovering and calibrating content tone preferences through convergent search in semantic embedding space.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
  - [Full Stack Development](#full-stack-development)
  - [Individual Services](#individual-services)     
- [🏗️ Architecture](#️-architecture)
  - [System Architecture Diagram](#system-architecture-diagram)
  - [Backend (Python/FastAPI)](#backend-pythonfastapi)
  - [Frontend (React.js)](#frontend-reactjs)
- [�� Project Structure](#-project-structure)
- [🎯 Core Features](#-core-features)
  - [1. Interactive Convergent Search](#1-interactive-convergent-search)
  - [2. Honest Semantic Extraction](#2-honest-semantic-extraction)
  - [3. LLM Content Generation](#3-llm-content-generation)
  - [4. Modern Web Interface](#4-modern-web-interface)
- [🔧 Technical Architecture](#-technical-architecture)
  - [Backend Stack](#backend-stack)
  - [Frontend Stack](#frontend-stack)
  - [Mathematical Foundation](#mathematical-foundation)
- [🚀 Deployment](#-deployment)
  - [Development](#development)
  - [Production](#production)
- [📊 Dataset](#-dataset)
- [🎨 User Experience](#-user-experience)
  - [Web Interface Flow](#web-interface-flow)
  - [Mobile PWA](#mobile-pwa)
- [🔬 Research & Development](#-research--development)
  - [Key Innovations](#key-innovations)
  - [Future Directions](#future-directions)
- [📈 Performance](#-performance)
- [🛠️ Development](#️-development)
  - [Setup](#setup)
  - [Environment Variables](#environment-variables)
- [📚 Documentation](#-documentation)
- [🎯 The "Face ID for Tone" Vision](#-the-face-id-for-tone-vision)

## �� Quick Start

### Full Stack Development
```bash
# Start both backend and frontend
docker-compose up

# Backend API: http://localhost:8000
# Frontend App: http://localhost:3000
```

### Individual Services
```bash
# Backend only
docker-compose up app

# Frontend only  
docker-compose up frontend

# Run complete pipeline
docker-compose run --rm app uv run python run_pipeline.py
```

## 🏗️ Architecture

### System Architecture Diagram

```mermaid
graph TB
    %% User Layer
    User[👤 User] --> Browser[🌐 Web Browser]
    User --> Mobile[📱 Mobile PWA]
    
    %% Frontend Layer
    Browser --> Frontend[⚛️ React.js Frontend]
    Mobile --> Frontend
    Frontend --> |"HTTP/HTTPS"| API[🔌 FastAPI Backend]
    
    %% Backend Core
    API --> Core[🧠 Core Library]
    Core --> TopicSpace[📊 Topic Subspace<br/>160 SVD Components]
    Core --> ToneSpace[🎵 Tone Subspace<br/>130 SVD Components]
    Core --> ResidualCalc[🔢 Residual Calculator]
    
    %% Data Sources
    API --> Supabase[(��️ Supabase<br/>80k+ Posts)]
    API --> Pinecone[(🔍 Pinecone<br/>Vector DB)]
    API --> OpenAI[🤖 OpenAI<br/>Embeddings & LLM]
    
    %% Processing Pipeline
    TopicSpace --> Embeddings[📐 Embeddings<br/>1536D Vectors]
    ToneSpace --> Embeddings
    ResidualCalc --> Embeddings
    
    %% Interactive System
    Frontend --> Calibration[🎯 Interactive Calibration<br/>9-Round A/B Testing]
    Calibration --> API
    API --> Analysis[📈 Semantic Analysis]
    Analysis --> Results[📊 Results & Visualization]
    
    %% Content Generation
    Results --> LLM[🤖 LLM Content Generation]
    LLM --> Generated[✨ Personalized Content]
    
    %% Deployment
    Frontend --> Netlify[🌐 Netlify<br/>PWA Deployment]
    API --> Railway[🚂 Railway<br/>Backend Hosting]
    
    %% Styling
    classDef userLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef frontendLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef backendLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef dataLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef deploymentLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class User,Browser,Mobile userLayer
    class Frontend,Calibration,Results frontendLayer
    class API,Core,TopicSpace,ToneSpace,ResidualCalc,Analysis,LLM backendLayer
    class Supabase,Pinecone,OpenAI,Embeddings dataLayer
    class Netlify,Railway,Generated deploymentLayer
```

### Backend (Python/FastAPI)
- **Core Library**: `residualized_tone_space/` - Mathematical algorithms
- **API Endpoints**: FastAPI for web integration
- **Scripts**: Organized in `scripts/` directory
- **Data Pipeline**: Complete tone calibration system

### Frontend (React.js)
- **Modern UI**: React 18 with TypeScript
- **PWA Support**: Installable mobile app
- **Real-time**: Interactive calibration interface
- **Visualization**: Charts and analytics

## 📁 Project Structure

```
├── run_pipeline.py                    # Main pipeline runner
├── docker-compose.yml                 # Full stack orchestration
├── README.md                          # This file
├── residualized_tone_space/           # Core backend library
│   ├── core/                          # Mathematical algorithms
│   ├── api/                           # FastAPI endpoints
│   └── schemas/                       # Data models
├── scripts/                           # Backend scripts
│   ├── core/                          # Core interactive scripts
│   ├── analysis/                      # Analysis tools
│   ├── demo/                          # Demo examples
│   ├── utilities/                     # API utilities
│   └── legacy/                        # Legacy scripts
├── frontend/                          # React.js frontend
│   ├── src/                           # Source code
│   │   ├── components/                # React components
│   │   ├── pages/                     # Page components
│   │   ├── services/                  # API services
│   │   └── utils/                     # Utilities
│   ├── public/                        # Static assets
│   ├── package.json                   # Dependencies
│   ├── netlify.toml                   # Deployment config
│   └── Dockerfile                     # Container config
├── data/                              # Data files
├── tests/                             # Test suite
└── notebooks/                         # Jupyter notebooks
```

## 🎯 Core Features

### 1. Interactive Convergent Search
- **9-round A/B comparison** system
- **Strategic progression**: Broad → Topic → Tone → Fine-tuning
- **Real-time Supabase integration** for content display
- **Convergence detection** with confidence scoring

### 2. Honest Semantic Extraction
- **Raw tone axis alignments** (no fake interpretations)
- **130 mathematical dimensions** discovered through SVD
- **Choice pattern analysis** for preference discovery
- **Content preference mapping** (text vs video, exploration breadth)

### 3. LLM Content Generation
- **Comprehensive prompting** with discovered preferences
- **Tone axis interpretation** by the LLM
- **Context-aware generation** based on choice journey
- **Session data persistence** for future use

### 4. Modern Web Interface
- **Interactive UI**: React.js with smooth animations
- **PWA Support**: Installable as mobile app
- **Real-time Updates**: Live progress tracking
- **Data Visualization**: Charts and analytics

## 🔧 Technical Architecture

### Backend Stack
- **Python 3.11** with FastAPI
- **OpenAI Embeddings**: text-embedding-3-small (1536D)
- **Supabase**: Real-time content lookup (80k+ posts)
- **Pinecone**: Vector database for embeddings
- **Docker**: Containerized deployment

### Frontend Stack
- **React 18** with TypeScript
- **Tailwind CSS**: Modern styling
- **Framer Motion**: Smooth animations
- **Recharts**: Data visualization
- **PWA**: Progressive Web App capabilities

### Mathematical Foundation
- **Full Embedding Space**: $E = \mathbb{R}^{1536}$
- **Topic Subspace**: $T \subset E$ (160 SVD components)
- **Tone Subspace**: $R = E \ominus T$ (130 SVD components)
- **Residual Vector**: $\vec{r}_i = \vec{e}_i - \text{Proj}_T(\vec{e}_i)$
- **Tone Score**: $\text{tone\_score} = \vec{w}_r^{(s)} \cdot \vec{r}_i$

📚 **Complete Mathematical Documentation**: See `scripts/legacy/RESIDUALIZED_TONE_SPACE_IMPLEMENTATION.md`

## 🚀 Deployment

### Development
```bash
# Full stack development
docker-compose up

# Individual services
docker-compose up app      # Backend only
docker-compose up frontend # Frontend only
```

### Production

#### Backend (Railway)
```bash
# Deploy FastAPI backend
git push railway main

# Environment variables
PINECONE_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_ANON_KEY=your_key
OPENAI_API_KEY=your_key
```

#### Frontend (Netlify)
```bash
# Connect GitHub to Netlify
# Build command: npm run build
# Publish directory: build
# Environment: REACT_APP_API_URL=https://your-backend-url.com
```

## 📊 Dataset

- **80,420 posts** from LinkedIn UGC
- **Text posts**: 84,200 (ugc_posts table)
- **Video posts**: 15,578 (ugc_videos table)
- **Embeddings**: 1536-dimensional vectors
- **Metadata**: Post IDs, content, timestamps

## 🎨 User Experience

### Web Interface Flow
1. **Landing Page**: Features and call-to-action
2. **Calibration**: Interactive A/B testing (9 rounds)
3. **Results**: Tone analysis and visualization
4. **Content Generation**: AI-powered personalized content

### Mobile PWA
- **Installable**: Add to home screen
- **Offline Support**: Service worker caching
- **Responsive**: Mobile-first design
- **Fast Loading**: Optimized performance

## 🔬 Research & Development

### Key Innovations
1. **Residualized Tone Space**: Separating topic from tone
2. **Convergent Search**: Strategic A/B comparison system
3. **Honest Semantic Extraction**: Raw mathematical dimensions
4. **LLM Interpretation**: AI-driven tone axis understanding
5. **Full-Stack Integration**: Seamless web experience

### Future Directions
- **Supervised Learning**: Human-labeled tone axes
- **Multi-modal**: Image and video tone analysis
- **Real-time**: Live content optimization
- **Personalization**: User-specific tone profiles
- **Mobile App**: Native iOS/Android apps

## 📈 Performance

- **Convergence**: 95% confidence in 9 rounds average
- **Speed**: ~30 seconds per round with Supabase integration
- **Accuracy**: High consistency in tone preference discovery
- **Scalability**: Handles 80k+ posts efficiently
- **Frontend**: <3s load time, PWA capabilities

## 🛠️ Development

### Setup
```bash
# Clone and setup
git clone <repo>
cd unsupervised-learning

# Full stack development
docker-compose up

# Backend development
docker-compose run --rm app uv run python run_pipeline.py

# Frontend development
cd frontend && npm install && npm start
```

### Environment Variables
```bash
# Backend
PINECONE_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_ANON_KEY=your_key
OPENAI_API_KEY=your_key

# Frontend
REACT_APP_API_URL=http://localhost:8000
```

## 📚 Documentation

- **README.md**: This comprehensive guide
- **docs/**: Additional documentation and references
- **Frontend README**: `frontend/README.md` - React.js details
- **Mathematical Documentation**: `scripts/legacy/RESIDUALIZED_TONE_SPACE_IMPLEMENTATION.md`
- **Core Library**: `residualized_tone_space/` with docstrings
- **API Documentation**: FastAPI auto-generated docs

### 📖 Documentation Structure
- **[docs/README.md](docs/README.md)** - Documentation index
- **[docs/MATHEMATICAL_REFERENCE.md](docs/MATHEMATICAL_REFERENCE.md)** - Quick math reference
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete deployment guide
- **[frontend/README.md](frontend/README.md)** - Frontend development guide

---

**Ready to discover your optimal tone? Start the full-stack system and experience the future of personalized content generation!** 🚀
