> Historical working notes, preserved for context. Claims and commands below are not verified handoff instructions. Use the root README and docs/HANDOFF.md; relative links may refer to the original layout.

# 🚀 Full-Stack Development Guide

Complete step-by-step guide for running the Tone Calibration System in development mode.

## 📋 Prerequisites

### Required Environment Variables
Create a `.env` file in the root directory:
```bash
# API Keys (required)
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
```

### Required Software
- Docker and Docker Compose
- Git (for version control)

## 🎯 Development Commands (In Order)

### Step 1: Clone and Setup
```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd unsupervised-learning

# Create environment file
cp .env.example .env
# Edit .env with your actual API keys
```

### Step 2: Start Full-Stack Development
```bash
# Start all services (backend + frontend + redis + jupyter)
docker-compose up

# Or start in background
docker-compose up -d
```

### Step 3: Access Your Applications
- **Frontend (React.js)**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Jupyter Notebooks**: http://localhost:8888
- **Redis**: localhost:6379

## 🔧 Individual Service Commands

### Backend Only
```bash
# Start just the backend services
docker-compose up app redis

# Or run backend directly (without Docker)
uv run uvicorn residualized_tone_space.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Only
```bash
# Start just the frontend
docker-compose up frontend

# Or run frontend directly (without Docker)
cd frontend
npm install
npm start
```

### Jupyter Notebooks
```bash
# Start Jupyter for data analysis
docker-compose up jupyter
# Access at: http://localhost:8888
```

## 🛠️ Development Workflow

### 1. First Time Setup
```bash
# 1. Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# 2. Start all services
docker-compose up

# 3. Verify everything is running
curl http://localhost:8000/health  # Backend health check
curl http://localhost:3000         # Frontend should load
```

### 2. Daily Development
```bash
# Start development environment
docker-compose up

# Make changes to code (auto-reload enabled)
# Backend: Changes in Python files auto-reload
# Frontend: Changes in React files auto-reload

# View logs
docker-compose logs -f app        # Backend logs
docker-compose logs -f frontend   # Frontend logs
```

### 3. Running the Complete Pipeline
```bash
# Run the complete tone calibration pipeline
docker-compose run --rm app uv run python run_pipeline.py

# Or run specific scripts
docker-compose run --rm app uv run python scripts/core/complete_pipeline_honest.py
```

## 🔍 Debugging Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f frontend
docker-compose logs -f redis
```

### Access Containers
```bash
# Access backend container
docker-compose exec app bash

# Access frontend container
docker-compose exec frontend sh

# Run commands in containers
docker-compose exec app uv run python --version
docker-compose exec frontend npm --version
```

### Restart Services
```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart app
docker-compose restart frontend
```

## 🧪 Testing Commands

### Backend Testing
```bash
# Run backend tests
docker-compose run --rm app uv run pytest

# Run specific test file
docker-compose run --rm app uv run pytest tests/test_api.py
```

### Frontend Testing
```bash
# Run frontend tests
docker-compose exec frontend npm test

# Run frontend build test
docker-compose exec frontend npm run build
```

## 🚀 Production Commands

### Build for Production
```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build app
docker-compose build frontend
```

### Production Deployment
```bash
# Use production docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

## 🛑 Stop Commands

### Stop All Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop and remove everything
docker-compose down --rmi all -v
```

## 📊 Service Status

### Check Running Services
```bash
# List running containers
docker-compose ps

# Check service health
curl http://localhost:8000/health  # Backend
curl http://localhost:3000         # Frontend
```

## 🔧 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
lsof -i :3000  # Frontend port
lsof -i :8000  # Backend port

# Kill process using port
sudo kill -9 <PID>
```

#### Docker Issues
```bash
# Clean up Docker
docker system prune -a

# Rebuild containers
docker-compose build --no-cache
```

#### Environment Variables
```bash
# Check environment variables
docker-compose exec app env | grep API_KEY

# Reload environment
docker-compose down
docker-compose up
```

## 📱 Development URLs

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | React.js app |
| Backend API | http://localhost:8000 | FastAPI server |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Jupyter | http://localhost:8888 | Data analysis |
| Redis | localhost:6379 | Cache server |

## 🎯 Quick Start Summary

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start everything
docker-compose up

# 3. Open in browser
# Frontend: http://localhost:3000
# Backend: http://localhost:8000/docs

# 4. Start developing!
# Changes auto-reload in both frontend and backend
```

---

**Happy coding! Your full-stack Tone Calibration System is ready for development!** 🚀
