> Historical working notes, preserved for context. Claims and commands below are not verified handoff instructions. Use the root README and docs/HANDOFF.md; relative links may refer to the original layout.

# 🚀 Deployment Guide - Tone Calibration System

Complete guide for deploying the full-stack Tone Calibration System to Netlify.

## 📋 Prerequisites

- GitHub repository with your code
- Netlify account (for both frontend and backend)
- Environment variables ready

## 🌐 Full-Stack Deployment (Netlify)

### Option 1: Netlify Functions (Recommended)

Netlify can host both your frontend and backend using Netlify Functions.

#### Step 1: Prepare Backend for Netlify Functions
```bash
# Create netlify/functions directory
mkdir -p netlify/functions

# Move your FastAPI app to functions
cp -r residualized_tone_space netlify/functions/
cp -r scripts netlify/functions/
```

#### Step 2: Create Netlify Function Handler
```javascript
// netlify/functions/api.js
const { spawn } = require('child_process');
const path = require('path');

exports.handler = async (event, context) => {
  // Handle CORS
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  };

  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers, body: '' };
  }

  // Run Python FastAPI app
  return new Promise((resolve) => {
    const python = spawn('python', [
      path.join(__dirname, 'residualized_tone_space/api/main.py')
    ]);

    let output = '';
    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.on('close', (code) => {
      resolve({
        statusCode: 200,
        headers,
        body: output
      });
    });
  });
};
```

#### Step 3: Configure Netlify
```toml
# netlify.toml
[build]
  base = "."
  publish = "frontend/build"
  command = "cd frontend && npm run build"

[build.environment]
  NODE_VERSION = "18"
  PYTHON_VERSION = "3.11"

[[redirects]]
  from = "/api/*"
  to = "/.netlify/functions/api"
  status = 200

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

### Option 2: Separate Netlify Sites

Deploy frontend and backend as separate Netlify sites.

#### Frontend Deployment
1. Go to [Netlify](https://netlify.com)
2. Click "New site from Git"
3. Connect your GitHub repository
4. Configure build settings:
   ```
   Build command: npm run build
   Publish directory: build
   Base directory: frontend
   ```

#### Backend Deployment
1. Create a separate Netlify site for the backend
2. Use Netlify Functions or serverless functions
3. Configure environment variables

## 🔧 Environment Variables

### Frontend Environment
```bash
# In Netlify dashboard
REACT_APP_API_URL=https://your-backend-site.netlify.app
```

### Backend Environment
```bash
# In Netlify dashboard
PINECONE_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_ANON_KEY=your_key
OPENAI_API_KEY=your_key
```

## 🐳 Docker Alternative (Railway)

If you prefer a traditional backend deployment:

### Railway Deployment
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Create project
railway init

# Set environment variables
railway variables set PINECONE_API_KEY=your_key
railway variables set SUPABASE_URL=your_url
railway variables set SUPABASE_ANON_KEY=your_key
railway variables set OPENAI_API_KEY=your_key

# Deploy
railway up
```

## 🔗 Connecting Frontend to Backend

### Update Frontend Environment
```bash
# In Netlify dashboard, update:
REACT_APP_API_URL=https://your-backend-site.netlify.app
```

### CORS Configuration
Ensure your backend allows requests from your frontend domain:

```python
# In FastAPI app
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📱 PWA Deployment

### HTTPS Requirement
- Netlify provides HTTPS by default
- Ensure all API calls use HTTPS
- Update service worker for production

### PWA Features
- **Installable**: Users can add to home screen
- **Offline**: Service worker caches resources
- **Responsive**: Works on all devices

## 🔒 Security Considerations

### Environment Variables
- Never commit API keys to repository
- Use Netlify environment variables for all secrets
- Rotate keys regularly

### HTTPS
- Required for PWA features
- Netlify provides free SSL certificates
- All API calls should use HTTPS

### CORS
- Configure CORS properly
- Only allow necessary origins
- Use credentials carefully

## 📊 Monitoring

### Netlify Dashboard
- Build logs in Netlify dashboard
- Analytics available
- Performance monitoring
- Function logs

### Custom Analytics
```bash
# Optional: Google Analytics
REACT_APP_GA_TRACKING_ID=your-ga-id
```

## 🚀 Performance Optimization

### Frontend
- Enable gzip compression (Netlify default)
- Optimize images
- Use Netlify CDN for static assets
- Implement lazy loading

### Backend (Netlify Functions)
- Optimize function cold starts
- Use connection pooling
- Implement caching
- Minimize dependencies

## 🔄 CI/CD Pipeline

### GitHub Actions with Netlify
```yaml
# .github/workflows/deploy.yml
name: Deploy to Netlify
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Netlify
        uses: nwtgck/actions-netlify@v1.2
        with:
          publish-dir: './frontend/build'
          production-branch: main
          github-token: ${{ secrets.GITHUB_TOKEN }}
          deploy-message: "Deploy from GitHub Actions"
```

## 🐛 Troubleshooting

### Common Issues

#### Frontend Build Fails
- Check Node.js version (18+)
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall

#### Backend Function Fails
- Check Python version (3.11+)
- Verify all dependencies
- Check environment variables
- Review function logs in Netlify dashboard

#### CORS Errors
- Verify CORS configuration
- Check allowed origins
- Ensure HTTPS for production

#### PWA Not Working
- Verify HTTPS is enabled
- Check manifest.json
- Test service worker

## 📞 Support

### Resources
- [Netlify Documentation](https://docs.netlify.com/)
- [Netlify Functions](https://docs.netlify.com/functions/overview/)
- [React PWA Guide](https://create-react-app.dev/docs/making-a-progressive-web-app/)

### Getting Help
- Check Netlify build logs
- Verify environment variables
- Test locally before deploying
- Use browser dev tools for frontend issues

---

**Your Tone Calibration System is now deployed on Netlify and ready to help users discover their optimal communication style!** 🎯
