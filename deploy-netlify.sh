#!/bin/bash

# Netlify Deployment Script for Tone Calibration System

echo "🚀 Deploying Tone Calibration System to Netlify..."

# Check if Netlify CLI is installed
if ! command -v netlify &> /dev/null; then
    echo "❌ Netlify CLI not found. Installing..."
    npm install -g netlify-cli
fi

# Login to Netlify (if not already logged in)
echo "🔐 Checking Netlify authentication..."
if ! netlify status &> /dev/null; then
    echo "Please login to Netlify:"
    netlify login
fi

# Build the frontend
echo "📦 Building frontend..."
cd frontend
npm install
npm run build
cd ..

# Deploy to Netlify
echo "🌐 Deploying to Netlify..."
netlify deploy --prod --dir=frontend/build

echo "✅ Deployment complete!"
echo "🌍 Your app is now live on Netlify!"
echo "📱 PWA features are enabled"
echo "🔒 HTTPS is automatically configured"
