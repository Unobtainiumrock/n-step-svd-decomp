> Historical frontend notes. Use the [root README](../README.md) and
> [handoff](../docs/HANDOFF.md) for current setup and known limitations.

# Tone Calibration Frontend

A modern React.js frontend for the Tone Calibration System - "Face ID for Tone".

## 🚀 Features

- **Interactive Calibration**: 9-round A/B comparison system
- **Real-time Results**: Dynamic tone axis visualization
- **PWA Support**: Installable as a mobile app
- **Responsive Design**: Works on all devices
- **Modern UI**: Built with Tailwind CSS and Framer Motion

## 🛠️ Tech Stack

- **React 18** with TypeScript
- **React Router** for navigation
- **Tailwind CSS** for styling
- **Framer Motion** for animations
- **Recharts** for data visualization
- **Axios** for API communication
- **PWA** capabilities

## 📱 PWA Features

- **Offline Support**: Service worker caching
- **Installable**: Add to home screen
- **Responsive**: Mobile-first design
- **Fast Loading**: Optimized bundle size

## 🚀 Quick Start

### Development
```bash
# Install dependencies
npm install

# Start development server
npm start

# Open http://localhost:3000
```

### Docker
```bash
# Build and run with Docker
docker-compose up frontend

# Or run directly
docker build -t tone-frontend .
docker run -p 3000:3000 tone-frontend
```

### Production Build
```bash
# Build for production
npm run build

# Serve locally
npx serve -s build
```

## 🌐 Deployment

### Netlify (Recommended)
1. Connect your GitHub repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `build`
4. Deploy!

### Environment Variables
```bash
# API Configuration
REACT_APP_API_URL=https://your-backend-url.com
```

## 📁 Project Structure

```
frontend/
├── public/
│   ├── index.html
│   ├── manifest.json      # PWA manifest
│   └── sw.js             # Service worker
├── src/
│   ├── components/       # Reusable components
│   │   └── layout/       # Layout components
│   ├── pages/           # Page components
│   ├── services/        # API services
│   ├── utils/           # Utility functions
│   ├── hooks/           # Custom React hooks
│   └── assets/          # Static assets
├── package.json
├── tailwind.config.js   # Tailwind configuration
├── tsconfig.json        # TypeScript configuration
├── netlify.toml         # Netlify deployment config
└── Dockerfile           # Docker configuration
```

## 🎨 Pages

- **HomePage**: Landing page with features and CTA
- **CalibrationPage**: Interactive A/B testing interface
- **ResultsPage**: Tone analysis and visualization
- **AboutPage**: Technical details and documentation

## 🔧 API Integration

The frontend communicates with the backend through:

- **Calibration API**: Start sessions, submit choices
- **Results API**: Get tone analysis and visualizations
- **Content API**: Generate personalized content
- **Posts API**: Fetch real post content from Supabase

## 📱 PWA Configuration

- **Manifest**: App metadata and icons
- **Service Worker**: Offline caching and updates
- **Responsive**: Mobile-first design
- **Installable**: Add to home screen support

## 🚀 Performance

- **Code Splitting**: Lazy loading of components
- **Bundle Optimization**: Tree shaking and minification
- **Caching**: Service worker for offline support
- **Images**: Optimized and responsive

## 🔒 Security

- **HTTPS**: Required for PWA features
- **CSP**: Content Security Policy headers
- **Environment Variables**: Secure API configuration
- **Input Validation**: Client-side validation

## 📊 Analytics

Optional Google Analytics integration:

```bash
REACT_APP_GA_TRACKING_ID=your-ga-id
```

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm run test:coverage
```

## 📦 Build Optimization

- **Tree Shaking**: Remove unused code
- **Code Splitting**: Lazy load routes
- **Bundle Analysis**: Analyze bundle size
- **Compression**: Gzip compression

## 🌍 Internationalization

Ready for i18n with react-i18next:

```bash
npm install react-i18next i18next
```

## 📱 Mobile Features

- **Touch Gestures**: Swipe navigation
- **Haptic Feedback**: Vibration API
- **Camera Access**: Photo upload (future)
- **Geolocation**: Location-based features (future)

---

**Ready to discover your tone? Start the frontend and begin your calibration journey!** 🎯
