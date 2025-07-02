# mess-ai Frontend

React-based web interface for music similarity search.

## Features

- Interactive music player
- Real-time similarity search
- Waveform visualization
- Responsive design with Tailwind CSS

## Development

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

## API Integration

Configure the backend API URL:

```javascript
// In development
const API_URL = 'http://localhost:8000'

// In production
const API_URL = 'https://your-api-domain.com'
```

## Deployment

This frontend is designed to be deployed separately from the backend:

- **Development**: Local dev server (`npm start`)
- **Production**: Static files on S3 + CloudFront CDN

## Architecture

The frontend communicates with the backend microservice via REST API endpoints:

- `GET /tracks` - List available tracks
- `GET /recommend/{track_id}` - Get similar tracks
- `GET /audio/{track_id}` - Stream audio
- `GET /waveform/{track_id}` - Get waveform data

## Technology Stack

- React 18
- TypeScript
- Tailwind CSS
- Modern async/await patterns