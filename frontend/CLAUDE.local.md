# Frontend Development Instructions

## React 19 + TypeScript Patterns

### Component Architecture
- Use functional components with hooks
- Implement proper TypeScript interfaces for props
- Follow single responsibility principle
- Use composition over inheritance

### State Management
```typescript
// Use React hooks for local state
const [tracks, setTracks] = useState<TrackMetadata[]>([]);
const [loading, setLoading] = useState<boolean>(false);
const [error, setError] = useState<string | null>(null);

// Use context for shared state
const AudioContext = createContext<AudioContextType | null>(null);
```

### API Integration Patterns
```typescript
// Reference: backend/CLAUDE.local.md for endpoint details
// Reference: CLAUDE.md for data flow architecture

// Error handling with retry logic
const fetchTracks = async (): Promise<TracksResponse> => {
  try {
    setLoading(true);
    setError(null);
    const response = await apiClient.getTracks();
    return response;
  } catch (error) {
    console.error('Failed to load tracks:', error);
    setError('Failed to load tracks. Is the API server running?');
    throw error;
  } finally {
    setLoading(false);
  }
};

// Loading states for better UX
{isLoading ? (
  <div className="flex justify-center">Loading...</div>
) : (
  <TrackList tracks={tracks} />
)}
```

## Development Commands

### Start Development Server
```bash
cd frontend
npm install
npm start
# Runs on http://localhost:3000
```

### Testing
```bash
# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run tests with coverage
npm test -- --coverage
```

### Build
```bash
# Development build
npm run build

# Production build (optimized)
npm run build:prod
```

## Component Guidelines

### File Structure
```
src/
├── components/
│   ├── ui/              # Reusable UI components
│   ├── MusicPlayer.tsx  # Feature-specific components
│   └── TrackList.tsx
├── lib/
│   ├── api.ts          # API client
│   └── utils.ts        # Utility functions
├── hooks/              # Custom React hooks
└── types/              # TypeScript type definitions
```

### TypeScript Conventions
```typescript
// Use interfaces for props
interface TrackListProps {
  tracks: TrackMetadata[];
  currentTrack: string | null;
  onTrackSelect: (trackId: string) => void;
  isLoading: boolean;
}

// Use proper typing for API responses
interface ApiResponse<T> {
  data: T;
  error?: string;
  loading: boolean;
}
```

## API Integration

### Data Flow Integration
- **Reference**: Global CLAUDE.md for overall system architecture
- **Reference**: backend/CLAUDE.local.md for specific endpoint patterns

### Error Handling Strategy
```typescript
// Centralized error handling
const handleApiError = (error: unknown, fallbackMessage: string) => {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.message || fallbackMessage;
  }
  return fallbackMessage;
};

// User-friendly error messages
const errorMessages = {
  NETWORK_ERROR: 'Network connection failed. Please check your connection.',
  SERVER_ERROR: 'Server error. Please try again later.',
  NOT_FOUND: 'Resource not found.',
  UNAUTHORIZED: 'Authentication required.',
};
```

### API Client Configuration
```typescript
// Reference: backend/CLAUDE.local.md for endpoint details
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Axios interceptors for global error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Global error handling logic
    return Promise.reject(error);
  }
);
```

## UI/UX Guidelines

### Design System
- Use consistent color palette and typography
- Implement responsive design patterns
- Follow accessibility guidelines (WCAG 2.1)
- Use loading states for all async operations

### Component Patterns
```typescript
// Loading component pattern
const LoadingSpinner = () => (
  <div className="flex justify-center items-center p-4">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
  </div>
);

// Error boundary pattern
class ErrorBoundary extends React.Component {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: any) {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return <div>Something went wrong.</div>;
    }
    return this.props.children;
  }
}
```

## Performance Optimization

### Code Splitting
```typescript
// Lazy load components
const LazyComponent = React.lazy(() => import('./HeavyComponent'));

// Use Suspense for loading states
<Suspense fallback={<LoadingSpinner />}>
  <LazyComponent />
</Suspense>
```

### Memoization
```typescript
// Memoize expensive calculations
const expensiveValue = useMemo(() => {
  return computeExpensiveValue(data);
}, [data]);

// Memoize callbacks
const handleClick = useCallback((id: string) => {
  onTrackSelect(id);
}, [onTrackSelect]);
```

## Testing Patterns

### Component Testing
```typescript
// Use React Testing Library
import { render, screen, fireEvent } from '@testing-library/react';

test('renders track list', () => {
  render(<TrackList tracks={mockTracks} />);
  expect(screen.getByText('Track 1')).toBeInTheDocument();
});

// Mock API calls
jest.mock('../lib/api', () => ({
  apiClient: {
    getTracks: jest.fn().mockResolvedValue({ tracks: mockTracks }),
  },
}));
```

### Integration Testing
```typescript
// Test user workflows
test('user can select and play track', async () => {
  render(<App />);
  
  // Wait for tracks to load
  await waitFor(() => {
    expect(screen.getByText('Track 1')).toBeInTheDocument();
  });
  
  // Click on track
  fireEvent.click(screen.getByText('Track 1'));
  
  // Verify track is selected
  expect(screen.getByText('Now Playing: Track 1')).toBeInTheDocument();
});
```

## Troubleshooting

### Common Issues
- **API connection failures**: Check REACT_APP_API_URL environment variable
- **CORS errors**: Verify backend CORS configuration
- **Build failures**: Clear node_modules and reinstall dependencies
- **Hot reload issues**: Restart development server

### Development Tips
- Use React DevTools for debugging
- Enable source maps for better error reporting
- Use console.log strategically (remove before production)
- Monitor network requests in browser DevTools

## Related Documentation
- **System Architecture**: See global CLAUDE.md for data flow and service communication
- **API Endpoints**: See backend/CLAUDE.local.md for implementation details
- **Deployment**: See deploy/CLAUDE.local.md for build and deployment procedures
- **Automation**: See scripts/CLAUDE.local.md for build scripts and utilities