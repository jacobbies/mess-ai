# Backend Development Instructions

## FastAPI Development Patterns

### API Endpoints
- Use dependency injection for database connections
- Implement proper error handling with HTTPException
- Follow REST conventions for resource naming
- Use Pydantic models for request/response validation

### Database Operations
- Use async/await for all database operations
- Implement proper transaction management
- Use connection pooling for performance
- Follow migration patterns for schema changes

### Testing
```bash
# Run backend tests
pytest backend/tests/

# Run with coverage
pytest --cov=backend backend/tests/

# Run specific test file
pytest backend/tests/test_api.py
```

## ML Pipeline Development

### Feature Extraction
- Use MERT-v1-95M for audio embeddings
- Process audio at 44kHz for MERT compatibility
- Cache extracted features as .npy files
- Use MPS acceleration on Apple Silicon

### FAISS Integration
- Use IndexFlatIP for cosine similarity
- Cache indices for fast startup
- Implement proper error handling for missing indices
- Use L2-normalized vectors for similarity search

### Performance Optimization
- Batch process audio files when possible
- Use async processing for I/O operations
- Implement proper memory management
- Monitor GPU/CPU usage during processing

## Code Quality Standards

### Formatting
```bash
# Format code
black backend/
isort backend/

# Type checking
mypy backend/
```

### Error Handling
- Use structured logging with correlation IDs
- Implement graceful degradation for ML failures
- Provide meaningful error messages to users
- Log performance metrics for monitoring

## Environment Configuration

### Development
```bash
# Set environment variables
export ENVIRONMENT=development
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start development server
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
# Use gunicorn for production
gunicorn backend.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Troubleshooting

### Common Issues
- **Import errors**: Check Python path and virtual environment
- **MERT model loading**: Ensure trust_remote_code=True
- **FAISS crashes**: Verify feature dimensions match index
- **Memory issues**: Monitor RAM usage during feature extraction

### Performance Tips
- Use async/await consistently
- Implement proper caching strategies
- Monitor database connection pool usage
- Use background tasks for heavy operations