# SEO Meta Title & Description Optimization API

A stateless FastAPI application that generates AI-powered SEO-optimized product titles and descriptions using multiple optimization strategies and competitor analysis.

## Features

- **AI-Powered Content Generation**: Uses LiteLLM for flexible AI provider integration
- **Multiple Optimization Strategies**: Keyword-focused, benefit-driven, emotional appeal, technical specs, and brand-centric approaches
- **Competitor Analysis**: Web scraping and analysis of competitor content patterns
- **SEO Scoring Engine**: Comprehensive scoring algorithm with weighted factors
- **Stateless Design**: No database dependencies, perfect for microservices architecture
- **Production Ready**: Comprehensive logging, monitoring, and error handling

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)

### Local Development

1. **Clone and setup**:
   ```bash
   cd seo-optimizer-api
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run the application**:
   ```bash
   python -m app.main
   # Or using uvicorn directly:
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/v1/health
   - Metrics: http://localhost:8000/api/v1/metrics

### Docker Deployment

1. **Build the image**:
   ```bash
   docker build -t seo-optimizer-api .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 \
     -e AI_API_KEY=your_openai_key \
     -e DEBUG=true \
     seo-optimizer-api
   ```

## Project Structure

```
seo-optimizer-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── models/              # Pydantic data models
│   ├── routers/             # API route handlers
│   │   └── health.py        # Health check and metrics endpoints
│   ├── services/            # Business logic services
│   ├── utils/               # Utility functions
│   │   └── logging.py       # Structured logging setup
│   └── tests/               # Test files
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

### Required Configuration

- `AI_API_KEY`: Your OpenAI API key (required for AI content generation)
- `AI_PROVIDER`: AI provider (openai, anthropic, etc.)
- `AI_MODEL`: Model to use (gpt-3.5-turbo, gpt-4, etc.)

### Optional Configuration

- `DEBUG`: Enable debug mode (default: false)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `RATE_LIMIT_REQUESTS`: Requests per hour limit (default: 100)
- `ENABLE_COMPETITOR_ANALYSIS`: Enable web scraping (default: true)

## API Endpoints

### Health & Monitoring

- `GET /` - API information
- `GET /api/v1/health` - Basic health check
- `GET /api/v1/health/detailed` - Detailed health with system metrics
- `GET /api/v1/metrics` - Application metrics
- `GET /api/v1/readiness` - Kubernetes readiness probe
- `GET /api/v1/liveness` - Kubernetes liveness probe

### SEO Optimization (Coming in Phase 2)

- `POST /api/v1/optimize-title` - Generate SEO-optimized titles and descriptions

## Development

### Running Tests

```bash
pytest app/tests/
```

### Code Formatting

```bash
black app/
flake8 app/
mypy app/
```

### Environment-Specific Configurations

The application supports multiple environments:

- **Development**: Debug enabled, verbose logging
- **Production**: Optimized for performance and security
- **Testing**: Mock AI provider, disabled external services

Set `ENVIRONMENT=production` for production deployment.

## Monitoring

The application provides comprehensive monitoring:

- **Health Checks**: Multiple endpoints for different monitoring needs
- **Metrics**: Request counts, response times, error rates
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **System Metrics**: CPU, memory, disk usage

## Security

- **Input Validation**: Comprehensive validation using Pydantic
- **Rate Limiting**: Configurable request rate limits
- **Security Headers**: CORS, XSS protection, content type validation
- **Non-root Container**: Docker container runs as non-privileged user

## Performance

- **Async Processing**: Parallel execution of AI generation and analysis
- **Connection Pooling**: Efficient HTTP client management
- **Response Compression**: Automatic compression for large responses
- **Stateless Design**: Easy horizontal scaling

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please check the documentation or create an issue in the project repository.