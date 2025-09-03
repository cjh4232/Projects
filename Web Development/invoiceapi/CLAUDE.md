# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Development server (auto-reload enabled)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Alternative: Run main module directly
python app/main.py
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/performance/   # Performance benchmarks

# Run single test file
pytest tests/test_invoice_extractor.py

# Run with verbose output
pytest -v
```

### Test Data Generation
```bash
# Generate test invoice PDFs (10 invoices, all styles)
python scripts/generate_test_invoices.py

# Generate with scanned versions for OCR testing
python scripts/generate_test_invoices.py --include-scanned --scan-quality low

# Generate specific count and style
python scripts/generate_test_invoices.py --count 5 --style modern_minimal
```

### Environment Setup
Uses `uv` for dependency management. Virtual environment is likely in `.venv/`.
```bash
# Install dependencies (if using uv)
uv sync

# Alternative with pip
pip install -r requirements.txt
```

## Architecture Overview

### Core Structure
- **FastAPI Application**: `app/main.py` - Main API server with invoice extraction endpoint
- **Models**: `app/models.py` - Pydantic models for invoice data structures
- **Configuration**: `app/config.py` - Environment-based settings with development/production modes
- **Authentication**: `app/auth.py` - JWT authentication with development test user fallback

### Invoice Processing Pipeline
1. **File Upload & Validation**: `app/utils/file_validator.py` - Comprehensive file validation
2. **Text Extraction**: `pdfplumber` extracts raw text from PDF files
3. **Text Linearization**: `app/extractors/linearizer.py` - **NEW**: Fixes side-by-side layouts and PDF artifacts
4. **Extraction**: `app/extractors/invoice_extractor_v2.py` - Enhanced extractor with linearized text processing
5. **Archive Processing**: `app/extractors/archive/` - Legacy OCR and parsing components (fallback)
6. **Data Models**: Structured extraction into `ExtractedInvoice` model with vendor, customer, line items

### Database Integration
- **Supabase**: Primary database for user management and extraction logging
- **Redis**: Optional caching layer
- **Usage Tracking**: Built-in user limits and subscription tiers
- **Development Mode**: Database operations skipped when credentials not configured

### Key Features
- **Multi-format Support**: PDF and image invoice processing
- **Text Linearization**: **NEW**: Fixes side-by-side layouts (FROM:/TO:), prevents extracting "INVOICE" as vendor names
- **Enhanced PDF Processing**: PDF-first approach with improved text extraction and table reconstruction
- **OCR Fallback**: Tesseract-based OCR when PDF text extraction fails
- **Rate Limiting**: Configurable per-minute request limits
- **Health Monitoring**: Comprehensive `/health` endpoint checking all dependencies
- **Environment-aware**: Different behaviors for development/production environments

### Testing Infrastructure
- **Comprehensive Test Suite**: Unit, integration, and performance tests
- **Mock Fixtures**: `tests/conftest.py` provides extensive test fixtures
- **Generated Test Data**: Scripts generate realistic PDF and scanned invoice samples
- **Quality Simulation**: Multiple scan quality levels for OCR testing

### File Processing
- **Temporary Storage**: Files processed in `/tmp/` with cleanup
- **PDF Text Linearization**: **NEW**: Converts side-by-side layouts to sequential format for better parsing
- **Multiple Extractors**: Enhanced PDF-first approach with text linearization, OCR fallback
- **File Size Limits**: Configurable via `max_file_size_mb` setting
- **Content Validation**: MIME type and magic number verification

## Environment Configuration

The application uses environment-specific settings:
- Development: Relaxed validation, database operations optional
- Production: Strict validation, all external services required
- Testing: Mock external services, isolated test environment

Key environment variables pattern: `.env.{environment}` files supported.

## Common Development Patterns

- **Error Handling**: Comprehensive logging with structured data (request_id, user_id)
- **Response Format**: Standardized `ExtractionResponse` with confidence scores
- **Processing Time Tracking**: Built-in performance monitoring
- **User Context**: All operations include user identification and usage tracking