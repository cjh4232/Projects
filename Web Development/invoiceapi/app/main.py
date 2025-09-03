from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import time
import os
import uuid
import logging
from datetime import datetime

from .models import ExtractionResponse, ExtractedInvoice
from .database import check_user_usage_limit, increment_user_usage, save_extraction
from .extractors.invoice_extractor_v2 import InvoiceExtractor
from .config import get_settings
from .auth import get_current_user, get_test_user
from .utils.file_validator import comprehensive_file_validation
from .middleware.rate_limiting import RateLimitMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="InvoiceAPI",
    description="Extract structured data from PDF and image invoices",
    version="1.0.0"
)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_requests_per_minute)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Choose authentication method based on environment
def get_auth_dependency():
    """Return appropriate authentication dependency based on environment"""
    if settings.is_development:
        logger.info("Using test authentication for development mode")
        return get_test_user
    else:
        logger.info("Using JWT authentication for production mode")
        return get_current_user

@app.get("/")
async def root():
    return {"message": "InvoiceAPI is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    checks = {
        "api": "healthy",
        "database": "unknown",
        "pdf_extraction": "unknown",
        "ocr_fallback": "unknown",
        "environment": settings.environment,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Check PDF extraction capabilities
        import pdfplumber
        checks["pdf_extraction"] = "healthy"
    except ImportError:
        checks["pdf_extraction"] = "unhealthy: pdfplumber not installed"
        logger.warning("pdfplumber not available")
    except Exception as e:
        checks["pdf_extraction"] = f"unhealthy: {str(e)}"
        logger.warning(f"PDF extraction health check failed: {e}")
    
    try:
        # Check if OCR fallback is available
        import pytesseract
        pytesseract.get_tesseract_version()
        checks["ocr_fallback"] = "healthy"
    except Exception as e:
        checks["ocr_fallback"] = f"unhealthy: {str(e)}"
        logger.warning(f"OCR fallback health check failed: {e}")
    
    # For development mode, don't check database if credentials are not set
    if settings.is_development and not settings.supabase_url:
        checks["database"] = "disabled (development mode)"
    else:
        try:
            # Check database connection (simplified)
            usage_info = await check_user_usage_limit("health-check")
            checks["database"] = "healthy"
        except Exception as e:
            checks["database"] = f"unhealthy: {str(e)}"
            logger.warning(f"Database health check failed: {e}")
    
    # Return appropriate status code
    status_code = 200 if all(
        v in ["healthy", "disabled (development mode)", "unknown"] 
        for v in [checks["api"], checks["database"], checks["pdf_extraction"], checks["ocr_fallback"]]
    ) else 503
    
    return JSONResponse(content=checks, status_code=status_code)

@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Depends(get_auth_dependency())
):
    """Extract data from uploaded invoice"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    logger.info(f"Invoice extraction started", extra={
        "request_id": request_id,
        "user_id": user_id,
        "file_name": file.filename,
        "content_type": file.content_type
    })
    
    # Read file content for validation
    file_bytes = await file.read()
    await file.seek(0)  # Reset file pointer
    
    # Comprehensive file validation
    try:
        file_info = comprehensive_file_validation(
            file_bytes=file_bytes,
            filename=file.filename or "unknown",
            content_type=file.content_type or "application/octet-stream",
            max_size_mb=settings.max_file_size_mb
        )
        
        logger.info(f"File validation passed", extra={
            "request_id": request_id,
            "file_info": file_info
        })
        
        file_size = file_info["size_bytes"]
        
    except HTTPException as e:
        logger.error(f"File validation failed", extra={
            "request_id": request_id,
            "error": e.detail
        })
        raise
    
    # Check user usage limit (skip in development mode if DB not configured)
    if settings.is_development and not settings.supabase_url:
        logger.info("Skipping usage limit check in development mode")
        usage_info = {"has_credits": True, "tier_limit": "unlimited", "subscription_tier": "development"}
    else:
        try:
            usage_info = await check_user_usage_limit(user_id)
            if not usage_info['has_credits']:
                raise HTTPException(
                    status_code=429,
                    detail=f"Monthly limit reached ({usage_info['tier_limit']} documents for {usage_info['subscription_tier']} tier)"
                )
        except Exception as e:
            logger.error(f"Usage limit check failed", extra={
                "request_id": request_id,
                "error": str(e)
            })
            if settings.is_production:
                raise HTTPException(status_code=503, detail="Service temporarily unavailable")
            else:
                # In development, continue without usage limits
                logger.warning("Continuing without usage limits in development mode")
                usage_info = {"has_credits": True}
    
    # Determine file type and validate PDF for optimal extraction
    file_type = "pdf" if file.content_type == "application/pdf" else "image"
    
    # For non-PDF files, show a warning but continue
    if file_type != "pdf":
        logger.warning(f"Non-PDF file uploaded: {file_type}. PDF-first extraction may fall back to OCR.")
    
    try:
        # Save uploaded file temporarily for processing
        temp_file_path = f"/tmp/invoice_extraction_{request_id}_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_bytes)
        
        # Use simple invoice extractor
        extractor = InvoiceExtractor()
        invoice_data = extractor.extract(temp_file_path)
        
        # For compatibility with existing response format, create mock confidence scores
        field_confidences = {
            "invoice_number": 0.95 if invoice_data.invoice_number else 0.0,
            "vendor_info": 0.90 if invoice_data.vendor else 0.0,
            "customer_info": 0.85 if invoice_data.customer else 0.0,
            "line_items": 0.90 if invoice_data.line_items else 0.0,
            "total_amount": 0.95 if invoice_data.total_amount else 0.0
        }
        
        # Clean up temp file
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass  # Don't fail if cleanup fails
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Calculate overall confidence score for logging
        overall_confidence = sum(field_confidences.values()) / len(field_confidences)
        
        # Save to database (skip in development if DB not configured)
        if settings.is_development and not settings.supabase_url:
            logger.info("Skipping database operations in development mode")
        else:
            try:
                await save_extraction(
                    user_id=user_id,
                    file_name=file_info.get("safe_filename", file.filename),
                    file_type=file_type,
                    file_size=file_size,
                    extracted_data=invoice_data.model_dump(),
                    confidence_score=overall_confidence,
                    processing_time_ms=processing_time,
                    status="completed"
                )
                
                # Increment usage
                await increment_user_usage(user_id)
                
            except Exception as e:
                logger.error(f"Database operations failed", extra={
                    "request_id": request_id,
                    "error": str(e)
                })
                # Don't fail the request if database save fails
                # The extraction was successful
        
        return ExtractionResponse(
            success=True,
            data=invoice_data,
            confidence_scores=field_confidences,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.error(f"Invoice extraction failed", extra={
            "request_id": request_id,
            "user_id": user_id,
            "error": str(e),
            "processing_time_ms": processing_time
        })
        
        # Save failed extraction (only if database is available)
        if not (settings.is_development and not settings.supabase_url):
            try:
                await save_extraction(
                    user_id=user_id,
                    file_name=file_info.get("safe_filename", file.filename) if 'file_info' in locals() else file.filename,
                    file_type=file_type,
                    file_size=file_size if 'file_size' in locals() else 0,
                    extracted_data={},
                    confidence_score=0,
                    processing_time_ms=processing_time,
                    status="failed",
                    error_message=str(e)
                )
            except Exception as db_e:
                logger.error(f"Failed to save error to database", extra={
                    "request_id": request_id,
                    "db_error": str(db_e)
                })
        
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@app.get("/usage")
async def get_usage(user_id: str = Depends(get_current_user)):
    """Get current usage information"""
    usage_info = await check_user_usage_limit(user_id)
    return usage_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
