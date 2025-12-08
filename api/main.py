"""
FastAPI Application untuk Sentiment Analysis Model Serving
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
import time
from contextlib import asynccontextmanager

from api.models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from api.utils import ModelLoader, validate_model_files

# Import logger untuk database logging
try:
    from dashboard.utils.logger import log_prediction
    LOGGING_ENABLED = True
except ImportError:
    LOGGING_ENABLED = False
    print("Warning: Dashboard logger not available. Predictions will not be logged.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model loader instance
model_loader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager untuk load model saat startup
    dan cleanup saat shutdown
    """
    global model_loader
    
    # Startup: Load model
    logger.info("Starting up API server...")
    logger.info("Loading machine learning model...")
    
    try:
        if not validate_model_files():
            logger.error("Model files tidak ditemukan!")
            raise FileNotFoundError("Model atau vectorizer tidak ditemukan di folder models/")
        
        model_loader = ModelLoader()
        model_loader.load_model()
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down API server...")


# Initialize FastAPI app
app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="REST API untuk prediksi sentiment dari movie reviews menggunakan Logistic Regression + TF-IDF",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - allow all origins untuk development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - basic info
    """
    return {
        "message": "IMDB Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "prediction": "/predict",
            "health": "/health",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint untuk monitoring
    
    Returns informasi status API dan model
    """
    is_model_loaded = model_loader is not None and \
                      model_loader.model is not None and \
                      model_loader.vectorizer is not None
    
    return HealthResponse(
        status="healthy" if is_model_loaded else "unhealthy",
        model_loaded=is_model_loaded,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get informasi detail tentang model yang digunakan
    
    Returns:
        ModelInfoResponse dengan metrics dan metadata model
    """
    if model_loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum di-load"
        )
    
    try:
        info = model_loader.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model info: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment dari input text
    
    Args:
        request: PredictionRequest berisi text yang akan diprediksi
        
    Returns:
        PredictionResponse dengan hasil prediksi dan confidence score
        
    Raises:
        HTTPException 503: Jika model belum ready
        HTTPException 500: Jika terjadi error saat prediksi
    """
    if model_loader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model belum di-load. Tunggu beberapa saat."
        )
    
    try:
        # Start timing
        start_time = time.time()
        
        # Predict
        result = model_loader.predict(request.text)
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Build response
        response = PredictionResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            text_length=len(request.text),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction: {result['sentiment']} (confidence: {result['confidence']:.4f})")
        
        # Log to database (if logger available)
        if LOGGING_ENABLED:
            try:
                log_prediction(
                    text=request.text,
                    sentiment=result["sentiment"],
                    confidence=result["confidence"],
                    probabilities=result["probabilities"],
                    response_time=response_time
                )
            except Exception as log_error:
                logger.warning(f"Failed to log prediction: {str(log_error)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing prediction: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler untuk HTTPException"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Custom exception handler untuk general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
