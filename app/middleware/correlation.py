import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.utils.logger import set_correlation_id, clear_correlation_id, logger


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract or generate correlation IDs for request tracing.
    
    Extracts correlation ID from request headers or generates a new one.
    Adds correlation ID to response headers and logging context.
    """
    
    def __init__(self, app, correlation_id_header: str = "X-Correlation-ID"):
        super().__init__(app)
        self.correlation_id_header = correlation_id_header
    
    async def dispatch(self, request: Request, call_next):
        """
        Process incoming request and add correlation ID.
        """
        # Extract or generate correlation ID
        correlation_id = request.headers.get(
            self.correlation_id_header,
            str(uuid.uuid4())
        )
        
        # Set in logging context
        set_correlation_id(correlation_id)
        
        # Log incoming request
        logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown"
            }
        )
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - Status: {response.status_code}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code
                }
            )
            
            # Add correlation ID to response headers for client tracking
            response.headers[self.correlation_id_header] = correlation_id
            
            return response
            
        finally:
            clear_correlation_id()
