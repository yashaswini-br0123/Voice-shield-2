import sys
import os
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.config import settings
from backend.api.routes import router as api_router
from backend.utils.rate_limiter import RateLimiterMiddleware

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="VoiceShield - Multimodal AI Deepfake Detection Platform"
)

app.add_middleware(RateLimiterMiddleware, max_requests=120, window_seconds=60)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
        headers={"Access-Control-Allow-Origin": "*"}
    )

app.include_router(api_router)

@app.get("/")
@app.get("/index.html")
async def serve_index():
    index_file = os.path.join(root_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "VoiceShield Backend API is running."}



