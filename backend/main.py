import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import settings
from backend.api.routes import router as api_router
from backend.utils.rate_limiter import RateLimiterMiddleware

from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="VoiceShield - Multimodal AI Deepfake Detection Platform"
)

# Middleware: RateLimiter added first, CORSMiddleware added second (so CORS is outermost)
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

# Include API endpoints
app.include_router(api_router)

# Mount frontend static files
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
css_path = os.path.join(root_path, "css")
js_path = os.path.join(root_path, "js")

if os.path.exists(css_path):
    app.mount("/css", StaticFiles(directory=css_path), name="css")

if os.path.exists(js_path):
    app.mount("/js", StaticFiles(directory=js_path), name="js")

@app.get("/")
@app.get("/index.html")
async def serve_index():
    index_file = os.path.join(root_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "VoiceShield Backend API is running."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=False)
