import sys
import os
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

app = FastAPI(title="VoiceShield")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/api/health")
@app.get("/health")
async def health_check():
    try:
        from backend.config import settings
        from backend.api.routes import router as api_router
        return {
            "status": "healthy",
            "version": settings.VERSION,
            "demo_mode": settings.DEMO_MODE,
            "message": "VoiceShield Backend is operational"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"{type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc()
            }
        )

try:
    from backend.api.routes import router as api_router
    app.include_router(api_router)
except Exception:
    pass


