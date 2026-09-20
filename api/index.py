import sys
import os
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import traceback

init_err = None
try:
    from backend.config import settings
    from backend.api.routes import router as api_router
    from backend.utils.rate_limiter import RateLimiterMiddleware
except Exception as e:
    init_err = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

if init_err:
    app = FastAPI(title="VoiceShield API Error")
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def catch_all_err(path: str):
        return JSONResponse(
            status_code=500,
            content={"status": "error", "init_error": init_err},
            headers={"Access-Control-Allow-Origin": "*"}
        )
else:
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

    app.include_router(api_router)


@app.get("/")
@app.get("/index.html")
async def serve_index():
    index_file = os.path.join(root_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "VoiceShield Backend API is running."}



