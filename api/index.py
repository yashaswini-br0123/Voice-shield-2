import sys
import os

# Ensure project root is in sys.path for Vercel serverless functions
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from backend.main import app
except Exception as e:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    app = FastAPI(title="VoiceShield Emergency Fallback")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def fallback_catchall(path: str):
        return JSONResponse(
            status_code=500,
            content={"error": "Backend initialization failed", "details": str(e)}
        )
