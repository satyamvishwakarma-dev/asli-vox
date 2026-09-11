import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from asli_vox.api.routes import router as routes_router
from asli_vox.api.websocket import router as ws_router

app = FastAPI(title="AsliVox API")

# Resolve absolute path to the root frontend/ folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Include API and WebSocket routers
app.include_router(routes_router)
app.include_router(ws_router)

# Mount the entire frontend directory at /static
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))