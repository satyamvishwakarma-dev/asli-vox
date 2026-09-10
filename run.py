# run.py
import uvicorn
from asli_vox.api.main import app

if __name__ == "__main__":
    uvicorn.run("asli_vox.api.main:app", port=8000, reload=True)