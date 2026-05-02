import uvicorn

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = ROOT_DIR / "webapp"


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(WEBAPP_DIR),
        reload_dirs=[str(WEBAPP_DIR)],
    )
