import uvicorn

from app.core.reminders import init_scheduler


if __name__ == "__main__":
    init_scheduler()
    uvicorn.run("app.server.main:app", host="0.0.0.0", port=8000, reload=True)
