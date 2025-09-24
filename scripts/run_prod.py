import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.server.main:app",
        host="0.0.0.0",
        port=8080,
        workers=2,
        reload=False,
        proxy_headers=True,
    )
