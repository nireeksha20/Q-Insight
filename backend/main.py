from fastapi import FastAPI

app = FastAPI(
    title="Q-Insight API",
    description="Manufacturing quality inspection and analytics",
    version="0.1.0",
)

@app.get("/health")
def health_check():
    return {"status": "ok", "project": "Q-Insight"}
