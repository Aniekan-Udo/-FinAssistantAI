# Entry point alias so both `uvicorn app:app` and `uvicorn main:app` start the API
from app import app

if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
