#!/usr/bin/env python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,  
        reload=False,
        log_level="info"
    )
