from __future__ import annotations

from fastapi import FastAPI

from demo import milvus

app = FastAPI()
app.include_router(milvus.router)
app.add_event_handler("startup", milvus.startup)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}




