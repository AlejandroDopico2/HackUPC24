from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/getRelatedGarments")
async def get_related_garments(request: Request):
    body = await request.body()
    
    # Decode the body from bytes to string
    image = body.decode("utf-8")
    
    return {"message": "Related garments fetched successfully"}