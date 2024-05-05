from enum import Enum

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from PIL import Image
from recommender import Recommender

app = FastAPI()

recommender = Recommender()

origins = ["http://localhost:3000", "localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GarmentType(Enum):
    SHIRT = "shirt"
    PANTS = "pants"
    JACKET = "jacket"


class Season(Enum):
    SUMMER = "summer"
    WINTER = "winter"
    AUTUMN = "autumn"
    SPRING = "spring"


class Color(Enum):
    BLACK = "black"
    WHITE = "white"
    RED = "red"
    PINK = "pink"


@app.get("/getRelatedGarments")
async def get_related_garments(request: Request):
    body = await request.body()

    # Decode the body from bytes to string
    image = body.decode("utf-8")

    print(type(image))

    image = Image.open(image)

    recommended_images = recommender.recommend_similar_images(image)

    print(recommended_images)

    return {"message": "Related garments fetched successfully"}


@app.get("/getSeasons")
async def get_seasons():
    return {season.name: season.value for season in Season}


@app.get("/getGarmentTypes")
async def get_garment_types():
    return {garment_type.name: garment_type.value for garment_type in GarmentType}


@app.get("/getColors")
async def get_colors():
    return {color.name: color.value for color in Color}
