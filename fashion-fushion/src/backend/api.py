from enum import Enum

from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from PIL import Image
from recommender.recommender import Recommender
import os
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import base64

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


@app.post("/getRelatedGarments")
async def get_related_garments(file: UploadFile = File(...)):

    image = Image.open(BytesIO(await file.read()))

    print(type(image))

    recommended_images = recommender.recommend_similar_images(image)

    # Convert the image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {"image": img_str}


@app.get("/getSeasons")
async def get_seasons():
    return {season.name: season.value for season in Season}


@app.get("/getGarmentTypes")
async def get_garment_types():
    return {garment_type.name: garment_type.value for garment_type in GarmentType}


@app.get("/getColors")
async def get_colors():
    return {color.name: color.value for color in Color}
