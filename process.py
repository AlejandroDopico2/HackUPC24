from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO

# Crea una instancia de la aplicación FastAPI
app = FastAPI()


# Define una ruta para procesar una imagen
@app.post("")
async def procesar_imagen(imagen: UploadFile = File(...)):
    content_file = await imagen.read()

    img = Image.open(BytesIO(content_file))
    return img.convert('RGB')


