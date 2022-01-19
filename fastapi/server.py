from fastapi import FastAPI, File, UploadFile
import uvicorn
from starlette.responses import RedirectResponse
from prediction import predict, read_imagefile, predict_class
from io import BytesIO
from starlette.responses import StreamingResponse







app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>
<br>by Inés Broto"""

app = FastAPI(title='Ready for German?', description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")




@app.post("/predict/Image class")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "JPEG")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict_class(image)
    name = prediction[9:]
    #return prediction, probs
    return name


    
@app.post("/predict/top-5 prediction")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "JPEG")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction, probs, plt = predict(image)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    #return prediction, probs
    return StreamingResponse(buf, media_type="image/png")




if __name__ == "__main__":
    uvicorn.run(app, port = 8080, host = '0.0.0.0',debug=True)

