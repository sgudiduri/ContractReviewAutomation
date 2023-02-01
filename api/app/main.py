from typing import Any

from fastapi import APIRouter, FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from loguru import logger
import numpy as np
from  contract_nli.predict import Predict

favicon_path = 'favicon.ico'
api_router = APIRouter()

app = FastAPI(
    title="Contract NLI V1", openapi_url=f"/api/v1/openapi.json"
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in set(['png', 'jpg', 'jpeg'])


root_router = APIRouter()


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

@app.get("/")
async def docs_redirect():
    return RedirectResponse(url='/docs')

@root_router.get("/health", response_model=dict, status_code=200)
async def health() -> dict:
    """
    Root Get
    """
    health = dict(
        name="CIFAR 10 Imgage Classification API", api_version="1.0.0", model_version="7.0.0"
    )

    return health

@root_router.get("/PredictNLI/{premise}/{hypothesis}", response_model=dict, status_code=200)
async def PredictNLI(premise: str, hypothesis: str ) -> dict:
    """
    Root Get
    """
    p = Predict(100, 200)
    res = p.make_single_prediction(premise, hypothesis)
    return health


app.include_router(root_router)

# #Set all CORS enabled origins
if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")