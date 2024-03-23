import fastapi
from challenge.model import DelayModel
from pydantic import BaseModel

# Please, take a look at the only comment on the 
# overall section of the challenge.md file.

class Data(BaseModel):
    OPERA: str
    TIPOVUELO: Optional[str] = None
    MES: int

app = fastapi.FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: Data) -> dict:
    # creating the instance of the model class
    model = DelayModel()
    
    # Here you need to parse the json Data to 
    # a pd.DataFrame with the proper format
    # and the 10 expected features in order
    # to be able to call :
    # model.predict(features:  pd.DataFrame)
    
    # then we parse put the results on the 
    # following dictionary:
    response = {"predict": [0]}
    
    return response
