from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import mlflow.pyfunc

# Load model from MLflow Model Registry
model = mlflow.pyfunc.load_model("models:/medical-insurance/Production")

app = FastAPI()

class InputData(BaseModel):
    age: float
    bmi: float

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>Medical Insurance Predictor</title></head>
        <body>
            <h2>Predict Insurance Cost</h2>
            <form action="/predict" method="post">
                Age: <input type="number" name="age" step="any"><br>
                BMI: <input type="number" name="bmi" step="any"><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    age = float(form["age"])
    bmi = float(form["bmi"])
    prediction = model.predict([[age, bmi]])[0]
    return f"""
    <html>
        <body>
            <h3>Predicted Insurance Cost: ₹{prediction:.2f}</h3>
            <a href="/">Try another prediction</a>
        </body>
    </html>
    """