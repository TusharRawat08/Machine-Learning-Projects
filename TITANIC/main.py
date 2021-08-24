from flask import Flask
import joblib

app = Flask(__name__)

model = joblib.load('titanic_model.sav')

#@app.route("/") # route # get
#def index():
#   return "hello world"

@app.route('/titanic')
def titanic_prediction():
    prediction = model.predict([[8,2,1]])
    if prediction[0] == 0:
        return "sorry the person died"
    else:
        return "congrats"

if __name__ == "__main__": # starting point of the application
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True)