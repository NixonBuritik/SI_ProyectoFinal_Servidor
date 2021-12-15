from flask import Flask,jsonify,request

from config import config
from PrediccionDeLaCNN import PrediccionDeLaCNN

app = Flask(__name__)


@app.route('/models',methods=['GET'])
def list_models():

    modelos = ["modelo1","modelo2","modelo3"]
    respuesta = {}
    respuesta["list_of_models"] = modelos

    return jsonify(respuesta)

@app.route('/predict',methods=['POST'])
def predict():
    print(request.json)
    infoClient = request.json
    prediccion = PrediccionDeLaCNN(infoClient["models"])
    resultados = prediccion.obtenerPrediccion(infoClient["images"])


    respuesta = {}
    respuesta["state"] = "success"
    respuesta["message"] = "Predictions made satisfactorily"
    respuesta["results"] = resultados

    return jsonify(respuesta)


def recurso_no_encontrado(error):
     return jsonify({
         "state": "error",
         "message": "Error making predictions"
     }),404



if __name__ == '__main__':
    app.config.from_object(config['development'])
    app.register_error_handler(404,recurso_no_encontrado)
    app.run(host="192.168.101.10")