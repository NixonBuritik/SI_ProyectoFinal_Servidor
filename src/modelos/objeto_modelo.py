from tensorflow.python.keras.models import load_model
import numpy as np
import cv2

class objeto_modelo():
    def __init__(self,ruta,id,ancho,alto):
        self.modelo=load_model(ruta)
        self.id = id
        self.alto=alto
        self.ancho=ancho
        self.clases = ["Martillo", "Destornillador", "Llave Inglesa", "Alicate", "Regla"]

    def predecir(self,lista_imagenes):
        imagenes_cargadas=[]
        for imagen in lista_imagenes:
            imagen1 = cv2.imread(imagen.ruta)
            imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
            imagen1 = cv2.resize(imagen1, (self.ancho, self.alto))
            imagen1 = imagen1.flatten()
            imagen1 = imagen1 / 255
            imagenes_cargadas.append(imagen1)


        imagenesCargadasNPA = np.array(imagenes_cargadas)
        predicciones = self.modelo.predict(x=imagenesCargadasNPA)
        print("Predicciones=", predicciones)
        clasesMayores = np.argmax(predicciones, axis=1)

        for i in range(0,len(clasesMayores)):
            lista_imagenes[i].clase = self.clases[clasesMayores[i]]


        return lista_imagenes