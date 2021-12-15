
import numpy as np
import cv2



class CargaDeImagenes:
    def __init__(self,rutaOrigen):
        self.rutaOrigen = rutaOrigen

    def cargarDatos(self, numeroCategorias, limite, ancho, alto):
        imagenesCargadas = []
        valorEsperado = []
        for categoria in range(0, numeroCategorias):
            for idImagen in range(0, limite[categoria]):
                ruta = self.rutaOrigen + str(categoria) + "/" + str(categoria) + "_" + str(idImagen) + ".jpg"
                print(ruta)
                imagen = cv2.imread(ruta)
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                imagen = cv2.resize(imagen, (ancho, alto))
                imagen = imagen.flatten()
                imagen = imagen / 255
                imagenesCargadas.append(imagen)

                probabilidades = np.zeros(numeroCategorias)
                probabilidades[categoria] = 1
                valorEsperado.append(probabilidades)
        imagenesEntrenamiento = np.array(imagenesCargadas)
        valoresEsperados = np.array(valorEsperado)
        return imagenesEntrenamiento, valoresEsperados

