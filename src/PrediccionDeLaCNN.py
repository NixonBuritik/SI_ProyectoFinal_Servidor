import base64

from modelos.objeto_modelo import objeto_modelo
from modelos.objeto_imagen import Imagen



class PrediccionDeLaCNN:


    def __init__(self,listaModelos):

        self.ancho=256
        self.alto=256
        self.misModelosCNN = []
        self.cargar_modelos_requeridos(listaModelos)
        self.lista_imagenes = []

    def cargar_modelos_requeridos(self,listaModelos):
        for modelo in listaModelos:
            newModelo = objeto_modelo("Modelos_CNN/" + modelo + ".h5", modelo[-1], self.ancho, self.alto)
            self.misModelosCNN.append(newModelo)

    def decodificacion_guardado_de_imagenes(self,lista_imagenes_b64):
        for imagen_b64 in lista_imagenes_b64:
            imagen_64_decode = base64.decodebytes(imagen_b64["content"].encode('ascii'))
            # image_result = open('deer_decode.gif', 'wb')  # create a writable image and write the decoding result
            image_result = open("Imagenes_a_predecir/imagen_"+str(imagen_b64["id"])+".jpg", 'wb')
            imagen = Imagen("Imagenes_a_predecir/imagen_"+str(imagen_b64["id"])+".jpg",str(imagen_b64["id"]))
            self.lista_imagenes.append(imagen)
            image_result.write(imagen_64_decode)


    def obtenerPrediccion(self,lista_imagenes_b64):

        self.decodificacion_guardado_de_imagenes(lista_imagenes_b64)
        prediccion_todos_los_modelos = []
        for modelo in self.misModelosCNN:
            self.lista_imagenes = modelo.predecir(self.lista_imagenes)
            info_a_mostrar = {}
            info_a_mostrar["model_id"] = modelo.id
            info_a_mostrar["results"] = self.dar_formato_JSON_resultados()
            prediccion_todos_los_modelos.append(info_a_mostrar)

        return prediccion_todos_los_modelos

    def dar_formato_JSON_resultados(self):
        list_of_resultados = []
        for imagen in self.lista_imagenes:
            result = {}
            result["class"] = imagen.clase
            result["id_image"] = imagen.id
            list_of_resultados.append(result)

        return list_of_resultados

