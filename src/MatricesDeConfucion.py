import time

from tensorflow.python.keras.models import load_model
from CargaDeImagenes import CargaDeImagenes
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2

#carga de modelos
modelo1=load_model("Modelos_CNN/modelo1.h5")
modelo2=load_model("Modelos_CNN/modelo2.h5")
modelo3=load_model("Modelos_CNN/modelo3.h5")
modelo4=load_model("Modelos_CNN/modelo4.h5")
modelo5=load_model("Modelos_CNN/modelo5.h5")

modelos = [modelo1,modelo2,modelo3,modelo4,modelo5]


#categorias
names = ['Numero 0','Numero 1','Numero 2','Numero 3','Numero 4',
         'Numero 5','Numero 6','Numero 7','Numero 8','Numero 9']

#carga de imagenes para la prueba
imagenesPruebas = CargaDeImagenes("DataSetDividido/ImagenesDePrueba/")
ancho=128
alto=128
numeroCategorias=10
cantidaDatosPruebas=[64,64,64,64,64,64,64,64,64,64]
imagenesPrueba,probabilidadesPrueba=imagenesPruebas.cargarDatos(numeroCategorias,cantidaDatosPruebas,ancho,alto)

numero_modelo = 0
for modelo in modelos:
    numero_modelo += 1
    # Prueba del modelo con el 20% del dataset
    inicio = time.time()
    resultados = modelo.predict(imagenesPrueba)
    ResultadosEvaluacion = modelo.evaluate(x=imagenesPrueba, y=probabilidadesPrueba)
    fin = time.time()

    # obteniendo resultados
    clasesMayores = np.argmax(resultados, axis=1)
    valorEsperado = []

    for i in range(0, len(clasesMayores)):
        probabilidades = np.zeros(numeroCategorias)
        probabilidades[clasesMayores[i]] = 1
        valorEsperado.append(probabilidades)

    valoresEsperados = np.array(valorEsperado)

    matc = confusion_matrix(y_true=probabilidadesPrueba.argmax(axis=1), y_pred=valoresEsperados.argmax(axis=1))
    # print(matc)
    matriz = plot_confusion_matrix(conf_mat=matc, figsize=(6, 6), class_names=names, cmap='Wistia', show_normed=False,
                          fontcolor_threshold=1)
    # plt.tight_layout()
    matriz[1].set_title("Matriz de confucion modelo "+str(numero_modelo))
    print(metrics.classification_report(probabilidadesPrueba, valoresEsperados, digits=4))
    print("Accuracy del modelo "+str(numero_modelo)+"=", ResultadosEvaluacion[1])
    print("Tiempo en ejecutar la pruebas del modelo"+str(numero_modelo)+"=",str(fin-inicio))
    print("----------------------------------------------------------------------------"+"\n")

plt.show()



