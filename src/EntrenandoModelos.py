import numpy as np

from CargaDeImagenes import CargaDeImagenes
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer, Input, Conv2D, MaxPool2D, Reshape, Dense, Flatten, LeakyReLU, \
    MaxPooling2D, Dropout, BatchNormalization, ReLU

from sklearn.model_selection import KFold

# Preparando para el entrenamiento
ancho = 256
alto = 256
pixeles = ancho * alto
# Imagen RGB -->3
numeroCanales = 1
formaImagen = (ancho, alto, numeroCanales)
numeroCategorias = 5

cantidaDatosEntrenamiento = [60, 60, 60, 60, 60]

# Cargar las imágenes
cargaDeImagenes = CargaDeImagenes("Dataset/ImagenesDeEntrenamiento/")
imagenes, probabilidades = cargaDeImagenes.cargarDatos(numeroCategorias, cantidaDatosEntrenamiento, ancho, alto)

# Lista de modelos y las epocas de entrenamiento
modelosEntrenados = []


# Modelo 1, tomado de: https://guru99.es/convnet-tensorflow-image-classification/
# --------------Primer Modelo  capas normales y no tiene nada de raro XD ---------------------------------------
model1 = Sequential()
# Capa entrada
model1.add(InputLayer(input_shape=(pixeles,)))
model1.add(Reshape(formaImagen))

# Capas Ocultas
# Capas convolucionales
model1.add(Conv2D(kernel_size=5, strides=2, filters=16, padding="same", activation="relu", name="capa_1"))
model1.add(MaxPool2D(pool_size=2, strides=2))

model1.add(Conv2D(kernel_size=3, strides=1, filters=36, padding="same", activation="relu", name="capa_2"))
model1.add(MaxPool2D(pool_size=2, strides=2))

# Aplanamiento y capas densas
model1.add(Flatten())
model1.add(Dense(128, activation="relu"))

# Capa de salida
model1.add(Dense(numeroCategorias, activation="softmax"))

model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

modelosEntrenados.append((model1,10))


#--------------------------------------------------------------------------------------------------
#Modelo 2, tomado de: https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
#Este modelo cuenta con capas extras de activacion esta capa es una version con fugas de relu
#f(x) = alpha * x if x < 0
#f(x) = x if x >= 0
#

model2 = Sequential()

# Capa entrada
model2.add(InputLayer(input_shape=(pixeles,)))
model2.add(Reshape(formaImagen))

# Capas Ocultas
# Capas convolucionales
model2.add(Conv2D(filters=32, kernel_size=3,strides=1 ,activation='linear', padding='same', name="capa_1"))
model2.add(LeakyReLU(alpha=0.1))
model2.add(MaxPooling2D(pool_size=2, padding='same'))
model2.add(Dropout(0.5))
# https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html
# Aplanamiento y capas densas
model2.add(Flatten())
model2.add(Dense(32, activation='linear'))
model2.add(LeakyReLU(alpha=0.1))
model2.add(Dropout(0.5))

#capa de salida
model2.add(Dense(numeroCategorias, activation='softmax'))

model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

modelosEntrenados.append((model2,6))


# --------------------------------------------------------------------------------------------------
# Modelo 3
#https://ichi.pro/es/reconocimiento-de-digitos-escritos-a-mano-usando-redes-neuronales
# -convolucionales-cnn-en-el-conjunto-de-datos-mnist-p-30698561832603
# En este modelo se uso el optimizador nadam que es una variacion del adam la razon es que es el que tiene
# el segundo mejor corportamiento para la clasificacion de imagenes multiclase segun este estudio 
# https://velascoluis.medium.com/optimizadores-en-redes-neuronales-profundas-un-enfoque-pr%C3%A1ctico-819b39a3eb5

model3 = Sequential()

# Capa entrada
model3.add(InputLayer(input_shape=(pixeles,)))
model3.add(Reshape(formaImagen))

# capas ocultas
model3.add(Conv2D(kernel_size=5, strides=2, filters=30, padding="same", activation="relu", name="capa_1"))
model3.add(MaxPool2D(pool_size=2, strides=2))

model3.add(Conv2D(kernel_size=3, strides=1, filters=15, padding="same", activation="relu", name="capa_2"))
model3.add(MaxPool2D(pool_size=2, strides=2))

# Aplanamientpo y cpas densas
model3.add(Flatten())
model3.add(Dense(500, activation='relu'))
model3.add(Dropout(0.5))

# capa de salida
model3.add(Dense(numeroCategorias, activation='softmax'))

model3.compile(optimizer="nadam", loss="categorical_crossentropy",
               metrics=["accuracy"])
modelosEntrenados.append((model3, 10))

#------------------------------------------------------------------------------------------------------------
#Modelo 4 se tomo como referencia https://medium.com/swlh/image-classification-for-playing-cards-26d660f3149e
#se aumento el numero de capas convolucionales y densas ademas se uso la funcion de activacion rmsprop

model4 = Sequential()

# Capa entrada
model4.add(InputLayer(input_shape=(pixeles,)))
model4.add(Reshape(formaImagen))

# capas ocultas
model4.add(Conv2D(kernel_size=3, filters=64, padding="same", activation="relu", name="capa_1"))
model4.add(MaxPool2D(pool_size=2, strides=2))

model4.add(Conv2D(kernel_size=3, filters=128, padding="same", activation="relu", name="capa_2"))
model4.add(MaxPool2D(pool_size=2, strides=2))


# Aplanamientpo y cpas densas
model4.add(Flatten())
model4.add(Dropout(0.5))
model4.add(Dense(512, activation='relu'))

# capa de salida
model4.add(Dense(numeroCategorias, activation='softmax'))

model4.compile(optimizer="adam", loss="categorical_crossentropy",
               metrics=["accuracy"])
modelosEntrenados.append((model4, 8))


#----------------------------------------------------------------------------------------------------
# Modelo 5 tomado de
# https://la.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html
# este modelo implementa una capa de normalizacion despues de cada capa convolucional+

model5 = Sequential()


# Capa entrada
model5.add(InputLayer(input_shape=(pixeles,)))
model5.add(Reshape(formaImagen))

# capas ocultas

model5.add(Conv2D(kernel_size=3, filters=32,activation="relu", name="capa_1"))
model5.add(MaxPool2D(pool_size=2, strides=2))
model4.add(Dropout(0.25))

model5.add(Conv2D(kernel_size=3, filters=64,activation="relu", name="capa_2"))
model5.add(Conv2D(kernel_size=3, filters=64,activation="relu", name="capa_3"))
model4.add(Dropout(0.25))

model5.add(Conv2D(kernel_size=3, filters=196,activation="relu", name="capa_4"))
model4.add(Dropout(0.25))


# Aplanamientpo y cpas densas
model5.add(Flatten())
model5.add(Dense(512, activation='relu'))
model5.add(Dropout(0.5))

# capa de salida
model5.add(Dense(numeroCategorias, activation='softmax'))

model5.compile(optimizer="adam", loss="categorical_crossentropy",
               metrics=["accuracy"])
modelosEntrenados.append((model5, 4))




# Entrenando modelos
numero_modelo = 0
for (model, epocas) in modelosEntrenados:
    numero_modelo += 1
    # Entrenando modelo con validacion cruzada usando el 80% del dataset
    accuracy_fold = []
    loss_fold = []

    myFolds = KFold(n_splits=5, shuffle=True)

    i = 1
    for train, test in myFolds.split(imagenes, probabilidades):
        print("############Training fold ", i, "########################")
        model.fit(x=imagenes[train], y=probabilidades[train], epochs=epocas, batch_size=150)
        resultados = model.evaluate(x=imagenes[test], y=probabilidades[test])
        accuracy_fold.append(resultados[1])
        loss_fold.append(resultados[0])
        i += 1

    print("Accuracy validacion Cruzada modelo" + str(numero_modelo) + "=", accuracy_fold)
    print("Accuracy mean validacion cruzada", np.mean(accuracy_fold))

    # Guardar modelo
    ruta = "Modelos_CNN/modelo" + str(numero_modelo) + ".h5"
    model.save(ruta)
    # Informe de estructura de la red
    print("Informe de estructura de la red del modelo" + str(numero_modelo))
    model.summary()
