import cv2
if __name__ == '__main__':


    for i in range(0,5):
        for j in range(0,60):
            ruta = "Dataset/ImagenesDeEntrenamiento/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
            imagen = cv2.imread(ruta)
            imagen = cv2.resize(imagen, (256, 256))
            cv2.imwrite(ruta,imagen)


