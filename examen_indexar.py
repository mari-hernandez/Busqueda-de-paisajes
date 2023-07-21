# CC5213 - EXAMEN
# 24 de julio de 2023
# Alumnos: María Hernández - José Triviño

import sys
import os.path
import examen_descriptores as desc
import numpy
import cv2

def examen_indexar(dir_dataset_r, dir_datos_temporales):

    # Revisión de directorios
    if not os.path.isdir(dir_dataset_r):
        print("ERROR: no existe directorio {}".format(dir_dataset_r))
        sys.exit(1)
    elif os.path.exists(dir_datos_temporales):
        print("ERROR: ya existe directorio {}".format(dir_datos_temporales))
        sys.exit(1)


    #  Leer imágenes en dir_dataset_r

    lista_nombres = []
    matriz_descriptores = []

    for count, archivo in enumerate(os.listdir(dir_dataset_r)):

        print(str(count+1) + "/" + str(len(os.listdir(dir_dataset_r))))

        if not archivo.endswith(".jpg"):
            continue

        filename = dir_dataset_r+'\\'+archivo
        img_color = cv2.imread(filename, cv2.IMREAD_COLOR)

        descriptores = desc.descriptores_full(img_color)

        # Agregar descriptor a la matriz de descriptores
        if len(matriz_descriptores) == 0:
            matriz_descriptores = descriptores
        else:
            matriz_descriptores = numpy.vstack([matriz_descriptores, descriptores])

        # agregar nombre del archivo a la lista de nombres
        lista_nombres.append(archivo)


    #  Crear dir_datos_temporales y escribir los descriptores

    os.makedirs(dir_datos_temporales, exist_ok=True)

    f= open(str(dir_datos_temporales)+"\\nombres.txt","w+")
    for nombre in lista_nombres:
        f.write(nombre+"\n")
    f.close()

    numpy.savetxt(str(dir_datos_temporales)+"\data.txt",matriz_descriptores,fmt='%s')


# Inicio de la ejecución
if len(sys.argv) < 3:
    print("Uso: {} [dir_dataset_r] [dir_datos_temporales]".format(sys.argv[0]))
    sys.exit(1)

dir_dataset_r = sys.argv[1]
dir_datos_temporales = sys.argv[2]

examen_indexar(dir_dataset_r, dir_datos_temporales)
