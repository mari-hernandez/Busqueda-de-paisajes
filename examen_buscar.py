# CC5213 - EXAMEN
# 24 de julio de 2023
# Alumnos: María Hernández - José Triviño

import sys
import os.path
import cv2
import numpy
import scipy.spatial
import examen_descriptores as desc

def resize_img(img_path, border=False, show=False):

    ref_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    white_canvas = numpy.ones((300, 300, 3), dtype=numpy.uint8) * 255
    border_canvas = numpy.ones((310, 310, 3), dtype=numpy.uint8) * 255

    if border:
        border_canvas = border_canvas * 0
    
    y = ref_img.shape[1]
    x = ref_img.shape[0]
    if x>y:
        new_y = int(y*(300/x))
        ref_resized = cv2.resize(ref_img, (new_y, 300))
        white_canvas[0:300, (300-new_y)//2:((300-new_y)//2)+new_y] = ref_resized
    
    else:
        new_x = int(x*(300/y))
        ref_resized = cv2.resize(ref_img, (300, new_x))
        white_canvas[(300-new_x)//2:((300-new_x)//2)+new_x, 0:300] = ref_resized
    
    border_canvas[5:305,5:305] = white_canvas

    if show:
        cv2.imshow("Imagen resized", border_canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return border_canvas


def examen_buscar(images_dir, dir_dataset_q, dir_datos_temporales, file_resultados, one_img = None):

    # Revisión de directorios
    if not os.path.isdir(dir_dataset_q):
        print("ERROR: no existe directorio {}".format(dir_dataset_q))
        sys.exit(1)
    elif not os.path.isdir(dir_datos_temporales):
        print("ERROR: no existe directorio {}".format(dir_datos_temporales))
        sys.exit(1)
    elif not os.path.isdir(images_dir):
        print("ERROR: no existe directorio {}".format(images_dir))
        sys.exit(1)
    if one_img:
        if not os.path.isfile(os.path.join(dir_dataset_q, one_img)):
            print("ERROR: no existe archivo {}".format(one_img))
            return
    
    # Calcular descriptores de la imagen buscada
    
    lista_nombres = []
    matriz_descriptores = []

    if not one_img:
        for archivo in os.listdir(dir_dataset_q):
            if not archivo.endswith(".jpg"):
                continue

            filename = os.path.join(dir_dataset_q, archivo)
            img_color = cv2.imread(filename, cv2.IMREAD_COLOR)

            descriptores_imagen = desc.descriptores_full(img_color)

            # agregar descriptor a la matriz de descriptores
            if len(matriz_descriptores) == 0:
                matriz_descriptores = descriptores_imagen
            else:
                matriz_descriptores = numpy.vstack([matriz_descriptores, descriptores_imagen])

            # agregar nombre del archivo a la lista de nombres
            lista_nombres.append(archivo)
        
    else:
        filename = os.path.join(dir_dataset_q, one_img)
        img_color = cv2.imread(filename, cv2.IMREAD_COLOR)

        descriptores_imagen = desc.descriptores_full(img_color)

        # agregar descriptor a la matriz de descriptores
        if len(matriz_descriptores) == 0:
            matriz_descriptores = descriptores_imagen
        else:
            matriz_descriptores = numpy.vstack([matriz_descriptores, descriptores_imagen])

        # agregar nombre del archivo a la lista de nombres
        lista_nombres.append(one_img)

    #  2-leer descriptores de R de dir_datos_temporales

    descriptores = numpy.loadtxt(os.path.join(dir_datos_temporales, "data.txt"))

    #  3-para cada descriptor q localizar el mas cercano en R

    if len(matriz_descriptores.shape) == 1:
        matriz_distancias = scipy.spatial.distance.cdist([matriz_descriptores], descriptores, metric='cityblock')
    else:
        matriz_distancias = scipy.spatial.distance.cdist(matriz_descriptores, descriptores, metric='cityblock')

    #  4-escribir en file_resultados
    f= open(file_resultados,"w+")
    n= open(os.path.join(dir_datos_temporales, "nombres.txt"),"r")
    nombres_r = n.read().splitlines()

    for i in range(len(matriz_distancias)): # Datos q
        distancias = matriz_distancias[i]
        # Se obtienen los índices ordenados de las distancias
        sorted_indices = numpy.argsort(distancias)
        # Se obtienen los 5 vecinos más cercanos
        nearest_indices = sorted_indices[:5]

        ref_path = os.path.join(dir_dataset_q, lista_nombres[i])
        ref_resized = resize_img(ref_path, True)
        row_2 = []
        
        for count, best_j in enumerate(nearest_indices):
            f.write(lista_nombres[i] + "\t" + nombres_r[best_j] + "\t" + str(distancias[best_j]) + "\n")
            best_path = os.path.join(images_dir, nombres_r[best_j])
            best_resized = resize_img(best_path)
            if count == 2:
                row_2 = best_resized
            elif count > 2:
                row_2 = numpy.concatenate((row_2, best_resized), axis=1)
            else:
                ref_resized = numpy.concatenate((ref_resized, best_resized), axis=1)
            if count == 4:
                ref_resized = numpy.concatenate((ref_resized, row_2), axis=0)
        
        if one_img:
            cv2.imshow("Resultados", ref_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    f.close()
    n.close()

# Inicio de la ejecución
if len(sys.argv) < 4:
    print("Uso: {} [dir_dataset_q] [dir_datos_temporales] [resultados.txt]".format(sys.argv[0]))
    sys.exit(1)

elif len(sys.argv) >= 4:
    images_dir = sys.argv[1]
    dir_dataset_q = sys.argv[2]
    dir_datos_temporales = sys.argv[3]
    file_resultados = sys.argv[4]

    if os.path.exists(file_resultados):
        os.remove(file_resultados)

    if len(sys.argv) == 5:
        examen_buscar(images_dir, dir_dataset_q, dir_datos_temporales, file_resultados)

    else:
        while True:
            img_name = input("Ingrese nombre de archivo: ")
            examen_buscar(images_dir, dir_dataset_q, dir_datos_temporales, file_resultados, img_name)
            continuar = input("Desea buscar otra imagen? (y/n): ")
            if continuar == "n":
                break
            elif continuar == "y":
                os.remove(file_resultados)
                continue
