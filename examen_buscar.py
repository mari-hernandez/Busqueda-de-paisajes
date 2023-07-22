# CC5213 - EXAMEN
# 24 de julio de 2023
# Alumnos: María Hernández - José Triviño

import json
import sys
import os.path
import time
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

def crear_rectangulo_con_texto(ancho, texto):
    """
    Crea un rectángulo blanco del ancho indicado 
    con el texto indicado en su interior.
    """
    canvas = numpy.ones((40, ancho, 3), dtype=numpy.uint8) * 255
    cv2.putText(canvas, texto, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas

def calcular_descriptores_q(archivo, dir_dataset_q):
    """
    Calcula los descriptores de la imagen de consulta.
    """
    filename = os.path.join(dir_dataset_q, archivo)
    img_color = cv2.imread(filename, cv2.IMREAD_COLOR)
    descriptores_imagen = desc.descriptores_full(img_color)
    return descriptores_imagen

def imagen_con_texto(path, texto='Imagen de consulta'):
    """
    Lee la imagen en la ruta indicada y la concatena con un rectángulo con texto.
    """
    resize = resize_img(path)
    risize_con_texto = cv2.vconcat([resize, crear_rectangulo_con_texto(resize.shape[1], texto)])
    return risize_con_texto
    

def examen_buscar(images_dir, json_descripcion, dir_dataset_q, dir_datos_temporales, file_resultados, one_img = None):
    """
    Busca las 5 imágenes más cercanas a la imagen de consulta.

    Parametros:
        images_dir -- directorio de imágenes (dataset inicial)
        json_descripcion -- archivo json con descripciones de imágenes
        dir_dataset_q -- directorio de imágenes de consulta
        dir_datos_temporales -- directorio de datos temporales (descriptores)
        file_resultados -- archivo donde se escribirán los resultados
        one_img -- nombre de la imagen de consulta (opcional)
    """

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
            descriptores_imagen = calcular_descriptores_q(archivo, dir_dataset_q)
            # agregar descriptor a la matriz de descriptores
            if len(matriz_descriptores) == 0:
                matriz_descriptores = descriptores_imagen
            else:
                matriz_descriptores = numpy.vstack([matriz_descriptores, descriptores_imagen])

            # agregar nombre del archivo a la lista de nombres
            lista_nombres.append(archivo)
        
    else:
        matriz_descriptores = calcular_descriptores_q(archivo, dir_dataset_q)

        # agregar nombre del archivo a la lista de nombres
        lista_nombres.append(one_img)

    #  2-leer descriptores de R de dir_datos_temporales
    descriptores = numpy.loadtxt(os.path.join(dir_datos_temporales, "data.txt"))

    #  3-para cada descriptor q localizar el mas cercano en R
    if len(matriz_descriptores.shape) == 1:
        matriz_descriptores = [matriz_descriptores]

    matriz_distancias = scipy.spatial.distance.cdist(matriz_descriptores, descriptores, metric='cityblock')

    # leer json metadata
    with open(json_descripcion) as json_file:
        metadata = json.load(json_file)

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
        ref_reseze_con_texto = imagen_con_texto(ref_path)

        ret = [ref_reseze_con_texto]
        for count, best_j in enumerate(nearest_indices):
            nombre_imagen_jpg = nombres_r[best_j]
            nombre_imagen_sin_ext = nombre_imagen_jpg.split('.')[0]
            descripcion = metadata[nombre_imagen_sin_ext]

            f.write(lista_nombres[i] + "\t" + nombre_imagen_jpg + "\t" + str(distancias[best_j]) +"\t"+descripcion + "\n")
            best_path = os.path.join(images_dir, nombre_imagen_jpg)
            best_resized_con_texto = imagen_con_texto(best_path, descripcion)
            ret.append(best_resized_con_texto)

        if one_img:
            busqueda_concatenada = cv2.hconcat(ret)
            cv2.imshow("Resultados", busqueda_concatenada)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
    
    f.close()
    n.close()

# Inicio de la ejecución
if len(sys.argv) < 5:
    print("Uso: {} [dir_dataset_q] [dir_datos_temporales] [resultados.txt]".format(sys.argv[0]))
    sys.exit(1)

elif len(sys.argv) >= 5:
    images_dir = sys.argv[1]
    json_descripcion = sys.argv[2]
    dir_dataset_q = sys.argv[3]
    dir_datos_temporales = sys.argv[4]
    file_resultados = sys.argv[5]

    if os.path.exists(file_resultados):
        os.remove(file_resultados)

    if len(sys.argv) == 6:
        examen_buscar(images_dir, json_descripcion, dir_dataset_q, dir_datos_temporales, file_resultados)

    else:
        while True:
            img_name = input("Ingrese nombre de archivo: ")
            inicio = time.time()
            examen_buscar(images_dir, json_descripcion,dir_dataset_q, dir_datos_temporales, file_resultados, img_name)
            fin = time.time()
            print("Tiempo de ejecución: {} segundos".format(fin-inicio))
            continuar = input("Desea buscar otra imagen? (y/n): ")
            if continuar == "n":
                break
            elif continuar == "y":
                os.remove(file_resultados)
                continue
