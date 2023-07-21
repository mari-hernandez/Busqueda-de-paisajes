# CC5213 - EXAMEN
# 24 de julio de 2023
# Alumnos: María Hernández - José Triviño

import numpy
import cv2

#############################
# DESCRIPTORES INDIVIDUALES #
#############################

def vector_de_intensidades(img_gris, x_zones, y_zones):
    imagen_2 = cv2.resize(img_gris, (x_zones, y_zones), interpolation=cv2.INTER_AREA)
    descriptor_imagen = imagen_2.flatten()
    return descriptor_imagen

def vector_de_intensidades_eq(img_gris, x_zones, y_zones):
    imagen_2 = cv2.equalizeHist(img_gris)
    imagen_2 = cv2.resize(img_gris, (x_zones, y_zones), interpolation=cv2.INTER_AREA)
    descriptor_imagen = imagen_2.flatten()
    return descriptor_imagen

def vector_de_intensidades_omd(img_gris, x_zones, y_zones):
    imagen_2 = cv2.resize(img_gris, (x_zones, y_zones), interpolation=cv2.INTER_AREA)
    descriptor_imagen = imagen_2.flatten()
    posiciones = numpy.argsort(descriptor_imagen)
    for i in range(len(posiciones)):
        descriptor_imagen[posiciones[i]] = i
    return descriptor_imagen

def histograma_por_zona(imagen, x_zones, y_zones):
    # divisiones
    num_zonas_x = x_zones
    num_zonas_y = y_zones
    num_bins_por_zona = 8
    ecualizar = True
    # leer imagen
    if ecualizar:
        imagen = cv2.equalizeHist(imagen)
    # procesar cada zona
    descriptor = []
    for j in range(num_zonas_y):
        desde_y = int(imagen.shape[0] / num_zonas_y * j)
        hasta_y = int(imagen.shape[0] / num_zonas_y * (j+1))
        for i in range(num_zonas_x):
            desde_x = int(imagen.shape[1] / num_zonas_x * i)
            hasta_x = int(imagen.shape[1] / num_zonas_x * (i+1))
            # recortar zona de la imagen
            zona = imagen[desde_y : hasta_y, desde_x : hasta_x]
            # histograma de los pixeles de la zona
            histograma, limites = numpy.histogram(zona, bins=num_bins_por_zona, range=(0,255))
            # normalizar histograma (bins suman 1)
            histograma = histograma / numpy.sum(histograma)
            # agregar descriptor de la zona al descriptor global
            descriptor.extend(histograma)
    return descriptor

def angulos_por_zona(imagen, x_zones, y_zones):

    def angulos_en_zona(imgBordes, imgSobelX, imgSobelY):

        solo_bordes = (imgBordes > 0)
        divided = numpy.divide(imgSobelY,imgSobelX,out=numpy.zeros_like(imgSobelX), where=imgSobelX != 0)

        angulos = numpy.select([imgSobelX != 0, imgSobelX == 0],
                        [numpy.degrees(numpy.arctan(divided)),solo_bordes.astype(int)*90])

        return angulos[solo_bordes]
    
    # divisioness
    num_zonas_x = x_zones
    num_zonas_y = y_zones
    num_bins_por_zona = 9
    threshold_magnitud_gradiente = 50
    # calcular filtro de sobel (usar cv2.GaussianBlur para borrar ruido)
    imagen = cv2.GaussianBlur(imagen, (3,3), 0, 0)
    sobelX = cv2.Sobel(imagen, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    sobelY = cv2.Sobel(imagen, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    magnitud = numpy.sqrt(numpy.square(sobelX) + numpy.square(sobelY))
    th, bordes = cv2.threshold(magnitud, threshold_magnitud_gradiente, 255, cv2.THRESH_BINARY)
    # procesar cada zona
    descriptor = []
    for j in range(num_zonas_y):
        desde_y = int(imagen.shape[0] / num_zonas_y * j)
        hasta_y = int(imagen.shape[0] / num_zonas_y * (j+1))
        for i in range(num_zonas_x):
            desde_x = int(imagen.shape[1] / num_zonas_x * i)
            hasta_x = int(imagen.shape[1] / num_zonas_x * (i+1))
            # calcular angulos de la zona
            angulos = angulos_en_zona(bordes[desde_y : hasta_y, desde_x : hasta_x],
                                    sobelX[desde_y : hasta_y, desde_x : hasta_x],
                                    sobelY[desde_y : hasta_y, desde_x : hasta_x])
            # histograma de los angulos de la zona
            histograma, limites = numpy.histogram(angulos, bins=num_bins_por_zona, range=(-90,90))
            # normalizar histograma (bins suman 1)
            if numpy.sum(histograma) != 0:
                histograma = histograma / numpy.sum(histograma)
            # agregar descriptor de la zona al descriptor global
            descriptor.extend(histograma)
            # dibujar histograma de la zona
    return descriptor

def vector_de_colores(img_rgb, x_zones, y_zones):
    imagen_2 = cv2.resize(img_rgb, (x_zones, y_zones), interpolation=cv2.INTER_AREA)

    imagen_r = imagen_2[:, :, 0]
    imagen_g = imagen_2[:, :, 1]
    imagen_b = imagen_2[:, :, 2]

    r_means = imagen_r.flatten()
    g_means = imagen_g.flatten()
    b_means = imagen_b.flatten()
    
    descriptor_imagen = numpy.concatenate((r_means, g_means))
    descriptor_imagen = numpy.concatenate((descriptor_imagen, b_means))

    return descriptor_imagen


#############################
#  CÁLCULO DE DESCRIPTORES  #
#############################

def descriptores_gris(img_gris):

    descriptores_imagen = vector_de_intensidades_eq(img_gris, 4, 4)

    lista_descriptores = [
        vector_de_intensidades_omd,
        histograma_por_zona,
        angulos_por_zona
    ]

    combinaciones = [(1,4),(4,4)]
    for d in lista_descriptores:
        for c in combinaciones:
            descriptor = d(img_gris, c[0], c[1])
            descriptores_imagen = numpy.concatenate((descriptores_imagen, descriptor))

    return descriptores_imagen

def descriptores_color(img_color):

    #combinaciones = [(1,4),(1,8),(4,4),(8,8)]
    combinaciones = [(1,4),(4,4)]
    descriptor = []
    for c in combinaciones:
        if len(descriptor) == 0:
            descriptor = vector_de_colores(img_color, c[0], c[1])
        else:
            descriptor = numpy.concatenate((descriptor, vector_de_colores(img_color, c[0], c[1])))

    return descriptor

def descriptores_full(img_color):
    img_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    desc_gris = descriptores_gris(img_gris)
    desc_color = descriptores_color(img_color)

    desc_total = numpy.concatenate((desc_gris,desc_color))

    return desc_total