import cv2
import numpy as np
from noise import noise
import os

if __name__ == '__main__':
    path = 'C:/Users/ASUS-PC/Desktop' #Ruta de la imágen.
    image_name = 'lena.png' #Imágen que se desea leer.
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file) #Lee y guarda la imagen en la variable image.
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convierte la imagen a tonos de grises.

    #Imagenes con ruido
    image_gray_noisy = noise("gauss", image_gray.astype(np.float) / 255) #Llama a la clase noise y crea la imagen con ruido gaussiano.
    image_gray_noisy = (255 * image_gray_noisy).astype(np.uint8) #Escaliza la imagen.
    cv2.imshow('lena_gauss_noisy', image_gray_noisy) #Muestra la imágen con ruido.
    cv2.waitKey(0)

    #image_gray_noisy = noise("s&p", image_gray.astype(np.float) / 255) #Llama a la clase noise y crea la imagen con ruido sal y pimienta.
    #image_gray_noisy = (255 * image_gray_noisy).astype(np.uint8) #Escaliza la imagen
    #cv2.imshow('lena_s&p_noisy', image_gray_noisy) #Muestra la imágen con ruido.
    #cv2.waitKey(0)


    #Filtro Gaussiano
    image_gaussian_filter = cv2.GaussianBlur(image_gray_noisy,(7,7),1.5) #Aplica el filtro gaussiano a la imagen.
    cv2.imshow('GaussianImage', image_gaussian_filter) #Muestra la imagen con filtro gaussiano.
    cv2.waitKey(0)
    image_noise = abs(image_gray_noisy - image_gaussian_filter) #Calcula el ruido estimado.
    cv2.imshow('GaussianImageEst', image_noise) #Muestra la imagen de ruido estimado.
    cv2.waitKey(0)

    difference1 = (image_gray - image_gaussian_filter)**2 #Operación de resta entre las 2 imágenes y se eleva al cuadrado.
    error1 = (1/ (image_gray.shape[0] * image_gray.shape[1]))* np.sum(difference1) #Calculo del error cuadrático medio.
    sqrt_mse1 = np.sqrt(error1) #Raíz del error cuadrático medio.
    print(sqrt_mse1) #Muestra en pantalla la raíz del error cuadrático medio.

    initial_time_gaussian = time() #Tiempo de inicio del Filtro Gausiano.
    final_time_gaussian = time() #Tiempo de finalización del Filtro Gausiano.
    total_time_gaussian = final_time_gaussian - initial_time_gaussian #Tiempo de ejecución del Filtro Gausiano.
    print('El tiempo de ejecucion del Filtro Gaussiano fue:', total_time_gaussian) #Muestra en pantalla el tiempo de ejecución.



    #Filtro mediana
    image_median_filter = cv2.medianBlur(image_gray_noisy, 7) #Aplica el filtro mediana a la imagen.
    cv2.imshow('MedianImage', image_median_filter) #Muestra la imagen con filtro mediana.
    cv2.waitKey(0)
    image_noise = abs(image_gray_noisy - image_median_filter) #Calcula el ruido estimado.
    cv2.imshow('MedianImageEst', image_noise) #Muestra la imagen de ruido estimado.
    cv2.waitKey(0)

    difference2 = (image_gray - image_median_filter)**2 #Operación de resta entre las 2 imágenes y se eleva al cuadrado.
    error2 = (1/ (image_gray.shape[0] * image_gray.shape[1]))* np.sum(difference2) #Calculo del error cuadrático medio.
    sqrt_mse2 = np.sqrt(error2) #Raíz del error cuadrático medio.
    print(sqrt_mse2) #Muestra en pantalla la raíz del error cuadrático medio.

    initial_time_median = time() #Tiempo de inicio del Filtro Mediana.
    final_time_median = time() #Tiempo de finalización del Filtro Mediana.
    total_time_median = final_time_median - initial_time_median  #Tiempo de ejecución del Filtro Mediana.
    print('El tiempo de ejecucion del Filtro Mediana fue:', total_time_median) #Muestra en pantalla el tiempo de ejecución.



    #Filtro bilateral
    image_bilateral_filter = cv2.bilateralFilter(image_gray_noisy, 7, 25, 15) #Aplica el filtro bilateral a la imagen.
    cv2.imshow('BilateralImage', image_bilateral_filter) #Muestra la imagen con filtro bilateral.
    cv2.waitKey(0)
    image_noise = abs(image_gray_noisy - image_bilateral_filter) #Calcula el ruido estimado.
    cv2.imshow('BilateralImageEst', image_noise) #Muestra la imagen de ruido estimado.
    cv2.waitKey(0)

    difference3 = (image_gray - image_bilateral_filter)**2 #Operación de resta entre las 2 imágenes y se eleva al cuadrado.
    error3 = (1/ (image_gray.shape[0] * image_gray.shape[1]))* np.sum(difference3) #Calculo del error cuadrático medio.
    sqrt_mse3 = np.sqrt(error3) #Raíz del error cuadrático medio.
    print(sqrt_mse3) #Muestra en pantalla la raíz del error cuadrático medio.

    initial_time_bilateral = time() #Tiempo de inicio del Filtro Bilateral.
    final_time_bilateral = time() #Tiempo de finalización del Filtro Bilateral.
    total_time_bilateral = final_time_bilateral - initial_time_bilateral #Tiempo de ejecución del Filtro Bilateral.
    print('El tiempo de ejecucion del Filtro Bilateral fue:', total_time_bilateral) #Muestra en pantalla el tiempo de ejecución.



    #Filtro NLM
    image_nml_filter = cv2.fastNlMeansDenoising(image_gray_noisy, 5, 15, 25) #Aplica el filtro NLM a la imagen.
    cv2.imshow('NLMFilter', image_nml_filter) #Muestra la imagen con filtro nlm.
    cv2.waitKey(0)
    image_noise = abs(image_gray_noisy - image_nml_filter) #Calcula el ruido estimado.
    cv2.imshow('NMLImageEst', image_noise) #Muestra la imagen de ruido estimado.
    cv2.waitKey(0)

    difference4 = (image_gray - image_nml_filter)**2 #Operación de resta entre las 2 imágenes y se eleva al cuadrado.
    error4 = (1/ (image_gray.shape[0] * image_gray.shape[1]))* np.sum(difference4) #Calculo del error cuadrático medio.
    sqrt_mse4 = np.sqrt(error4) #Raíz del error cuadrático medio.
    print(sqrt_mse4) #Muestra en pantalla la raíz del error cuadrático medio.

    initial_time_nlm = time() #Tiempo de inicio del Filtro NLM
    final_time_nlm = time() #Tiempo de finalización del Filtro NLM
    total_time_nlm = final_time_nlm - initial_time_nlm #Tiempo de ejecución del Filtro NLM
    print('El tiempo de ejecucion del Filtro NLM fue:', total_time_nlm) #Muestra en pantalla el tiempo de ejecución.