Lesiones con más de dos imágenes


Dataset con 3 columnas (lesion_id, lista de id imagenes, cantidad de imagenes) filtrado para cantidad_imagenes > 2 -> Resumen-IdImagenPorLesion.csv

Dataset con Metadatos originales (lesion_id, image_id, dx, dx_type, age, sex, localization) + cantidad_imagenes  filtrados:

 por cantidad_imagenes >2 (LesionesConMas2IMG.csv)

 por cantidad_imagenes = 3 (LesionesCon3IMG.csv)
 por cantidad_imagenes = 4 (LesionesCon4IMG.csv)
 por cantidad_imagenes = 5 (LesionesCon5IMG.csv)
 por cantidad_imagenes = 6 (LesionesCon6IMG.csv)

En el notebook https://colab.research.google.com/drive/12YaW2DawqVavVrJaQjFSuwvjhly2jlgG?usp=sharing  está el Código usado para generar los archivos