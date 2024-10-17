import os
import pandas as pd
import numpy as np
import h5py

#leer archivos metadatos


#leer datos de ambos datasets
df = pd.read_csv('HAM10000_metadata_cleaned.csv')
df2 = pd.read_csv('imagenes_mel_bcc.csv')

print("datasetOriginal:", df.columns)
print("dataset2:", df2.columns)

#poner extension a los nombres de imagenes en el dataset original
df['image_id'] = df['image_id'].apply(lambda x: x + '.jpg') 

# Eliminar la columna dx_type de datasetOriginal
df.drop(columns=['dx_type'], inplace=True)

# Renombrar las columnas de dataset2
df2.rename(columns={
    'diagnostic': 'dx',
    'gender': 'sex',
    'region': 'localization',
    'img_id': 'image_id'
}, inplace=True)



# Unir ambos datasets
dataset_unido = pd.concat([df, df2], ignore_index=True)
# Convertir los valores de la columna 'dx' a minúsculas
dataset_unido['dx'] = dataset_unido['dx'].str.lower()

# Guardar el dataset unido en un archivo CSV
dataset_unido.to_csv('dataset_unido3.csv', index=False)


