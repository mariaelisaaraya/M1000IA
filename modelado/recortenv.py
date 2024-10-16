import pandas as pd

# Cargar el dataset
ruta_archivo = 'metadatos_actualizados_limpios_normalizados.csv'
df = pd.read_csv(ruta_archivo)

# Filtrar los registros que tienen la etiqueta 'nv' en la columna 'dx'
df_nv = df[df['dx'] == 'nv']

# Verificar cuántos registros 'nv' hay
cantidad_nv = len(df_nv)

# Verificar si hay suficientes registros 'nv' para eliminar 1000
if cantidad_nv > 1000:
    # Eliminar aleatoriamente 1000 registros con etiqueta 'nv'
    df_nv_a_eliminar = df_nv.sample(1000, random_state=42)
    # Crear un nuevo DataFrame sin los 1000 registros 'nv' seleccionados
    df_final = df.drop(df_nv_a_eliminar.index)
else:
    print(f"Solo hay {cantidad_nv} registros con 'nv', eliminando todos.")
    # Si no hay suficientes, eliminamos todos los registros 'nv'
    df_final = df[df['dx'] != 'nv']

# Guardar el dataset actualizado sin los 1000 registros 'nv'
df_final.to_csv('metadatos_actualizados_sin_nv_reducidos.csv', index=False)

print("Se han eliminado 1000 registros con la etiqueta 'nv' (o todos si eran menos de 1000).")
