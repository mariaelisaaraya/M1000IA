<h1 align="center">PROYECTO DermAI</h1>

1. Carpetas Imagenes sin nevos.

   1.1. Se subio al [drive](https://drive.google.com/drive/folders/1TwvbjPg7L-1Bkbd98IqCnHw8GM8IpQBH?usp=drive_link)  la imagenes el total de las mismas son 5021. Corroborar que todas usemos las mismas imagenes. Estas son las imagenes sin nevos.

3. Utilizar Metadatos filtrados.
   
    2.1. Se puede recortar los metadatos con: modelado/recortenv.py
   
    2.2. Se puede utilizar directamente los metadatos recortados actualizados de la fecha del [16/10](https://github.com/mariaelisaaraya/M1000IA/blob/master/16_10_metadatos_actualizados_sin_nv_reducidos.csv)

4. Correr.

    3.1. Kat modelado/model2.py, subir TXT resultados
    
    3.2. Eli - subir TXT resultados -> Corriendo model1Eli.py

    3.3. Marce el suyo con metadatos e imagenes filtradas, subir TXT resultados

    3.4. Lu el suyo con metadatos e imagenes filtradas, subir TXT resultados


# Guía para Levantar la Interfaz de la Aplicación

## Requisitos Previos

1. Asegúrate de que el entorno virtual esté activado (debería verse como `(env)` en la terminal).
2. Debes estar dentro de la carpeta `interfaz`.

## Pasos para Ejecutar la Aplicación

1. Abre la terminal en la parte inferior (donde dice `bash - interfaz`).
2. Ejecuta el siguiente comando para correr la aplicación:

   ```bash
   python apiokc.py
   ```
## Instalación de Dependencias

Si encuentras errores relacionados con Flask o CORS, asegúrate de instalarlos dentro del entorno virtual:

```bash
pip install Flask
pip install flask-cors
```

Aquí tienes todo el contenido en formato Markdown, listo para que lo copies y lo pegues en tu archivo README.md.

markdown
Copiar código
# Guía para Levantar la Interfaz de la Aplicación

## Requisitos Previos

1. Asegúrate de que el entorno virtual esté activado (debería verse como `(env)` en la terminal).
2. Debes estar dentro de la carpeta `interfaz`.

## Pasos para Ejecutar la Aplicación

1. Abre la terminal en la parte inferior (donde dice `bash - interfaz`).
2. Ejecuta el siguiente comando para correr la aplicación:

```bash
python apiokc.py
```
   
Instalación de Dependencias
Si encuentras errores relacionados con Flask o CORS, asegúrate de instalarlos dentro del entorno virtual:

```bash
pip install Flask
pip install flask-cors
```

## Errores Comunes

1. Error: El archivo modelo_entrenadomok.pth no se encuentra
Verifica que el archivo modelo_entrenadomok.pth esté en la misma carpeta que el script apiokc.py. Si está en otra carpeta, debes proporcionar la ruta completa o mover el archivo al directorio correcto.

2. Verifica el Funcionamiento del Servidor Flask
Tu aplicación Flask está funcionando correctamente en el servidor de desarrollo. Puedes acceder a ella abriendo un navegador web y visitando la dirección:

```bash
http://127.0.0.1:5000
```

Puedes probar la ruta básica:

```bash
@app.route('/')
def index():
    return "Bienvenido a la aplicación Flask"
```

## Error al Abrir el HTML

Si abres el archivo HTML usando file://, no funcionará correctamente. Usa un servidor local como http-server para servir tu frontend o ábrelo desde Flask si es posible. También puedes usar un servidor local como Live Server en Visual Studio Code (disponible como una extensión que puedes instalar).

## Flujo Completo de la Aplicación

1. El usuario carga una imagen en index.html.
2. app.js envía los datos al servidor Flask en la ruta /predict.
3. Flask procesa los datos, predice el resultado y devuelve una respuesta.
4. El usuario es redirigido a resultado.html con los resultados.
5. appb.js muestra los resultados en la página.
