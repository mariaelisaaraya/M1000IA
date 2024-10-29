<h1 align="center">PROYECTO DermAI</h1>

- [Las imagenes que se usan de prueba](https://github.com/mariaelisaaraya/M1000IA/tree/2e90d79719e22be8ebb1adfe6bec1046ff995cfe/test/test_images)
- [Archivo que se utiliza con pytest para hacer las pruebas](https://github.com/mariaelisaaraya/M1000IA/blob/2e90d79719e22be8ebb1adfe6bec1046ff995cfe/testModel/test_predictModel.py)


# Guía para Levantar la Interfaz de la Aplicación 
- ### Carpeta Interfaz

## Requisitos Previos

1. Asegúrate de que el entorno virtual esté activado (debería verse como `(env)` en la terminal).
2. Debes estar dentro de la carpeta `interfaz`.

## Pasos para Ejecutar la Aplicación

1. Abre la terminal en la parte inferior (donde dice `bash - interfaz`).
2. Ejecuta el siguiente comando para correr la aplicación:

   ```bash
   python apiokc.py
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

3. Si encuentras errores relacionados con Flask o CORS, asegúrate de instalarlos dentro del entorno virtual:

```bash
pip install Flask
pip install flask-cors
```

## Error al Abrir el HTML

Si abres el archivo HTML usando file://, no funcionará correctamente. Usa un servidor local como http-server para servir tu frontend o ábrelo desde Flask si es posible. También puedes usar un servidor local como Live Server en Visual Studio Code (disponible como una extensión que puedes instalar).

## Flujo Completo de la Aplicación

1. El usuario carga una imagen en index.html.
2. app.js envía los datos al servidor Flask en la ruta /predict.
3. Flask procesa los datos, predice el resultado y devuelve una respuesta.
4. El usuario es redirigido a resultado.html con los resultados.
5. appb.js muestra los resultados en la página.
