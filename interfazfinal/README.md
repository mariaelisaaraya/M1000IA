# 🚀 NuevaAPI

Bienvenido a **Nuevapi**. Esta API se integra con una interfaz HTML final y permite acceder a sus funcionalidades de manera sencilla. A continuación, se presentan las instrucciones para ejecutar la API de forma rápida.

## 🛠️ Requisitos Previos

Asegúrate de tener instalados lo siguiente en tu sistema:

- [Python](https://www.python.org/) (versión 3.6 o superior)
- [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) (extensión para Visual Studio Code)

## 📁 Estructura del Proyecto

- **`index.html`**: Archivo principal que proporciona la interfaz de usuario.
- **`nuevapi`**:  API nueva , donde ejecutas y listo.
- **`requirements.txt`**: Archivo que contiene las dependencias necesarias para el entorno de Python.
- **`README.md`**: Este archivo, que documenta el proyecto.

## 📦 Dependencias

Asegúrate de instalar las dependencias necesarias para el funcionamiento de la API. Las dependencias están listadas en el archivo `requirements.txt`. Puedes instalar estas dependencias utilizando el siguiente comando:

```bash
pip install -r requirements.txt

##📋 Contenido de requirements.txt
Aquí hay un ejemplo de lo que podrías encontrar en requirements.txt:
numpy==1.26.2
PyTorch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1


##🚀 Instrucciones para Ejecutar la API
A - Navegar a la Carpeta del Proyecto: Abre tu terminal o consola y navega a la carpeta donde se encuentra NuevaAPI. 
    #Usa el siguiente comando: 
B - 
    ejecutar archivo de python que esta arriba en el dibujo de play 
                  O
    escribes en terminal python nuevapi.py Y....

    ---LISTO YA ESTA EJECUTANDOSE!!!!---

C--Iniciar el Servidor: Para ejecutar la API, asegúrate de utilizar el archivo index.html que se encuentra en la carpeta principal de NuevaAPI. 

 Abrir la Interfaz:
 situate en el archivo index.html Puedes iniciar Live Server desde Visual Studio Code. Abre index.html y haz clic en "Go Live" en la parte inferior derecha.

 ## Asegúrate de que Live Server esté instalado y ejecútalo desde la barra de estado en la parte inferior derecha de Visual Studio Code.

⚠️ Nota: No uses ningún otro archivo index que no sea el de esta carpeta, ya que esto podría causar errores en la ejecución.
 

D -Verificar la API: Una vez que el servidor esté en funcionamiento
    http://localhost:5000/api/welcome

    #Deberías ver una respuesta similar a esta:

    {
    "message": "¡Bienvenido a la API!"
    }



######/////si no te funciona ////-------------------
si no te funciona asi o queres ejecutarlo de otra manera podes ejecutar la api con otro comando:
    1- Abre tu terminal o consola y navega a la carpeta donde se encuentra NuevaAPI.
    2- Instalar las Dependencias: Ejecuta el siguiente comando para instalar las dependencias necesarias: ## solo si no te funciona
    en terminal colocas el siguiente comando en caso que no te funcione solo con ejecutar :
    pip install -r requirements.txt
    3- ejecutas la api como dice arriba en ##🚀 Instrucciones para Ejecutar la API

--------------------####///////////------------------------------------



📚 Uso de la API
Para utilizar la API, consulta la lógica implementada en la carpeta api. Asegúrate de seguir la estructura y las rutas definidas en el código para acceder a las funcionalidades.

🤝 Contribuciones
Si deseas contribuir a NuevaAPI, no dudes en abrir un issue o enviar un pull request. Tu ayuda es bienvenida.

