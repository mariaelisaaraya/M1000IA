
# Flask API para Predicción con Imágenes y Metadatos

Este proyecto es una API construida en Flask que permite realizar predicciones utilizando un modelo de aprendizaje profundo que recibe una imagen y metadatos (edad y sexo). El modelo procesa la imagen a través de una red neuronal convolucional (CNN) y combina la información de los metadatos para realizar la predicción.

## Estructura del Proyecto

```
.
├── app.py                # Archivo principal de la aplicación Flask
├── model.py              # Definición y carga del modelo
├── prediction.py         # Lógica de predicción
├── uploads/              # Carpeta donde se almacenan las imágenes subidas
└── requirements.txt      # Lista de dependencias
```

### Requisitos Previos

- Python 3.x
- Pip (Administrador de paquetes de Python)

### Instalación

1. Clona este repositorio o descarga los archivos.

   ```bash
   git clone https://github.com/mariaelisaaraya/M1000IA.git
   ```

2. Entra al directorio del proyecto.

   ```bash
   cd M1000IA
   ```

3. Instala las dependencias requeridas:

   ```bash
   pip install -r api/requirements.txt
   ```

4. Asegúrate de que tengas el modelo entrenado `modelo_entrenadomok.pth` en el directorio raíz del proyecto. Si no lo tienes, tendrás que entrenar uno y guardarlo con este nombre.

5. Crea la carpeta `api/uploads/` donde se almacenarán las imágenes subidas por el usuario (si no se crea automáticamente al iniciar la app):

   ```bash
   mkdir api/uploads
   ```

### Ejecución

Para ejecutar la aplicación, simplemente ejecuta el siguiente comando en la raíz del proyecto:

```bash
python api/app.py
```

Esto iniciará un servidor local que estará disponible en `http://localhost:5000`.

### Endpoints

#### 1. Verificación de estado del servidor

**Ruta:** `/health`  
**Método:** `GET`

Este endpoint verifica si la API está funcionando correctamente.

**Ejemplo de respuesta:**
```json
{
  "status": "API is running"
}
```

#### 2. Predicción con imagen y metadatos

**Ruta:** `/predict`  
**Método:** `POST`  
**Parámetros:**
- `image`: La imagen a subir (debe ser una imagen en formato `png`, `jpg`, `jpeg` o `gif`).
- `edad`: La edad de la persona (valor numérico).
- `sexo`: El sexo de la persona (`male` o `female`).

**Ejemplo de uso:**
Puedes hacer una petición con herramientas como **Postman** o usando **cURL** en la terminal:

```bash
curl -X POST http://localhost:5000/predict   -F "image=@/ruta/a/tu_imagen.jpg"   -F "edad=30"   -F "sexo=male"
```

**Ejemplo de respuesta exitosa:**
```json
{
  "message": "Prediction successful",
  "predicted_class": 0,
  "predicted_probabilities": [0.95, 0.05]
}
```

### Manejo de Errores

- Si se sube un archivo que no es una imagen o se omiten los metadatos necesarios (`edad`, `sexo`), el servidor responderá con un código de estado `400` y un mensaje de error correspondiente.

**Ejemplo de respuesta de error:**
```json
{
  "error": "Age and Sex are required"
}
```

### Pruebas

Desde la raíz del proyecto para probar que la API funciona correctamente levantar la API y en terminal ejecutar
```bash
pytest
```

### Notas Adicionales

- **Límites de tamaño de archivo**: Asegúrate de que las imágenes subidas no sean demasiado grandes para evitar problemas de rendimiento.
- **Extensiones permitidas**: El servidor solo acepta imágenes con las siguientes extensiones: `png`, `jpg`, `jpeg`, `gif`.
