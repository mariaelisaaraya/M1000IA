
import pickle
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
import uvicorn
from multiprocessing import Process
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('wordnet')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from fastapi.middleware.cors import CORSMiddleware

# Definición de la clase para el input del modelo
class SentimentRequest(BaseModel):
    text: str
# función para limpiar el texto
nltk.download('stopwords')
nltk.download('punkt')
# Función para limpiar el texto
def clean_text(text):
    # Eliminar las etiquetas HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Eliminar caracteres no alfabéticos
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convertir el texto a minúsculas
    text = text.lower()
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    # Lematización
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)


# Carga del modelo y el vectorizador
def load_model_and_vectorizer():
    try: 
        with open('proyectoM1000IA\Mymodel.pkl', 'rb') as f:
        
            model = pickle.load(f)
            if not hasattr(model, 'predict'):
                raise ValueError("El archivo model.pkl no contiene un modelo válido.")
        
        with open('proyectoM1000IA/Myvectorizer.pkl', 'rb') as f:
        
            vectorizer = pickle.load(f)
            if not hasattr(vectorizer, 'transform'):
                raise ValueError("El archivo vectorizer.pkl no contiene un vectorizador válido.")
        
        return model, vectorizer
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo y el vectorizador: {str(e)}")

model, vectorizer = load_model_and_vectorizer()

# Inicio de la API con FastAPI
app = FastAPI()
#configuraciones de la api
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto según las necesidades de tu aplicación
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Asegúrate de incluir POST si es necesario
    allow_headers=["*"],
)
# Endpoint para análisis de sentimientos
@app.post("/sentiment/")
def predict_sentiment(request: SentimentRequest):
    try:
        # limpiar texto
        cleaned_text = clean_text(request.text)
        # Transformar el texto usando el vectorizador
        #vectorized_text = vectorizer.transform([request.text])
        vectorized_text = vectorizer.transform([cleaned_text])
        # Realizar la predicción
        prediction = model.predict(vectorized_text)

        # Devolver la predicción como una respuesta
        #sentiment_label = "positivo" if prediction[0] == 1 else "negativo"
        #sentiment_label = "positivo" if prediction[0] == "Positive" else "negativo"
        
        return {"sentiment": prediction[0]}
        #return {"sentiment": sentiment_label }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecución del servidor de la API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
