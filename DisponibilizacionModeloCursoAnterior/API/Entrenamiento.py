import pickle
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#from gensim.models import Word2Vec, KeyedVectors
#import gensim.downloader as api

# Configuración visual para los gráficos
#sns.set(style="whitegrid")

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
# Cargar el dataset

url = "https://raw.githubusercontent.com/mvegetti/IAcourse/main/20191226-reviews.csv"

url2 = "https://raw.githubusercontent.com/mvegetti/IAcourse/main/20191226-items.csv"

Review = pd.read_csv(url)
Items = pd.read_csv(url2)

Amazon = pd.merge(Review, Items, on="asin", how="outer")

# Cálculo de la variable sentiment
# dependiendo del valor de la variable rating, se clasificará la revisión en: "positiva", "neutral" o "negativa"

Amazon['sentiment'] = Amazon['rating_x'].apply(lambda x: 'Positive' if x > 3 else('Neutral' if x == 3  else 'Negative'))

# Eliminar filas con valores faltantes en la columna 'body'
Amazon = Amazon.dropna(subset=['body'])

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

# Aplicar la función de limpieza al DataFrame
Amazon['cleaned_body'] = Amazon['body'].apply(clean_text)

# Filtrar el DataFrame eliminando las filas donde 'sentiment' es 'Neutral'
Amazon = Amazon[Amazon['sentiment'] != 'Neutral']

#Separación dataframe
X_train, X_test, y_train, y_test = train_test_split(Amazon['cleaned_body'], Amazon['sentiment'], test_size=0.2, random_state=42)
# Crear el dataframe de entrenamiento
df_train = pd.DataFrame({'cleaned_body': X_train, 'sentiment': y_train})

# Crear el dataframe de prueba
df_test = pd.DataFrame({'cleaned_body': X_test, 'sentiment': y_test})

# Vectorización con TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Se Puede ajustar el número de características según sea necesario
X_tfidf_train = tfidf_vectorizer.fit_transform(X_train)
X_tfidf_test = tfidf_vectorizer.transform(X_test)

# Entrenamiento del modelo
model = LogisticRegression(multi_class='ovr', max_iter=1000)  # Aumentamos max_iter para asegurar la convergencia
model.fit(X_tfidf_train, y_train)
# Predicciones
y_pred = model.predict(X_tfidf_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo'])

print(f'Exactitud: {accuracy}')
print('Classification Report:')
print(report)
def predict_sentiment(review):
    cleaned_review = clean_text(review)
    vectorized_review = tfidf_vectorizer.transform([cleaned_review])
    prediction = model.predict(vectorized_review)
    return prediction[0]

positive_review = "This phone is fantastic! I love the battery life and the camera."

print(f'Review: {positive_review}\nSentiment: {predict_sentiment(positive_review)}')

positive_review = "This phone is incredible ."
print(f'Review: {positive_review}\nSentiment: {predict_sentiment(positive_review)}')

positive_review = "This phone is excellent!. It has a very good battery life"
print(f'Review: {positive_review}\nSentiment: {predict_sentiment(positive_review)}')

positive_review = "This phone is horrible"
print(f'Review: {positive_review}\nSentiment: {predict_sentiment(positive_review)}')

with open('Myvectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('Mymodel.pkl', 'wb') as f:
    pickle.dump(model, f)
