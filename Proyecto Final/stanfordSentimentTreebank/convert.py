import pandas as pd
from sklearn.model_selection import train_test_split

# Ruta a los archivos en la carpeta 'stanfordSentimentTreebank'
dictionary_path = '/Users/rosannabm/Desktop/Inteligencia-Artificial/Proyecto Final/stanfordSentimentTreebank/dictionary.txt'
sentiment_labels_path = '/Users/rosannabm/Desktop/Inteligencia-Artificial/Proyecto Final/stanfordSentimentTreebank/sentiment_labels.txt'

# Cargar el archivo de frases
phrases = pd.read_csv(dictionary_path, sep='|', header=None, names=['phrase', 'phrase_id'])

# Cargar el archivo de etiquetas
labels = pd.read_csv(sentiment_labels_path, sep='|', header=0)

# Unir las frases con sus etiquetas
data = pd.merge(phrases, labels, left_on='phrase_id', right_on='phrase ids')
data = data[['phrase', 'sentiment values']]

# Convertir las etiquetas de sentimiento en etiquetas binarias (0 o 1)
data['label'] = data['sentiment values'].apply(lambda x: 1 if x >= 0.5 else 0)
data = data[['phrase', 'label']]
data.columns = ['sentence', 'label']

# Dividir en conjuntos de entrenamiento y validación
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# Guardar en archivos CSV para facilitar la carga más tarde
train_data.to_csv('/Users/rosannabm/Desktop/Inteligencia-Artificial/Proyecto Final/stanfordSentimentTreebank/train_data.csv', index=False)
val_data.to_csv('/Users/rosannabm/Desktop/Inteligencia-Artificial/Proyecto Final/stanfordSentimentTreebank/val_data.csv', index=False)

print("Archivos CSV generados exitosamente.")