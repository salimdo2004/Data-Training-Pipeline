# Generated from: Untitled47.ipynb
# Converted at: 2026-03-30T12:13:38.553Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# <a href="https://colab.research.google.com/github/salimdo2004/Data-Training-Pipeline/blob/main/Untitled47.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# 
# ***Chargement des données JSON***
# ---
# 
# 


#Charger les données du fichier Json dans une variable de type dictionnaire.
import json

with open('data.json', 'r', encoding='UTF-8') as file:
    Datas = json.load(file)
print(json.dumps(Datas, indent=2, ensure_ascii=False))

# ***Affichage des intentions***


# Afficher toutes les intentions dans le dictionnaire.

for intent in Datas["intents"]:
  print(intent['intent'])

# ***Création de la liste des classes (intentions)***


classes=[]
for intent in Datas['intents']:
  classes.append(intent['intent'])
print(classes)
print(len(classes))

# ***Extraction des questions (patterns)***



input_texts=[]
for intent in Datas['intents']:
  for question in intent['questions']:
    input_texts.append(question)
print(input_texts)
print(len(input_texts))

# ***Création des intentions correspondantes***



output_intentions=[]
for intent in Datas['intents']:
  for question in intent['questions']:

    output_intentions.append(intent['intent'])

print(output_intentions)
print(len(output_intentions))

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ***Tokenization des phrases***


nltk.download('punkt_tab')

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('french'))

ls = []

for intent in Datas['intents']:
    for question in intent['questions']:
        tokens = word_tokenize(question)

        for token in tokens:   # boucle sur chaque mot
            if token.lower() not in ls and token.lower() not in stop_words and token not in string.punctuation:
                ls.append(token.lower())

print(ls)
print(f"size tokens: {len(ls)}")

def clean_text(text):

    text = text.lower()

    tokens = word_tokenize(text)

    words = []

    for word in tokens:

        if word not in string.punctuation and word not in stop_words:
            word = lemmatizer.lemmatize(word)
            words.append(word)

    return words

# ***Suppression de la ponctuation***


ls = [t for t in ls if t not in string.punctuation]
print(ls)
print(f"size tokens : {len(ls)}")

# ***Lemmatisation des mots***



from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
tokens = []
lemmatizer = WordNetLemmatizer()
for token in ls:
  token=lemmatizer.lemmatize(token.lower())
  if token not in tokens:
    tokens.append(token)
print(tokens)
print(f"size tokens : {len(tokens)}")

# ***Création des données d'entraînement (Bag of Words)***


training = []
for mot in input_texts :
  vect_x=[0 for i in range(len(tokens))]
  t=word_tokenize(mot)
  vect_y=[0 for i in range(len(classes))]
  idx=input_texts.index(mot)
  intent=output_intentions[idx]
  idx=classes.index(intent)
  vect_y[idx]=1
  print(vect_y)
 # print(t)
  for item in t:
      if item not in string.punctuation:

        result=lemmatizer.lemmatize(item.lower())
        if result in tokens:
            idx=tokens.index(result)
            #print(idx)
            vect_x[idx]=1
  training.append([vect_x, vect_y])

  print(vect_x)
  print(len(input_texts))

# ***Mélange et conversion en tableau NumPy***


import random
import numpy as np

print(len(training))
# mélanger les données et les convertir en array
random.shuffle(training)
training = np.array(training, dtype=object)
print(training)

print(input_texts)
print(output_intentions)
print(classes)

for mot in input_texts :
  vect_y=[0 for i in range(len(classes))]
  idx=input_texts.index(mot)
  intent=output_intentions[idx]
  idx=classes.index(intent)
  vect_y[idx]=1
  print(vect_y)
print(len(input_texts))


# ***Séparation des données d'entraînement***


train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

print(train_x)
print( train_y)

# ***Paramètres du modèle***


input_size = (len(train_x[0]),)
output_shape = len(train_y[0])
epochs = 200

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# ***Création du modèle Deep Learning***


model = Sequential()

model.add(Dense(128, input_shape=input_size, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))

model.add(Dense(output_shape, activation="softmax"))

print(model.summary())

# ***Entraînement du modèle***


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(train_x, train_y, epochs=epochs, verbose=1)

# ***Sauvegarde du modèle***


model.save("chatbot_model.keras")


# ***Chargement du modèle***


import tensorflow as tf
model2 = tf.keras.models.load_model("chatbot_model.keras")


model2 = tf.keras.models.load_model("chatbot_model.keras")

exemple = model.predict(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
).reshape(1, -1))


print(exemple)

idx = np.argmax(exemple)
print(idx)

print(classes[idx])

# ***Fonction de transformation du texte (vectorisation)***


def input_presentation(text):

    words = clean_text(text)

    vect_x = [0] * len(tokens)

    for word in words:

        if word in tokens:
            idx = tokens.index(word)
            vect_x[idx] = 1

    return vect_x

def input_presentation(text):
  t=word_tokenize(text)

  vect_x=[0]*len(tokens)
  for item in t:
        if item not in string.punctuation:
          result=lemmatizer.lemmatize(item.lower())
          if result in tokens:
              idx=tokens.index(result)
              #print(idx)
              vect_x[idx]=1
  return vect_x

print(input_presentation("farewell"))

reponse=[item['responses'] for item in Datas['intents']]
print(reponse)

# ***Fonction de prédiction***


def pred_f(text):

    vect = input_presentation(text)

    if vect == [0]*len(tokens):
        return "Je ne comprends pas votre question", 0

    prediction = model2.predict(np.array(vect).reshape(1,-1))

    idx = np.argmax(prediction)

    confidence = max(prediction[0]) * 100

    if confidence < 60:
        return "Je ne suis pas sûr de comprendre", confidence

    return random.choice(reponse[idx]), confidence

# **`*Test du chatbot*`**


print(pred_f("Orientation académique"))

# ***Interface console du chatbot***


while True:
  user_input = input("Entrez une phrase : ")
  if user_input == "exit":
    break
  y, score = pred_f(user_input)
  print(f"bot: {y}\n --> {score:.2f}")

from textblob import TextBlob

pip install symspellpy



from symspellpy import SymSpell, Verbosity

# Initialisation avec un dictionnaire pré-entraîné
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Chargement du dictionnaire
dictionary_path = "google-10000-english.txt"
if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
    print("Erreur : Impossible de charger le dictionnaire")
else:
    # Mot mal orthographié
    misspelled_word = "prica"

    # Correction avec SymSpell
    suggestions = sym_spell.lookup(misspelled_word, Verbosity.CLOSEST, max_edit_distance=2)

    # Affichage du meilleur résultat
    if suggestions:
        print(f"Correction suggérée : {suggestions[0].term}")
    else:
        print("Aucune suggestion trouvée")
