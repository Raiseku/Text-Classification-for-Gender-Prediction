# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:02:39 2020

@author: Raise
"""

# Apparently you may use different seed values at each stage
seed_value= 0

import os
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)

from keras import backend as K

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

import xml.etree.ElementTree as et 
import pandas as pd
pd.set_option('display.max_rows',50)
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import string
import re
import nltk
from keras.layers import Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText

import emoji

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Conv1D, Flatten,Dropout, GlobalMaxPooling1D

xtree = et.parse("Dataset_def.xml")
xroot = xtree.getroot()

df_cols = ["testo", "sesso"]
righe = []

for node in xroot: 
    id = node.attrib.get("id")
    genere = node.attrib.get("genre")
    sesso = node.attrib.get("gender")
    testo = node.text
    righe.append({"testo": testo, "sesso": sesso})
    
df = pd.DataFrame(righe, columns = df_cols)

print(df.head())


print()
def clean_text(text):  
    text = " " + emoji.demojize(text, delimiters=(" ", " ")) 
    text = text.translate(string.punctuation)
    #Converto il testo in minuscolo e lo divido parola per parola
    #definisco quali sono le stopwords, quindi dal vocabolario mi prende tutte le stopwords in italiano
    stops = set(stopwords.words("italian-Progetto"))
    text = re.sub(r"'", " ' ", text)
    text = text.lower().split()
    text = [w for w in text if not w.lower() in stops]
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r":// ", "://", text) 
    text = re.sub(r" /", "/", text) 
    text = re.sub(r"/ ", "/", text) 
    text = re.sub(r"0 4", "04", text)
    text = re.sub(r"fb", "facebook", text)
    text = re.sub(r"i t", "it", text)
    text = re.sub(r" status", "status", text)
    text = re.sub(r"www. ", "www", text)
    text = re.sub(r"est eri ", "esteri", text)
    text = re.sub(r" status", "status", text)
    text = re.sub(r"status ", "status", text)
    text = re.sub(r"s tatus", "status", text)
    text = re.sub(r"st atus", "status", text)
    text = re.sub(r"sta tus", "status", text)
    text = re.sub(r"stat us", "status", text)
    text = re.sub(r"statu s", "status", text)
    text = re.sub(r"make- up ", "make-up", text)
    text = re.sub(r"pic.twitter.com", "http://pic.twitter.com", text)
    text = re.sub(r"http\S+", "", text) #sia http
    text = re.sub(r"html\S+", "", text) #sia http
    text = re.sub(r"https\S+", "", text) #sia https
    text = re.sub(r"<3", " cuore ", text)
    text = re.sub(r"_", "\_", text)
    text = re.sub(r"\\n", " ", text)     
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", "! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"-", "", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\+", "", text)  
    text = re.sub(r"\°", "", text) 
    text = re.sub(r"\s{2,}", " ", text)  
    text = re.sub(r"\#", "", text)  #Tolgo gli Ashtag
    text = re.sub(r"\@", "", text)  #Tolgo i Tag

    return text

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

frase_selezionata = 657
print("Testo originale:")
print(df['testo'].iloc[frase_selezionata])


# Applico la funzione per la pulizia del testo a tutto quello dentro la colonna 'testo'
df['testo'] = df['testo'].map(lambda x: clean_text(x))
df['testo'] = df[df['testo'] != ""]


# Nel file M indicava Maschio F Femmina quindi inserico i maschi a 0 e le femmine ad 1
print()
print("Testo filtrato:")
print()
print(df['testo'].iloc[frase_selezionata])
print()
print()
print()
df["sesso"].replace({"M": 0, "F": 1}, inplace=True)
solo_maschi = df[df["sesso"] == 0]
conta_maschi = solo_maschi.shape
print("Quelli classificati come Maschi nel Dataset sono:", conta_maschi)
solo_femmine = df[df["sesso"] == 1]
conta_femmine = solo_femmine.shape
print("Quelli classificati come Femmine nel Dataset sono:", conta_femmine)


from sklearn.utils import shuffle
# Effettuo lo shuffle del dataset così da fare in modo che al modello non vengano proposti
# prima tutti i maschi e poi tutte le femmine o viceversa.


df = shuffle(df)
print(df.head())


# Metti tutto quello che trovi nella colonna 'testo' nella lista chiamata lista_testo
lista_testo = df["testo"].fillna('').to_list() 

# prendo tutti i valori della mia lista e li casto a stringa
lista_testo = [str(i) for i in lista_testo] 
lista_sesso = df["sesso"].fillna('').to_list() 

random_state = 42
# Effettuo lo splitting dei dati, in questo caso avremo che l'80% è il per training e 20% per il testing.
text_train, text_test, y_train, y_test = train_test_split(lista_testo,
                                                          lista_sesso,
                                                          test_size=0.2,
                                                          random_state=random_state
                                                          )



# Inizializzo il Tokenizer
# Qui traduco tutte le frasi in vettori del tipo [[1,2,3,4,5,6]] se la frase ha 6 parole
# Mi salvo in vocab_size il numero di parole uniche presenti nel testo analizzato

# Rimpiazzami la lista X_train con le parole contenute in X_train aggiungendo POST padding,
# ovvero tanti 0 per fare in modo che tutte le parole siano della stessa lunghezza
#Voglio tutte le parole della stessa lunghezza quindi aggiungo zero padding

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lista_testo)
vocab_size = len(tokenizer.word_index) + 1
X_train = tokenizer.texts_to_sequences(text_train)
X_test  = tokenizer.texts_to_sequences(text_test)
print()
print("Le parole all'interno del vocabolario sono: ", vocab_size) 

# Mi calcolo la lunghezza massima delle frasi sia nel Training che nel Testing
maxlen_train_X = max( len(x) for x in X_train)
maxlen_test_X = max( len(x) for x in X_test)

print("Lunghezza massima delle frasi nel train set: ", maxlen_train_X) 
print("Lunghezza massima delle frasi nel test set: ", maxlen_test_X) 
print()
# ZERO PADDING: 
maxlen_X = maxlen_train_X #mi salvo la lunghezza massima della frase nel training set
print("Frase selezionata:")
print(X_train[1])
print("La lunghezza prima del Post Padding è: ", len(X_train[1]))
print()
X_train = pad_sequences(X_train, padding = "post", maxlen = maxlen_X)
print(X_train[1])
print("La lunghezza dopo il Post Padding è: ", len(X_train[1]))
X_test = pad_sequences(X_test, padding = "post", maxlen = maxlen_X)
print()

#________________________ INIZIO PARTE DI DEFINIZIONE DEI MODELLI ______________________#
#_______________________________________________________________________________________#

#Modello 1
embedding_dim_modello_50 = 50
num_epoche = 8
batch_size = 250
neuroni_dense_1 = 32
neuroni_dense_2 = 16


def create_model():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim_modello_50, input_length = maxlen_train_X, trainable = True))
    model.add(Flatten())
    model.add(Dense(neuroni_dense_1, activation="relu", bias_initializer='zeros'))
    model.add(Dense(neuroni_dense_2, activation="relu", bias_initializer='zeros'))
    model.add(Dense(32, activation="relu", bias_initializer='zeros'))
    model.add(Dense(16, activation="relu", bias_initializer='zeros'))
    model.add(Dense(8, activation="relu", bias_initializer='zeros'))
    model.add(Dense(1, activation="sigmoid", bias_initializer='zeros'))
    model.compile(optimizer='adam' ,loss="binary_crossentropy", metrics=["accuracy"])
    return model
    

model_50 = create_model()
model_50.summary()
history = model_50.fit(np.array(X_train), np.array(y_train), epochs=num_epoche, verbose=True, batch_size=batch_size)
loss, accuracy = model_50.evaluate(np.array(X_test), np.array(y_test), verbose = 1)

#Ultima parte: Predizioni e valutazione del modello creato
y_pred = model_50.predict_classes(X_test, batch_size = 20, verbose = 1)
errori_commessi = 0
corretto = 0
count = 0
print()
for i in range(0, y_pred.size):
    count += 1
    if(np.around(y_pred[i], decimals = 0) != y_test[i]):
        errori_commessi+=1
    else:
        corretto+=1

cm = confusion_matrix(y_test,y_pred)


print()
print()
print('Utilizzati come training set: ', np.array(y_train).size)
print('Utilizzati come test set: ', np.array(y_pred).size)
print()
print('Previsioni Corrette: ', corretto)
print('Errori Commessi: ', errori_commessi)
accuratezza = round((corretto/np.array(y_pred).size*100),2)
print('Accuratezza: ', accuratezza,'%')
print()
print(classification_report(y_test,y_pred))
print('Matrice di Confusione:')
print(cm)
print()



plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5), dpi = 130)
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.title('Accuracy durante il training')
    plt.xlabel('Numero di epoche')
    plt.ylabel('Accuratezza')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.title('Loss durante il training')
    plt.xlabel('Numero di epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)




#_____________________________________________________________________________#
#_____________________________________________________________________________#
#______________________MODELLO CON 100 FEATURE PER PAROLA_____________________#
#_____________________________________________________________________________#
#_____________________________________________________________________________#


embedding_dim_modello_100 = 100
embeddings_index = dict()
f = open("dict/modello_100_feature.txt", encoding = 'utf-8')
for line in f:
    values = line.split() 
    try:
        word = values[0] #in posizione 0 c'è la parola
        coefs = np.array(values[1:], dtype = "float32") #tutto il resto sono valori
        embeddings_index[word] = coefs
        #print(embeddings_index[word])
    except ValueError:
        #print(values)
        continue
f.close()

#Creo la matrice dei pesi della dimensione adatta e la riempio di zero
embeddings_matrix = np.zeros((vocab_size, embedding_dim_modello_100)) 

for word, index in tokenizer.word_index.items():
  if index > vocab_size-1:
    break
  else:
    embedding_vector = embeddings_index.get(word)
    #se il mio vettore non è nullo lo metto nella matrice
    if(embedding_vector) is not None:
      #per ogni parola del vocabolario prendo la prappresentazione del dizionario già pre-allenato
      embeddings_matrix[index] = embedding_vector
      



model_100 = Sequential()
model_100.add(Embedding(vocab_size, embedding_dim_modello_100, input_length=maxlen_X, weights = [embeddings_matrix], trainable = False))
model_100.add(Flatten())
model_100.add(Dense(neuroni_dense_1, activation="relu"))
model_100.add(Dense(neuroni_dense_2, activation="relu"))
model_100.add(Dense(1, activation="sigmoid"))


model_100.summary()
model_100.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model_100.fit(np.array(X_train), np.array(y_train), epochs=num_epoche,verbose=True, batch_size=batch_size)
loss, accuracy = model_100.evaluate(np.array(X_test), np.array(y_test), verbose = 1)

#Ultima parte: Predizioni e valutazione del modello creato
y_pred = model_100.predict_classes(X_test, batch_size = 20, verbose = 1)
errori_commessi = 0
corretto = 0
count = 0
print()
for i in range(0, y_pred.size):
    #print(count, ')','Previsione: ', np.around(y_pred[i], decimals = 0), 'Obbiettivo:', y_test[i])
    count += 1
    if(np.around(y_pred[i], decimals = 0) != y_test[i]):
        errori_commessi+=1
    else:
        corretto+=1

#rounded_predictions = model.predict_classes(X_test, batch_size = 20, verbose = 1)
cm = confusion_matrix(y_test,y_pred)

print()
print()
print('Utilizzati come training set: ', np.array(y_train).size)
print('Utilizzati come test set: ', np.array(y_pred).size)
print()
print('Previsioni Corrette: ', corretto)
print('Errori Commessi: ', errori_commessi)
accuratezza = round((corretto/np.array(y_pred).size*100),2)
print('Accuratezza: ', accuratezza,'%')
print()
print(classification_report(y_test,y_pred))
print('Matrice di Confusione:')
print(cm)


print(history.history)


#_____________________________________________________________________________#
#_____________________________________________________________________________#
#______________________MODELLO CON 300 FEATURE PER PAROLA_____________________#
#_____________________________________________________________________________#
#_____________________________________________________________________________#



embedding_dim_modello_300 = 300
embeddings_index = dict()
f = open("dict/modello_300_feature.txt", encoding = 'utf-8')
for line in f:
    values = line.split() 
    #print(values)
    try:
        word = values[0] #in posizione 0 c'è la parola
        coefs = np.array(values[1:], dtype = "float32") #tutto il resto sono valori
        #Quindi adesso ho parola -> valore e lo aggiungo al dizionario
        embeddings_index[word] = coefs
    except ValueError:
        #print(values)
        continue
f.close()

#Creo la matrice dei pesi per adesso vuota
embeddings_matrix = np.zeros((vocab_size, embedding_dim_modello_300)) 

for word, index in tokenizer.word_index.items():
  if index > vocab_size-1:
    break
  else:
    embedding_vector = embeddings_index.get(word)
    #se il mio vettore non è nullo lo metto nella matrice
    if(embedding_vector) is not None:
      embeddings_matrix[index] = embedding_vector
      #per ogni parola del vocabolario prendo la prappresentazione del dizionario di Glove.

model_300 = Sequential()
model_300.add(Embedding(vocab_size, embedding_dim_modello_300, input_length=maxlen_X, weights = [embeddings_matrix], trainable = True))
model_300.add(Flatten())
model_300.add(Dense(neuroni_dense_1, activation="relu"))
model_300.add(Dense(neuroni_dense_2, activation="relu"))
model_300.add(Dense(1, activation="sigmoid"))

model_300.summary()
model_300.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model_300.fit(np.array(X_train), np.array(y_train), epochs=num_epoche,verbose=True, batch_size=batch_size)
loss, accuracy = model_300.evaluate(np.array(X_test), np.array(y_test), verbose = 1)

#Ultima parte: Predizioni e valutazione del modello creato
y_pred = model_300.predict_classes(X_test, batch_size = 20, verbose = 1)
errori_commessi = 0
corretto = 0
count = 0
print()
for i in range(0, y_pred.size):
    count += 1
    if(np.around(y_pred[i], decimals = 0) != y_test[i]):
        errori_commessi+=1
    else:
        corretto+=1

cm = confusion_matrix(y_test,y_pred)

print()
print()
print('Utilizzati come training set: ', np.array(y_train).size)
print('Utilizzati come test set: ', np.array(y_pred).size)
print()
print('Previsioni Corrette: ', corretto)
print('Errori Commessi: ', errori_commessi)
accuratezza = round((corretto/np.array(y_pred).size*100),2)
print('Accuratezza: ', accuratezza,'%')
print()
print(classification_report(y_test,y_pred))
print('Matrice di Confusione:')
print(cm)

print(history.history)
