import codecs
import pickle
import re
import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from datacleaner import DataCleaner

MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2


def get_stop_words():
    path = "data/stop_words.txt"
    stop_words = []
    with codecs.open(path, "r", encoding="utf-8", errors="ignore") as myfile:
        stop_words = myfile.readlines()
    stop_words = [word.strip() for word in stop_words]
    return stop_words


def get_text_sequences(texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=100)
    return data, word_index


# Clean/Normalize Arabic Text

def clean_str(text):
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim
    text = text.strip()

    return text


df = pd.read_csv("data/final.csv")
## Clean and drop stop words
df['text'] = df.text.apply(lambda x: clean_str(x))
stop_words = r'\b(?:{})\b'.format('|'.join(get_stop_words()))
df['text'] = df['text'].str.replace(stop_words, '')
df['text'] = df.text.apply(lambda x: DataCleaner.stemming(x))
df['binary_sentiment'] = df.sentiment.map(dict(positive=1, negative=0))
df = shuffle(df)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['binary_sentiment'], test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
data = pad_sequences(sequences, maxlen=100)
test_data = pad_sequences(test_sequences, maxlen=100)
kernel_size=3
# Model defnition
model_lstm = Sequential()
model_lstm.add(Embedding(20000, 100, input_length=100))
model_lstm.add(Dropout(0.4))
model_lstm.add(Conv1D(600, kernel_size, padding='valid', activation='relu', strides=1))
model_lstm.add(Conv1D(300, kernel_size, padding='valid', activation='relu', strides=1))
model_lstm.add(Conv1D(150, kernel_size, padding='valid', activation='relu', strides=1))
model_lstm.add(Conv1D(75, kernel_size, padding='valid', activation='relu', strides=1))
model_lstm.add(Flatten())
model_lstm.add(Dense(600))
model_lstm.add(Dropout(0.5))
model_lstm.add(Activation('relu'))
model_lstm.add(Dense(1))
model_lstm.add(Activation('sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model_lstm.fit(data, y_train, validation_split=0.4, epochs=10)


def plot_history(model):
    acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)
with open("models/cnn.pickle", "wb") as f:
    pickle.dump(tokenizer, f)
model_lstm.save('models/cnn.hdf5')

"""model_json = model_lstm.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)

model_lstm.save_weights("models/model.h5")
print("Saved model to disk")"""

from keras.models import model_from_json
import json

with open('models/model.json','r') as f:
    model_json = json.load(f)
model_json=json.dumps(model_json)
model = model_from_json(model_json)
model.load_weights('models/model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
plot_history(model)
