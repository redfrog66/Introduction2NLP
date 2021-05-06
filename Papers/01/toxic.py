#SOURCE: https://stackabuse.com/python-for-nlp-multi-label-text-classification-with-keras/?fbclid=IwAR1bzkoGj99BBvbLZf7cIoAnhPDt_5WSrlTHVxiLnyPyU48t_x41N1TAxbM

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
import tensorflow as tf


import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt

#Let's now load the dataset into the memory
toxic_comments = pd.read_csv("toxic_comments.csv")


#The following script displays the shape of the dataset and it also prints the header of the dataset.
print(toxic_comments.shape)

toxic_comments.head()


#Let's remove all the records where any row contain a null value or empty string.
filter = toxic_comments["comment_text"] != ""
toxic_comments = toxic_comments[filter]
toxic_comments = toxic_comments.dropna()


#The comment_text column contains text comments. Let's print a random comment and then see the labels for the comments.
print(toxic_comments["comment_text"][168])


#This is clearly a toxic comment. Let's see the associated labels with this comment:
print("Toxic:" + str(toxic_comments["toxic"][168]))
print("Severe_toxic:" + str(toxic_comments["severe_toxic"][168]))
print("Obscene:" + str(toxic_comments["obscene"][168]))
print("Threat:" + str(toxic_comments["threat"][168]))
print("Insult:" + str(toxic_comments["insult"][168]))
print("Identity_hate:" + str(toxic_comments["identity_hate"][168]))


#Let's now plot the comment count for each label. To do so, we will first filter all the label or output columns.
toxic_comments_labels = toxic_comments[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
toxic_comments_labels.head()


#Using the toxic_comments_labels dataframe we will plot bar plots that show the total comment counts for different labels.
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

toxic_comments_labels.sum(axis=0).plot.bar()



#In this section, we will create multi-label text classification model with 
#single output layer. As always, the first step in the text classification 
#model is to create a function responsible for cleaning the text.
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


#Let's create our input and output set. The input is the comment from the comment_text column. 
#We will clean all the comments and will store them in the X variable. The labels or outputs 
#have already been stored in the toxic_comments_labels dataframe. We will use that dataframe 
#values to store output in the y variable. Look at the following script
X = []
sentences = list(toxic_comments["comment_text"])
for sen in sentences:
    X.append(preprocess_text(sen))

y = toxic_comments_labels.values


#We will divide our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


#We need to convert text inputs into embedded vectors.
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()


#We will be using GloVe word embeddings to convert text inputs to their numeric counterparts
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


#The following script creates the model. Our model will have one input layer, 
# one embedding layer, one LSTM layer with 128 neurons and one output layer with 6 
# neurons since we have 6 labels in the output
deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(6, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


#Let's print the model summary:
print(model.summary())


#The following script prints the architecture of our neural network:
tf.keras.utils.plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)


#Let's now train our model with 5 epochs
history = model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1, validation_split=0.2)


#Let's now evaluate our model on the test set:
score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

import matplotlib.pyplot as plt


#Finally, we will plot the loss and accuracy values for training and test sets to see if our model is overfitting
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()