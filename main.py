import nltk

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()


import tensorflow as tf
import numpy as np
import tflearn
import random
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.compat.v1.disable_eager_execution()



with open("intents.json") as json_data:
    intents = json.load(json_data)
# Empty lists for appending the data after processing NLP
words = []
documents = []
classes = []


ignore = ["?"]

# Starting a loop through each intent in intents["patterns"]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        # print(w)
        words.extend(w)
        # print(words)
        documents.append((w, intent["tag"]))
        # print(documents)

        if intent["tag"] not in classes:
            classes.append(intent["tag"])
            # print(classes)

words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))  #Removing Duplicates in words[]

#Removing Duplicate Classes
classes = sorted(list(set(classes)))

#Printing length of lists we formed
# print(len(documents),"Documents \n")
# print(len(classes),"Classes \n")
# print(len(words), "Stemmed Words ")

training = []
output = []


output_empty = [0] * len(classes)

#Creating Training set and bag of words for each sentence
for doc in documents:
    bag = []
    pattern_words = doc[0]
    # print(pattern_words)
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # print(pattern_words)
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training,dtype='object') #

#Creating Training Lists
train_x = list(training[:,0])
train_y = list(training[:,1])
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir="tflearn_logs")
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

import pickle

#Dumping training data by using dump() and wr

pickle.dump({"words": words, "classes": classes, "train_x": train_x, "train_y": train_y}, open("training_data", "wb"))
# Restoring all data structure
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
with open("intents.json") as json_data:
    intents = json.load(json_data)

model.load("./model.tflearn")


# Cleaning User Input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    return sentence_words


# Returning bag of words array: 0 or 1 or each word in the bag that exists in as we have declared in above lines
def bow(sentence, words, show_details=False):
    # Tokenizing the user input
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return (np.array(bag))

context = {}

ERROR_THRESHOLD = 0.25


def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if results:


        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    if 'context_set' in i:
                        if show_details: print('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print('tag:', i['tag'])

                        # A random response from the intent
                        return print(random.choice(i['responses']))


            results.pop(0)



