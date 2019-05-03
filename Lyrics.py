from __future__ import absolute_import, division, print_function
from tensorflow import keras
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb
# dictionary of the words
word_index = imdb.get_word_index()
# before updates


#tensorflow code
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

        
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


#print(text_to_integer("this"))
#print(decode_review([1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65]))






# 
# Author: Jouya Mahmoudi
#


#
def lyric_extract(name):
    Lyriclist = []
    with open("billboard_lyrics_1964-2015.csv") as csvfile:
        file = csv.reader(csvfile, delimiter = ',')
        #skip first line
        next(csvfile)
        for row in file:
            if (name == "bad"):
                if (int(row[0]) >= 51):
                    Lyriclist.append(row[4])
            if (name == "good"):
                if (int(row[0]) <= 50):
                    Lyriclist.append(row[4])
        return Lyriclist
    
def text_to_integer(text):
    dict_size = get_dict_size(word_index)
    for word in word_index.keys():
        if word == text:
            num = word_index.get(word)
            return num;
        # add new words to the dictionary 
    word_index[text] = dict_size
    num = dict_size
    return num;    


# vectorize the list of lyrics
# takes a while :(
def convert_to_integer(list):
    processed_list = []
    for lyric in list:
        # if the line is not empty
        if (len(lyric) > 2):
            processed_lyrics = []
            # splits up lyrics into a list of words
            words = lyric.split()
            for word in words:
                # each word is processed individually 
                processed_lyrics.append(text_to_integer(word))
            processed_lyrics.append("STOP")
            processed_list.append(processed_lyrics)
    return processed_list

def get_dict_size(list):
    return len(list) - 1;

# for padding purposes
def average_length():
    total_length = 0
    counter = 0
    with open("billboard_lyrics_1964-2015.csv") as csvfile:
        file = csv.reader(csvfile, delimiter = ',')
        #skip first line
        next(csvfile)
        for row in file:
            total_length += len(row[4])
            counter += 1
    return int(total_length/counter)

# from the CSV file
def get_values(name):
    processed_list = []
    with open(name) as csvfile:
        file = csv.reader(csvfile,  delimiter = ',')
        for row in file:
            if (len(row) > 0):
                j = 0
                processed_lyrics = []
                while (row[j] != "STOP"):
                    processed_lyrics.append(row[j])
                    j += 1;
                processed_list.append(processed_lyrics)
    return processed_list

def main():
    bad_list = lyric_extract("bad")
    good_list = lyric_extract("good")
    #print(average_length())
    #pbad_list = convert_to_integer(bad_list)
    #print(len(pbad_list))
    #w = csv.writer(open("bad_list.csv", "w"))
    #for lyric in pbad_list:
        #print(counter)
        #w.writerow(lyric)
    #pgood_list = convert_to_integer(good_list)
    #w = csv.writer(open("good_list.csv", "w"))
    #for lyric in pgood_list:
        #w.writerow(lyric)
    Final_blist = get_values("bad_list.csv")
    Final_glist = get_values("good_list.csv")
    print(Final_blist[0])
    train_data = Final_blist + Final_glist
    
    
    # 1 is the label for good, 0 is the label for bad   
    train_labels = []
    for i in range(0,len(Final_blist)):
        train_labels.append(1)
    for i in range(0,len(Final_glist)):
        train_labels.append(0)
    
    test_list = []
    test_labels = []
    #with open("Test Lyrics.csv", encoding="utf8" ) as csvfile:
        #file = csv.reader(csvfile, delimiter = ',')
        #skip first line
        #next(csvfile)
        #for row in file:
            #test_list.append(row[1])
     
    

    #test_conversion = convert_to_integer(test_list)
    #w = csv.writer(open("Test_Data.csv", "w"))
    #for lyric in test_conversion:
        #w.writerow(lyric)    
    test_data = get_values("Test_Data.csv")
    print(len(test_data))
    # 1 is the label for good, 0 is the label for bad       
    for i in range(0,299):
        counter = 0
        switch = False     
        if (counter == 50):
            switch = True
            counter = 0
        
        if (counter == 50 and switch == True):
            switch = False
            counter = 0
            
        if (switch == False):
            test_labels.append(1)
            counter += 1
        
        if (switch == True):
            test_labels.append(0)
            counter += 1
            
    print(len(test_labels))
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=1510)    
    
    
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=1510)      
    
    vocab_size = 120000
    
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    
    #model.summary()    
    
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])    
    
    
    x_val = train_data[:1000]
    partial_x_train = train_data[1000:]
    
    y_val = train_labels[:1000]
    partial_y_train = train_labels[1000:]   
    
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=250,
                        batch_size=150,
                        validation_data=(x_val, y_val),
                        verbose=1)      
    
    print(len(test_data))
    print(len(test_labels))
    results = model.evaluate(test_data, test_labels)
    
    print(results) 
    
    history_dict = history.history
    history_dict.keys()
    dict_keys = (['loss', 'acc', 'val_loss', 'val_acc'])
    
    
    
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()  
    
    plt.clf()   # clear figure
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()    

main()
