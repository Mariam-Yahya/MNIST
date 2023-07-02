import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from keras import backend as K 
# from itertools import combinations


############################loading required data
#load data
def load_dataset():
 # load dataset
 (trainX, trainY), (testX, testY) = mnist.load_data()
 # reshape dataset to have a single channel
 trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
 testX = testX.reshape((testX.shape[0], 28, 28, 1))
 # one hot encode target values
 trainY = to_categorical(trainY)
 testY = to_categorical(testY)
 return trainX, trainY, testX, testY


def prep_pixels(train, test):
 # convert from integers to floats
 train_norm = train.astype('float32')
 test_norm = test.astype('float32')
 # normalize to range 0-1
 train_norm = train_norm / 255.0
 test_norm = test_norm / 255.0
 # return normalized images
 return train_norm, test_norm



##############################deifinition of the CNN model
class Model_CNN:
  def build(self):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model
     
     
     
############################Federated learning

###########definiton of clients
#creating clietns that would excute the FL locally
def create_clients(image_list, label_list, num_clients=100, initial='clients'):

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)
    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    
    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))}

#creating clients with Non IID data  
def non_iid_x(image_list, label_list, x=2, num_intraclass_clients=20):
        ''' creates x non_IID clients
        args: 
            image_list: python list of images or data points
            label_list: python list of labels
            x: none IID severity, 1 means each client will only have one class of data
            num_intraclass_client: number of sub-client to be created from each none IID class,
            e.g for x=1, we could create 10 further clients by splitting each class into 10
            
        return - dictionary 
            keys - clients's name, 
            value - client's non iid 1 data shard (as tuple list of images and labels) '''
        non_iid_x_clients = dict()
                
                #create unique label list and shuffle
        unique_labels = np.unique(np.array(range(10)))
        random.shuffle(unique_labels)
            #create sub label lists based on x
        sub_lab_list = [unique_labels[i:i + x] for i in range(0, len(unique_labels), x)]
                    
        for item in sub_lab_list :
            class_data = [(image, label) for (image, label) in zip(image_list, label_list) if list(label).index(1) in item]
                    
                    #decouple tuple list into seperate image and label lists
            images, labels = zip(*class_data)
                    
                    # create formated client initials
            initial = 'client_' + '_'
            for lab in item:
                initial = initial + str(lab) + '_' 
                    
                    #create num_intraclass_clients clients from the class 
            intraclass_clients = create_clients(list(images), list(labels), num_intraclass_clients, initial)
                    
                    #append intraclass clients to main clients'dict
            non_iid_x_clients.update(intraclass_clients)
                
        return non_iid_x_clients

#batch the clients data
def batch_data(data_shard, bs=10):
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)




#######definition of each weight corresponding to the client's local data
def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count

########scale model weights with the weight of the client
def scale_model_weights(weight, scalar):
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

#######sum of each scaled local model wieghts
def sum_scaled_weights(scaled_weight_list):
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss


###############
trainX, trainY, testX, testY = load_dataset()
trainX, testX = prep_pixels(trainX, testX)
clients = non_iid_x(trainX, trainY,2,20)
#clients = create_clients(trainX, trainY, num_clients=100, initial='client')
C=0.1 #fraction of the clients that we are going to use in the federated learning part
nb_clients=100
m = max(C * nb_clients, 1)
def clients_batch():
    clients_batched = dict()
    for client_key in random.sample(list(clients.keys()), int(m)):
        clients_batched[client_key] = batch_data(clients[client_key])
    return clients_batched




#process and batch the test set  
test_batched = tf.data.Dataset.from_tensor_slices((testX, testY)).batch(len(testY))

############Executing
lr = 0.01
comms_round = 1000
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr)   

#initialize global model
CNN_global = Model_CNN()
global_model = CNN_global.build()
global_acc_prog_CNN=list()
global_loss_prog_CNN=list()
#commence global training loop
for comm_round in range(comms_round):
        
    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list()
    clients_batched=dict()
    clients_batched=clients_batch()
    #randomize client data - using keys
    client_names= list(clients_batched.keys())
    random.shuffle(client_names)

    #loop through each client and create new local model
    for client in client_names:
        print("client {}, round = {}".format(client, comm_round))
        CNN_local = Model_CNN()
        local_model = CNN_local.build()
        local_model.compile(loss=loss, 
                    optimizer=optimizer, 
                    metrics=metrics)
    
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
    
        #fit local model with client's data
        local_model.fit(clients_batched[client], epochs=5, verbose=0)
    
        #scale the model weights and add to list
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
    
        #clear session to free memory after each communication round
        tf.keras.backend.clear_session()
    
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    #update global model 
    global_model.set_weights(average_weights)

    #test global model and print out metrics after each communications round
    for(X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
    global_acc_prog_CNN.append(global_acc)
    global_loss_prog_CNN.append(global_loss)



# global_acc_prog_CNN,global_loss_prog_CNN=run_global_model()
# plot accuracy and loss performance of the globel model
plt.subplot(2, 1, 1)
plt.title('accuracy performance throughout the rounds of the CNN model')
plt.plot(global_acc_prog_CNN, color='blue', label='acc')
plt.subplot(2, 1, 2)
plt.title('loss performance throughout the rounds of the CNN model')
plt.plot(global_loss_prog_CNN, color='blue', label='acc')
plt.show()


with open("global_acc_prog_CNN_Non_IID_"+str(int(m))+".txt", 'w') as f:
    for s in global_acc_prog_CNN:
        f.write(str(s) + '\n')

with open("global_loss_prog_CNN_Non_IID_"+str(int(m))+".txt", 'w') as f:
    for s in global_loss_prog_CNN:
        f.write(str(s) + '\n')








