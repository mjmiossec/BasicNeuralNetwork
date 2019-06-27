#Adventures in learning about artificial neural network...
#Originally for UNAB course => spanglish variable names.
#The data and the methods to convert it and display it at the end come from:
#https://www.python-course.eu/neural_network_mnist.php
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

fac = 0.99/255

def sigmoide(z):
    return 1/(1+np.exp(-z))

def sigmoder(x):
    return x*(1-x)

def load_data(filename,need_Y,n_set,iterations=0):
    datos = np.loadtxt(filename, delimiter=",",skiprows= iterations*n_set,max_rows=n_set)
    conjunto = np.insert(np.asfarray(datos[:, 1:])*fac+0.01,0,1,axis=1)
    digit = np.asfarray(datos[:, :1])
    
    if need_Y:
        Y =(np.arange(10)==digit).astype(np.float)
        Y[Y==0] = 0.01
        Y[Y==1] = 0.99
        return conjunto, digit, Y
    else:
        return conjunto, digit

def clasification(data,thetas_uno,thetas_dos,thetas,training): #used both for training and test.
    activ_dos = np.insert(sigmoide(np.dot(data,thetas_uno)),0,1,axis=1) #hidden layer activation + column of 1s for bias unit
    activ_tres = np.insert(sigmoide(np.dot(activ_dos,thetas_dos)),0,1,axis=1) ##new hidden layer
    result = sigmoide(np.dot(activ_tres,thetas))
        
    if training:
        return result, activ_tres, activ_dos
    else:
        return np.argmax(result,axis=1)

def train(conjunto,thetas_uno,thetas_dos,thetas,Y):

    for i in range(100):
        ##if(i%100 == 0):
        ##    print("forwad propagation:",i)
        result, activ_tres, activ_dos = clasification(conjunto,thetas_uno,thetas_dos,thetas,True) #forward propagation
        
        #start of backpropagation
        error_salida = result-Y
        error_tres = np.dot(error_salida,thetas[1:].T)*sigmoder(np.delete(activ_tres,0,axis=1)) #calc. error, removing bias unit.
        error_dos = np.dot(error_tres,thetas_dos[1:].T)*sigmoder(np.delete(activ_dos,0,axis=1))

        thetas -= 0.005*np.dot(activ_tres.T,error_salida)
        thetas_dos -= 0.005*np.dot(activ_dos.T,error_tres) 
        thetas_uno -= 0.005*np.dot(conjunto.T,error_dos) #as initial values never change, use onetime_conjunto.
        #end of backpropagation

        if(not np.count_nonzero(digit.T - np.argmax(result,axis=1))):
            print(i,"iterations necessary")
            break   
    #print(digit.T - np.argmax(result,axis=1))
    return thetas_uno, thetas_dos, thetas

thetas_uno = 2*np.random.random((785,784))-1
thetas_dos = 2*np.random.random((785,784))-1 ##for new hidden layer
thetas = 2*np.random.random((785,10))-1

for i in range(600):
    print("training with dataset subset (100) n°",i+1)
    conjunto, digit, Y = load_data("mnist_train.csv",True,100,i) #data from training set
    thetas_uno, thetas_dos, thetas = train(conjunto,thetas_uno,thetas_dos,thetas,Y)
    
##save weights.
archiv_theta = open("thetas.dat","wb")
pickle.dump((thetas_uno, thetas_dos, thetas),archiv_theta,True)
archiv_theta.close()

##get the weights as previously saved.
archiv_theta = open("thetas.dat","rb")
thetas_uno,thetas_dos,thetas = pickle.load(archiv_theta)

##for each subset of 1000 in the test data...
for i in range(10):
    test, digitest = load_data("mnist_test.csv",False,1000,i) ##data from test set.
    results = clasification(test,thetas_uno,thetas_dos,thetas,False)
    ##for i in range(1000):
    ##    print("number is",digitest.T[0][i],"ANN guessed",results[i])
    print(np.round((np.sum(digitest.T == results)/np.size(digitest.T))*100,decimals=2),"% of digits classified correctly")
    frst_error_inset = (digitest.T == results).tolist()[0].index(False)
    print(frst_error_inset)
    print("example of misclassified in this set. Classified as",results[frst_error_inset],"but actually a",digitest.T[0][frst_error_inset])
    img = test[frst_error_inset][1:].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
