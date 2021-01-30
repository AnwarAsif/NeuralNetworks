#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math 
import random 
import numpy as np 
import matplotlib.pyplot as plt 

# Activation Functions 
def sigmoid(inputs):
    return [(i/(i + math.exp(-i))) for i in inputs]
def softmax(inputs):
    exp_val = [math.exp(i) for i in inputs]
    return [exp/sum(exp_val) for exp in exp_val]
# Loss function 
def log_loss(y_hat, y_target):
    y_hat = np.clip(y_hat, 1e-7, 1 - 1e-7)
    return sum([- np.log(y_hat[i]) * y_target[i] for i in range(len(y_hat))])
# Gradient funcrtions for loss and activation 
def gradient_loss(y_hat, y_target):
    return [y - yt for y, yt in zip(y_hat,y_target) ]
def gradient_sigmoid(layer, grads):
    return [g*exp*(1-exp) for g, exp in zip(grads, layer.activation_output)]
# Optimizer 
def SGD(layer, lr):
    updated_weights = [ [w - lr * dw for w, dw in zip(ws, dws)] for ws, dws in  zip(layer.weights, layer.dw)]
    layer.weights = updated_weights
    layer.biases = [(b - lr*db) for b,db in zip(layer.biases, layer.db)]
# Data load and processing 
def load_synth(num_train, num_val):
    THRESHOLD = 0.6
    quad = np.asarray([[1, 0.5], [1, .2]])
    ntotal = num_train + num_val
    x = np.random.randn(ntotal, 2)
    q = np.einsum('bf, fk, bk -> b', x, quad, x)
    y = (q > THRESHOLD).astype(np.int)
    return (x[:num_train, :], y[:num_train]), (x[num_train:, :], y[num_train:]), 2
def convertToOneHot(vector, num_classes=None):
    
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)
def plot_graph(x, y, z):
    plt.title('Training vs Validation Loss')
    plt.plot(x,y, linewidth=2, color='g', label='train')
    plt.legend("Training loss")
    plt.plot(x,z, linewidth=2, color='r', label='val')
    plt.legend()
    plt.grid()
    plt.show()
# layer setup 
class layer():
    def __init__ (self, input_size, neuron_no):
        self.neuron_no = neuron_no
        self.biases = [0. for i in range(neuron_no)]
        self.weights = [[.1*random.randint(0,9) for i in range(neuron_no)] for i in range(input_size)]
    def forward_pass(self, inputs, activation = 'sigmoid'):
        self.inputs = inputs 
        self.outputs = [sum([w[n]*x for w,x in zip(self.weights, self.inputs)]) + self.biases[n] for n in range(self.neuron_no)]
        if activation == 'sigmoid': self.activation_output = sigmoid(self.outputs) 
        elif activation == 'softmax': self.activation_output = softmax(self.outputs)
    def backward_pass(self, grads):
        self.dw = [[i*g for g in grads] for i in self.inputs]
        self.db = grads
        self.grads = [sum([w[i] * grads[i] for i in range(len(w))]) for w in self.weights]

if __name__ == '__main__':

    # Data import and preprocessing 
    train, test, _= load_synth(60000, 10000)
    # Visualize the 
    x_train, y_train = train[0], train[1]
    plt.scatter(x_train[:,0],x_train[:,1], c=y_train, cmap='brg')
    plt.show()
    # train, test, _= load_synth(100, 200)
    inputs, outputs = train[0], train[1]
    inputs_val, outputs_val = test[0], test[1]
    outputs = convertToOneHot(outputs)
    outputs_val = convertToOneHot(outputs_val)
    
    # Network setup 
    layer1 = layer(2,3)
    layer2 = layer(3,2)

    # Hyper parameter setup 
    epochs = 5
    lr = 0.00001
    
    epochs_loss = []
    epochs_no = []
    epochs_val = []

    for epoch in range(epochs):
        epoch_loss = []
        val_loss = []
        for x, y in zip(inputs, outputs):

            layer1.forward_pass(x)
            layer2.forward_pass(layer1.activation_output)
            loss = log_loss(layer2.activation_output, y) 

            #  if not epoch % 100:
            #      print('epoch:', epoch,'loss:', loss)
            epoch_loss.append(loss)
            
            dL = gradient_loss(layer2.activation_output, y)
            layer2.backward_pass(dL)
            d_sig = gradient_sigmoid(layer1, layer2.grads)
            layer1.backward_pass(d_sig)

            SGD(layer1, lr)
            SGD(layer2, lr)
            
        for x, y in zip(inputs_val, outputs_val):
    
            layer1.forward_pass(x)
            layer2.forward_pass(layer1.activation_output)
            loss = log_loss(layer2.activation_output, y)

            val_loss.append(loss)


        epochs_loss.append(sum(epoch_loss)/len(epoch_loss))
        epochs_val.append(sum(val_loss)/len(val_loss))
        epochs_no.append(epoch)

    # plot training and validation loss 
    plot_graph(epochs_no, epochs_loss, epochs_val)


