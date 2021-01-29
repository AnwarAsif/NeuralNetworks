#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math 
import random 

# Activation Functions 
def sigmoid(inputs):
    return [(i/(i + math.exp(-i))) for i in inputs]
def softmax(inputs):
    exp_val = [math.exp(i) for i in inputs]
    return [exp/sum(exp_val) for exp in exp_val]
def log_loss(y_hat, y_target):
    return sum([-math.log(y_hat[i]) * y_target[i] for i in range(len(y_hat))])
def gradient_loss(y_hat, y_target):
    return [y - yt for y, yt in zip(y_hat,y_target) ]
def gradient_sigmoid(layer, grads):
    return [g*exp*(1-exp) for g, exp in zip(grads, layer.activation_output)]

# layer setup 
class layer():
    def __init__ (self, input_size, neuron_no):
        self.neuron_no = neuron_no
        self.biases = [0. for i in range(neuron_no)]
        self.weights = [[.01*random.randint(0,9) for i in range(neuron_no)] for i in range(input_size)]
    def forward_pass(self, inputs, activation = 'sigmoid'):
        self.inputs = inputs 
        self.outputs = [sum([w[n]*x for w,x in zip(self.weights, self.inputs)]) + self.biases[n] for n in range(self.neuron_no)]
        if activation == 'sigmoid': self.activation_output = sigmoid(self.outputs) 
        elif activation == 'softmax': self.activation_output = softmax(self.outputs)
    def backword_pass(self, grads):
        self.dw = [[i*g for g in grads] for i in self.inputs]
        self.db = grads
        self.grads = [sum([w[i] * grads[i] for i in range(len(w))]) for w in self.weights]
if __name__ == '__main__':

    # Input and target 
    inputs = [1,-1]
    outputs = [1,0]
    
    # Network setup and update the weights 
    layer1 = layer(2,3)
    layer1.weights =[[1.,1., 1.], [-1., -1, -1.]]
    layer2 = layer(3,2)
    layer2.weights = [[1., 1.],[-1.,-1.],[-1., -1.]]


    print("Layer 1 biases", layer1.biases)
    print("Layer 1 weights", layer1.weights)    
    print("Layer 2 biases", layer2.biases)
    print("Layer 2 weights", layer2.weights)    
    
    # Forward pass 
    layer1.forward_pass(inputs)
    print('Layer 1 output',layer1.outputs)
    print('layer 1 activation output', layer1.activation_output)
    
    layer2.forward_pass(layer1.activation_output, activation='softmax') 
    print("layer 2 output:",layer2.outputs)
    print('layer 2 activation:',layer2.activation_output)

    # Calculate Loss 
    loss = log_loss(layer2.activation_output, outputs)
    print('total loss', loss)

    # Back tracking 
    dL = gradient_loss(layer2.activation_output, outputs)
    print('Gradients of cross entropy loss', dL)
    layer2.backword_pass(dL)
    print('Layer 2 dW:', layer2.dw)
    print('Layer 2 db:', layer2.db)
    print('Layer 2 Gradients', layer2.grads)
    d_sig = gradient_sigmoid(layer1,layer2.grads)
    print('derivative of Sigmoid:', d_sig)
    layer1.backword_pass(d_sig)
    print("Layer 1 dW:", layer1.dw)
    print("layer1 db:", layer1.db)
    print("Layer 1 grads:", layer1.grads)
    
