# NeuralNetworks
My source code for different types of Neural Networks using **Numpy**, **PyTorch** and **TensorFlow**

1. Building Neural Network from scratch 
    1.1 Math behind NN (forward pass, backward propagation, log loss, cross entropy loss, derivatives of sigmoid, softmax, layers)
    1.2 NN without numpy and train over synthetic data and Minist Data set
    1.3 NN with numpy train over Minist dataset using different optimizers 

2. Customer Two layers network
    2.1 Customer Two layers network build to train over CIFAR10 data set. 
    2.2 Hyperparameter tunning and select model 

3. Convolutional neural network (CNN)
    3.1 CNN using Adam optimizer 
4. recurrent neural network (RNN) and Long short-term memory (LSTM) Neural Network 
    4.1 **Custom Dataloader:** Loading IMDB data set by building custom dataloader with `variable batch size`[extra padding ] and `mini batch`[reduce memory] to train the model 
    4.2 **Non-recurrent model:** The  non  recurrent  network  consists  of  one  embedding  layer  and  two  linear  layes and Relu activation function is used.

    4.3 **Elman RNN:** A new class Elman was create to implement this recurrent network.  An instance of the class was introduced as an layer of the MLP network.  An object ’hidden’ was created to reserve short time weights of the layer which used for training the elman network.
    
    4.4 **pyTorch RNNs**:The  Network  was  on  top  of  the  MLP  network,  where  between  linear  layers  pytorch  recurrent layers were introduced.

5. Generative model 
    5.1 GAN to reproduce MNIST and images data set
    5.2 VAE on MNIST and image dataset 

6. Transfer Learning 
    6.1 InceptionV3 - Transfer Knowledge 
7. Self Attention and Transformers
    7.1 Simple Transformer with customer self attention layer 
    7.2 Movie recommendation using Transformer
8. GPT3
9. Tools and techniques 
    9.1 Custom Dataloader pytorch 
