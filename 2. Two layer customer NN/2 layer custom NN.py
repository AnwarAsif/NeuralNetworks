import vugrad as vg 
import numpy as np 
import matplotlib.pyplot as plt 

# Load Data 
(xtrain, ytrain), (xval, yval), num_classes = vg.load_synth()

print(f'## loaded data:')
print(f'number of instances: {xtrain.shape[0]} in training, {xval.shape[0]} in validation')
print(f'training class distribution: {np.bincount(ytrain)}')
print(f'val. class distribution: {np.bincount(yval)}')

num_instances, num_features = xtrain.shape

num_instances, num_features = xtrain.shape

# Hyper parameter setup 
n, m = xtrain.shape
b = 128 
epochs = 20 
lr = 0.001

mlp = vg.MLP(input_size=num_features, output_size=num_classes)

print('train vanilla network')
loss1 =[]
acc1 = []
for epoch in range(epochs):

    if epoch % 1 == 0:
        o = mlp(vg.TensorNode(xval))
        oval = o.value
        predictions = np.argmax(oval, axis=1)
        num_correct = (predictions == yval).sum()
        acc = num_correct / yval.shape[0]
        o.clear() # gc the computation graph
        # print(f'       accuracy: {acc:.4}')
    cl = 0.0 # running sum of the training loss

    for fr in range(0, n, b):
        to = min(fr + b, n)
        batch, targets = xtrain[fr:to, :], ytrain[fr:to]
        batch = vg.TensorNode(value=batch)
        outputs = mlp(batch)
        loss = vg.celoss(outputs, targets)

        cl += loss.value
        loss.backward()

        for parm in mlp.parameters():
            parm.value -= lr * parm.grad

        loss.zero_grad()
        loss.clear()

    # print(f'   running loss: {cl:.4}')
    # print(f'Loss: {cl:.4}, Acc: {acc:.4}')
    loss1.append(cl)
    acc1.append(acc)


mlp = vg.MLP2(input_size=num_features, output_size=num_classes)

print('train network with Relu')
loss2 =[]
acc2 = []
for epoch in range(epochs):

    if epoch % 1 == 0:
        o = mlp(vg.TensorNode(xval))
        oval = o.value
        predictions = np.argmax(oval, axis=1)
        num_correct = (predictions == yval).sum()
        acc = num_correct / yval.shape[0]
        o.clear() # gc the computation graph
        # print(f'       accuracy: {acc:.4}')
    cl = 0.0 # running sum of the training loss

    for fr in range(0, n, b):
        to = min(fr + b, n)
        batch, targets = xtrain[fr:to, :], ytrain[fr:to]
        batch = vg.TensorNode(value=batch)
        outputs = mlp(batch)
        loss = vg.celoss(outputs, targets)

        cl += loss.value
        loss.backward()

        for parm in mlp.parameters():
            parm.value -= lr * parm.grad

        loss.zero_grad()
        loss.clear()

    # print(f'   running loss: {cl:.4}')
    # print(f'Loss: {cl:.4}, Acc: {acc:.4}')
    loss2.append(cl)
    acc2.append(acc)

print('3 Layer network with Relu')

mlp = vg.MLP3L(input_size=num_features, output_size=num_classes)
loss3 =[]
acc3 = []
for epoch in range(epochs):

    if epoch % 1 == 0:
        o = mlp(vg.TensorNode(xval))
        oval = o.value
        predictions = np.argmax(oval, axis=1)
        num_correct = (predictions == yval).sum()
        acc = num_correct / yval.shape[0]
        o.clear() # gc the computation graph
        # print(f'       accuracy: {acc:.4}')
    cl = 0.0 # running sum of the training loss

    for fr in range(0, n, b):
        to = min(fr + b, n)
        batch, targets = xtrain[fr:to, :], ytrain[fr:to]
        batch = vg.TensorNode(value=batch)
        outputs = mlp(batch)
        loss = vg.celoss(outputs, targets)

        cl += loss.value
        loss.backward()

        for parm in mlp.parameters():
            parm.value -= lr * parm.grad

        loss.zero_grad()
        loss.clear()

    # print(f'   running loss: {cl:.4}')
    # print(f'Loss: {cl:.4}, Acc: {acc:.4}')
    loss3.append(cl)
    acc3.append(acc)


mlp = vg.MLP2(input_size=num_features, output_size=num_classes)

print('train network with Momentum')
loss4 =[]
acc4 = []
momentum = .1
for epoch in range(epochs):

    if epoch % 1 == 0:
        o = mlp(vg.TensorNode(xval))
        oval = o.value
        predictions = np.argmax(oval, axis=1)
        num_correct = (predictions == yval).sum()
        acc = num_correct / yval.shape[0]
        o.clear() # gc the computation graph
        # print(f'       accuracy: {acc:.4}')
    cl = 0.0 # running sum of the training loss

    for fr in range(0, n, b):
        to = min(fr + b, n)
        batch, targets = xtrain[fr:to, :], ytrain[fr:to]
        batch = vg.TensorNode(value=batch)
        outputs = mlp(batch)
        loss = vg.celoss(outputs, targets)

        cl += loss.value
        loss.backward()

        # apply learning rate momentum 
        

        for parm in mlp.parameters():
            parm.value -= lr * parm.grad
        
        loss.zero_grad()
        loss.clear()
    lr = (lr*epoch)+lr
    loss4.append(cl)
    acc4.append(acc)

# Plot Loss 
plt.title("Training Loss Coperission")
# plt.plot(loss1, color='b',label='2L Sig')
plt.plot(loss2, color='r', label='2L Relu')
# plt.plot(loss3,color='g',label='3L Relu')
plt.plot(loss4,color='y', label='relu + Momentum')
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.title("Validation Accuracy Coperission") 
# plt.plot(acc1, color='b',label='2L Sig')
plt.plot(acc2, color='r', label='2L Relu')
# plt.plot(acc3,color='g',label='3L Relu')
plt.plot(acc4,color='y', label='relu + Momentum')
plt.legend()
plt.grid(True)
plt.show()



