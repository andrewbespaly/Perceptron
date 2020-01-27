import random
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

def onezerovect(listinput):
    newList = np.zeros(len(listinput))
    for i in range(0, len(listinput)):
        if listinput[i] > 0:
            newList[i] = 1
        else:
            newList[i] = 0
    return newList

def targetArr(listinput):
    newList = np.zeros(10)
    newList[listinput] = 1
    return newList

def target_same_check(y_vector, t):
    classifier = np.argmax(y_vector)
    if classifier == t:
        return True
    else:
        return False

def weight_update(weight_vect, eta, target, y_vect, input_img_vect):
    tarr = targetArr(target)
    yarr = onezerovect(y_vect)
    tsuby = np.subtract(tarr, yarr)
    etastep = np.dot(eta, tsuby)
    for p in range(0, len(etastep)):
        finalstep = np.dot(etastep[p], input_img_vect)
        weight_vect[p] += finalstep
    return weight_vect

train_file_img = r'C:\Users\andyg\Desktop\Perceptron\train-images.idx3-ubyte'
train_file_lab = r'C:\Users\andyg\Desktop\Perceptron\train-labels.idx1-ubyte'
test_file_img = r'C:\Users\andyg\Desktop\Perceptron\t10k-images.idx3-ubyte'
test_file_lab = r'C:\Users\andyg\Desktop\Perceptron\t10k-labels.idx1-ubyte'

train_img = idx2numpy.convert_from_file(train_file_img)
train_lab = idx2numpy.convert_from_file(train_file_lab)
test_img = idx2numpy.convert_from_file(test_file_img)
test_lab = idx2numpy.convert_from_file(test_file_lab)



epoch = 0
maxEpoch = 3
trainingsize = 500#len(train_img)
testsize = 500#len(test_img)
train_accuracy_arr = np.zeros(maxEpoch)
test_accuracy_arr = np.zeros(maxEpoch)
epochIterArr = np.zeros(maxEpoch)

confusionMatrix = np.zeros((maxEpoch, 10, 10), dtype=np.int)
print(confusionMatrix)
#weights initialization
image_size_vect = train_img[0].flatten()
weights = np.zeros((10, len(image_size_vect)+1), dtype=np.float64)
for p in range(0, 10):
    for r in range(0, len(image_size_vect)):
        weights[p][r] = random.uniform(-0.05, 0.05)

while(epoch < maxEpoch):
    #training data learning
    correctAmt = 0
    for img_num in range(0, trainingsize):
        train_img_vect = train_img[img_num].flatten()
        bias = 1
        train_img_vect = np.insert(train_img_vect, 0, bias)
        train_img_vect = np.true_divide(train_img_vect, 255)

        neuron_sum_vect = np.dot(weights, train_img_vect)

        same_check = target_same_check(neuron_sum_vect, train_lab[img_num])
        if not same_check:
            weights = weight_update(weights, 0.1, train_lab[img_num], neuron_sum_vect, train_img_vect)
        else:
            correctAmt += 1
        maxInd = np.argmax(neuron_sum_vect)
        confusionMatrix[epoch][train_lab[img_num]][maxInd] += 1
    train_accuracy_arr[epoch] = (correctAmt / trainingsize) * 100

    #test data learning
    correctAmt = 0
    for img_num in range(0, testsize):
        test_img_vect = test_img[img_num].flatten()
        bias = 1
        test_img_vect = np.insert(test_img_vect, 0, bias)
        test_img_vect = np.true_divide(test_img_vect, 255)

        neuron_sum_vect = np.dot(weights, test_img_vect)

        same_check = target_same_check(neuron_sum_vect, test_lab[img_num])
        if same_check:
            correctAmt += 1
        maxInd = np.argmax(neuron_sum_vect)
        confusionMatrix[epoch][test_lab[img_num]][maxInd] += 1

    test_accuracy_arr[epoch] = (correctAmt / testsize) * 100

    print('Epoch:', epoch, 'Finished')
    epochIterArr[epoch] = epoch

    epoch += 1


print(confusionMatrix)
print()
print(epochIterArr)
print(train_accuracy_arr)
print(test_accuracy_arr)

plt.plot(train_accuracy_arr,label='Training Set')
plt.legend()
plt.plot(test_accuracy_arr, label='Test Set')
plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.savefig('graph.png')
plt.show()








