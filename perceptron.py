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

def print_matrix(matrix, eta):
    print('Confusion Matrix for', eta, 'Step Experiment')
    for x in range(-1, len(matrix[0])):
        if x == -1:
            print('-',end='\t\t')
        elif x == 9:
            print(x)
        else:
            print(x, end='\t\t')
    for i in range(0, len(matrix[0])):
        print(i, end='\t\t')
        for j in range(0, len(matrix[0])):
            print(matrix[i][j], end='\t\t')
        print()


train_file_img = r'C:\Users\Andrew\Desktop\perceptron\train-images.idx3-ubyte'
train_file_lab = r'C:\Users\Andrew\Desktop\perceptron\train-labels.idx1-ubyte'
test_file_img = r'C:\Users\Andrew\Desktop\perceptron\t10k-images.idx3-ubyte'
test_file_lab = r'C:\Users\Andrew\Desktop\perceptron\t10k-labels.idx1-ubyte'

train_img = idx2numpy.convert_from_file(train_file_img)
train_lab = idx2numpy.convert_from_file(train_file_lab)
test_img = idx2numpy.convert_from_file(test_file_img)
test_lab = idx2numpy.convert_from_file(test_file_lab)

etaGlobal = 0.1

epoch = 0
maxEpoch = 20
trainingsize = 5000#len(train_img)
testsize = 5000#len(test_img)
train_accuracy_arr = np.zeros(maxEpoch)
test_accuracy_arr = np.zeros(maxEpoch)
epochIterArr = np.zeros(maxEpoch)
difference = 0
lastRun = False

confusionMatrix = np.zeros((10, 10), dtype=np.int)
#weights initialization
image_size_vect = train_img[0].flatten()
weights = np.zeros((10, len(image_size_vect)+1), dtype=np.float64)
for p in range(0, 10):
    for r in range(0, len(image_size_vect)):
        weights[p][r] = random.uniform(-0.05, 0.05)

while(epoch < maxEpoch and lastRun == False):
    if(epoch > 1):
        difference = (train_accuracy_arr[epoch-1] - train_accuracy_arr[epoch-2])
        if(maxEpoch-1 == epoch or difference <= 0.01):
            lastRun = True
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
            if epoch != 0:
                weights = weight_update(weights, etaGlobal, train_lab[img_num], neuron_sum_vect, train_img_vect)
        else:
            correctAmt += 1
        maxInd = np.argmax(neuron_sum_vect)
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
        if lastRun:
            confusionMatrix[maxInd][test_lab[img_num]] += 1

    test_accuracy_arr[epoch] = (correctAmt / testsize) * 100

    print('Epoch:', epoch, 'Finished')
    epochIterArr[epoch] = epoch
    print(train_accuracy_arr[epoch-1], train_accuracy_arr[epoch-2])
    print(difference)

    if lastRun:
        train_accuracy_arr = np.delete(train_accuracy_arr, slice(epoch, maxEpoch))
        test_accuracy_arr = np.delete(test_accuracy_arr, slice(epoch, maxEpoch))

    epoch += 1

print(train_accuracy_arr)
print_matrix(confusionMatrix, etaGlobal)

# print(epochIterArr)
# print(train_accuracy_arr)
# print(test_accuracy_arr)

plt.plot(train_accuracy_arr, label='Training Set')
plt.legend()
plt.plot(test_accuracy_arr, label='Test Set')
plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.title('Learning Rate: '+str(etaGlobal))
plt.savefig('graph.png')
plt.show()
