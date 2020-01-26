import random
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import cv2

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

train_file_img = r'C:\Users\Andrew\Desktop\perceptron\train-images.idx3-ubyte'
train_file_lab = r'C:\Users\Andrew\Desktop\perceptron\train-labels.idx1-ubyte'
test_file_img = r'C:\Users\Andrew\Desktop\perceptron\t10k-images.idx3-ubyte'
test_file_lab = r'C:\Users\Andrew\Desktop\perceptron\t10k-labels.idx1-ubyte'

train_img = idx2numpy.convert_from_file(train_file_img)
train_lab = idx2numpy.convert_from_file(train_file_lab)
test_img = idx2numpy.convert_from_file(test_file_img)
test_lab = idx2numpy.convert_from_file(test_file_lab)



epoch = 0
correct = 0
trainingsize = len(train_img)

while(epoch <= 30):
    for img_num in range(0, 1000): #len(train_img)):
        train_img_vect = train_img[img_num].flatten()
        bias = 1
        train_img_vect = np.insert(train_img_vect, 0, bias)
        train_img_vect = np.true_divide(train_img_vect, 255)

        weights = np.zeros((10, len(train_img_vect)), dtype=np.float64)
        for p in range(0, 10):
            for r in range(0, len(train_img_vect)):
                weights[p][r] = random.uniform(-0.05, 0.05)

        neuron_sum_vect = np.dot(weights, train_img_vect)

        same_check = target_same_check(neuron_sum_vect, train_lab[img_num])
        print(img_num, end=' ')
        if not same_check:
            print('-')
            weights = weight_update(weights, 0.1, train_lab[img_num], neuron_sum_vect, train_img_vect)
        else:
            print('CORRECT')
            correct += 1
    epoch += 1





# height, width = train_img[0].shape
# for i in range (0,height):
#     for j in range(0,width):
#         print(train_img[0][i][j], end='\t')
#     print()
#
# print(flat_train_img)
# cv2.imshow("image", train_img[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()




