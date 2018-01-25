import numpy as np
import cv2
import matplotlib.pyplot as plt
import os



def dist(x, y, t = 'A') :  
    if t == 'A' :
        return np.sqrt(((x-y)**2 ).sum())
    elif t == 'B' :
    	return abs(x-y).sum() 

def knn (x_train , y_train , x , k = 5) :
    vals = []
    for ix in range(x_train.shape[0]) :
        v = [dist(x , x_train[ix , : ] , 'B') , y_train[ix] ]
        vals.append(v)  
    updated_vals = sorted (vals , key = lambda x : x[0])
    pred_arr = np.asarray(updated_vals[:k])
#     for numpy.__version__ >= 1.9
    pred_arr = np.unique(pred_arr[: , 1], return_counts = True)
    pred = pred_arr[1].argmax()
    return pred_arr[0][pred]

face = np.load('./data/my_face_data.npy')
label = np.load('./data/my_face_labels.npy')


def recognize_face(im):

    im = cv2.resize(im, (100, 100))

    # im = im.flatten()
    p = knn (face , label , im , k = 5)
    out = p 
    return out 

