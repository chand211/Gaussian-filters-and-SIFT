#Implementing K-means

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import random
start_time = datetime.now()

#img1 = cv2.imread('/Users/chand/Desktop/CompVision/Programming/Programming1/Kmean_img1.jpg')
img1 = cv2.imread('/Users/chand/Desktop/CompVision/Programming/Programming1/Kmean_img2.jpg')

x = []
for i in range(0,672):
    for j in range(0,1200):
        x.append(img1[i][j][0])
y = []
for i in range(0,672):
    for j in range(0,1200):
        y.append(img1[i][j][1])
z = []
for i in range(0,672):
    for j in range(0,1200):
        z.append(img1[i][j][2])

df = pd.DataFrame(list(zip(x,y,z)),
               columns =['x', 'y','z'])
print("df made")


#Number of clusters
k = 10
#seed cluster starts
centroids = []
for i in range(0,k):
    num = random.randint(0,len(df))
    x = [ df['x'][num] , df['y'][num], df['z'][num] ]
    centroids.append(x)
    print("Made centroid ", i)
centroids = np.array(centroids)
#Vec to represent the classifications
classes = np.zeros(len(df))

#This function makes plots
def plotter(n):
    df['classes'] = classes.tolist()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = {0.0: 'green', 1.0: 'red', 2.0: 'blue', 3.0: 'black', 4.0: 'purple', 5.0: 'violet', 6.0: 'orange',
              7.0: 'pink', 8.0: 'brown', 9.0: 'cyan', 10.0: 'magenta'}
    ax.scatter(df['x'], df['y'], df['z'], s=1, c=df['classes'].map(colors))
    plt.title('Clusters= '+str(k)+', Iteration = '+str(n))

#This function performs the E step of EM, classification
def classify(ct,r):
    for i in range(0,len(df)):
        norms = []
        for j in range(0,k):
            x = np.array([ df['x'][i], df['y'][i], df['z'][i] ])
            xx = np.linalg.norm(x - centroids[j])
            norms.append(xx)
            if i % 1000 == 0:
                print(ct, "/", r, " ", round(100 * (i / len(df)), 3))
        min_val = min(norms)
        min_index = norms.index(min_val)
        classes[i]=min_index

#This function adjusts centroid, the M step of EM
def maximize():
    refs = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
    counts = []
    sums = []
    for i in range(0,k):
        ct = 0
        summ = np.array([0,0,0])
        for j in range(0,len(classes)):
            if classes[j] == refs[i]:
                ct+=1
                summ = np.array([ df['x'][j] , df['y'][j], df['z'][j] ]) + summ
        counts.append(np.array([ct,ct,ct]))
        sums.append(summ)
    counts = np.array(counts)
    cents = sums / counts
    return cents

#Sum square error

def sumsquare():
    refs = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
    avgs = []
    for i in range(0,k):
        cts = 0
        summ = np.array([0,0,0])
        for j in range(0,len(df)):
            if classes[j] == refs[i]:
                cts+=1
                summ = np.array([ df['x'][j] , df['y'][j], df['z'][j] ]) + summ
        avgs.append(summ / cts)
    sumsq = 0
    for i in range(0,k):
        for j in range(0,len(df)):
            if classes[j] == refs[i]:
                sumsq += np.linalg.norm( np.array([ df['x'][j] , df['y'][j], df['z'][j] ]) - avgs[i] )**2
    return sumsq


classify(1,1)
runs = 5
plotter(0)
for i in range(0,runs):
    classify(i,runs)
    x = sumsquare()
    print(x)
    centroids = maximize()
    if i == (runs-1):
        plotter(i+1)

plt.show()
#Adjoin the classes vector and then plot




end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
