import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

img1 = cv2.imread('/Users/chand/Desktop/CompVision/Programming/Programming1/SIFT1_img.jpg')
img2 = cv2.imread('/Users/chand/Desktop/CompVision/Programming/Programming1/SIFT2_img.jpg')

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

gray1 = cv2.resize(gray1, (3008,3008))

sift = cv2.SIFT_create()
kp1 = sift.detect(gray1,None)
kp1vecs = np.array(sift.detectAndCompute(gray1,None)[1])
kp2 = sift.detect(gray2,None)
kp2vecs = np.array(sift.detectAndCompute(gray2,None)[1])

img1 = cv2.drawKeypoints(gray1,kp1,img1)
img2 = cv2.drawKeypoints(gray2,kp2,img2)


plt.imshow(img1)
plt.show()
plt.clf()
plt.imshow(img2)
plt.show()
plt.clf()


normsmin = []
maxrng = len(kp1vecs)
for i in range(0,maxrng):
    norms = []
    if i % 10 == 0:
        print(round(i/maxrng,8))
    for j in range(0,len(kp2vecs)):
        x = np.linalg.norm(kp1vecs[i] - kp2vecs[j])
        norms.append(x)
    mini = min(norms)
    index = norms.index(mini)
    normsmin.append([mini,i,index])

normsmin.sort()
print(normsmin)

matches = []
subset = 100
for i in range(0,700):
    item = cv2.DMatch(normsmin[i][1],normsmin[i][2],0)
    matches.append(item)


matchedimage = cv2.drawMatches(gray1,kp1,gray2,kp2,matches,None)
plt.imshow(matchedimage)
plt.show()

