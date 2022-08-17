import numpy as np
import cv2
import matplotlib.pyplot as plt

#Read images in and convert to grayscale
image1 = cv2.imread("filter1_img.jpg")
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.imread("filter2_img.jpg")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


#Storing Gaussian Filters
three = (1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]])
five = (1/273)*np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
DoGgx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
DoGgy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

#Convolution Function
def convolution(image, kernel):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    return output

#Operations on image 1
image1_three = convolution(image1, three)
image1_five = convolution(image1, five)
image1_DoGgx =  convolution(image1, DoGgx)
image1_DoGgy = convolution(image1, DoGgy)
#Manually calculating the sobel filter result
image1_sobel = np.sqrt(image1_DoGgx**2 + image1_DoGgy**2)

#Operations on image 2
image2_three = convolution(image2, three)
image2_five = convolution(image2, five)
image2_DoGgx =  convolution(image2, DoGgx)
image2_DoGgy = convolution(image2, DoGgy)
#Manually calculating the sobel filter result
image2_sobel = np.sqrt(image2_DoGgx**2 + image2_DoGgy**2)

plt.imshow(image1_five, cmap='gray')
plt.title("Image 1, convoluted with 5x5")
plt.show()
plt.clf()
plt.imshow(image1_DoGgx, cmap='gray')
plt.title("Image 1, convoluted with DoG gx")
plt.show()
plt.clf()
plt.imshow(image1_DoGgy, cmap='gray')
plt.title("Image 1, convoluted with DoG gy")
plt.show()
plt.clf()
plt.imshow(image1_sobel, cmap='gray')
plt.title("Image 1, Sobel filter")
plt.show()
plt.clf()

plt.imshow(image2_three, cmap='gray')
plt.title("Image 2, convoluted with 3x3")
plt.show()
plt.clf()
plt.imshow(image2_five, cmap='gray')
plt.title("Image 2, convoluted with 5x5")
plt.show()
plt.clf()
plt.imshow(image2_DoGgx, cmap='gray')
plt.title("Image 2, convoluted with DoG gx")
plt.show()
plt.clf()
plt.imshow(image2_DoGgy, cmap='gray')
plt.title("Image 2, convoluted with DoG gy")
plt.show()
plt.clf()
plt.imshow(image2_sobel, cmap='gray')
plt.title("Image 2, Sobel filter")
plt.show()
plt.clf()
