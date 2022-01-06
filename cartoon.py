import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)


while True:

        #Gaussian Blur
    success, img = cap.read()
    img_gb = cv2.GaussianBlur(img, (11, 11) ,0)
    #lapklace filter for edge detection
    img_lp_gb = cv2.Laplacian(img_gb, cv2.CV_8U, ksize=5)
    #grayscale conversion
    img_lp_gb_grey = cv2.cvtColor(img_lp_gb, cv2.COLOR_BGR2GRAY)
    #remove additional noise -- again blur
    blur_gb = cv2.GaussianBlur(img_lp_gb_grey, (9, 9), 0)
    #Thresholding
    _, tresh_gb = cv2.threshold(blur_gb, 245, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #create a mask
    inverted_GaussianBlur = cv2.subtract(255, tresh_gb)


    # Reshape the image
    img_reshaped = img.reshape((-1,3))
    # convert to np.float32
    img_reshaped = np.float32(img_reshaped)
    # Set the Kmeans criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set the amount of K (colors)
    K = 8
    # Apply Kmeans
    _, label, center = cv2.kmeans(img_reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Covert it back to np.int8
    center = np.uint8(center)
    res = center[label.flatten()]
    # Reshape it back to an image
    img_Kmeans = res.reshape((img.shape))
    # Reduce the colors of the original image
    div = 64
    img_bins = img // div * div + div // 2

    # Convert the mask image back to color 
    inverted_GaussianBlur = cv2.cvtColor(inverted_GaussianBlur, cv2.COLOR_GRAY2RGB)
    # Combine the edge image and the binned image
    cartoon_Gaussian = cv2.bitwise_and(inverted_GaussianBlur, img_bins)
    image = np.concatenate((inverted_GaussianBlur,img_Kmeans),axis=1)
    image = np.concatenate((image,cartoon_Gaussian),axis=1)
    cv2.imshow('Final',image)
    #cv2.imwrite('Final cartoon.png', cartoon_Gaussian)
    #cv2.imwrite('WorkFlow.png', cartoon_Gaussian)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cv2.destroyAllWindows()