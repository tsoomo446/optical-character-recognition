import cv2
import numpy as np
import os
import pandas as pd

folder_path = 'images/'

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]

A = np.zeros((len(image_files), 26))

for i, filename in enumerate(image_files):
    im = cv2.imread(os.path.join(folder_path, filename))
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imr = cv2.resize(img, (5, 5))
    A[i, 0:-1] = imr.flatten()
    if filename[0] == 'c':
        A[i,-1] = 0
    elif filename[0] == 'd':
        A[i,-1] = 1
    elif filename[0] == 'm':
        A[i, -1] = 2
        
    


data = pd.DataFrame(A)
data.to_csv('data.csv', header=None, index=None)

print("CSV data file 'data.csv' has been created.")
