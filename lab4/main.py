import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        label = int(filename.split('_')[0]) 
        
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        flattened_img = img.flatten()
        images.append(flattened_img)
        labels.append(label)
    return np.array(images), np.array(labels)

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def k_nearest_neighbors(X_train, y_train, x_test, k):
    distances = []

    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], x_test)
        distances.append((distance, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    neighbor_labels = [neighbor[1] for neighbor in neighbors]
    prediction = max(set(neighbor_labels), key=neighbor_labels.count)

    return prediction

image_folder_path = 'datasets/training_data'
test_folder_path = 'datasets/real_testing_data'

X, y = load_images_from_folder(image_folder_path)
X_test, y_test = load_images_from_folder(test_folder_path)
k_value = 41

tp = 0  
fp = 0  
tn = 0
fn = 0 

for i in range(len(X_test)):
    test_example = X_test[i]
    true_label = y_test[i]
    prediction = k_nearest_neighbors(X, y, test_example, k_value)

    if prediction == true_label:
        if true_label == 1:
            tp += 1
        else:
            tn += 1
    else:
        if true_label == 1:
            fp += 1
        else:
            fn += 1

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) 

print(tp, fp, tn, fn)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')


