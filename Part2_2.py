"""
@author: Chinny Emeka
"""


import utils
import math
import numpy as np
import operator
import time





def euclidean_distance(image1, image2):
    """
    Used as similarity function for comparing features of images
    """
    
    distance = (image1 - image2)**2 #numpy performs the operation elementwise
    distance = distance.sum()
    distance = math.sqrt(distance)
    return distance;


def hamming_distance(image1, image2):
    """Similarity function for comparing features of images. Assuming equal set of features for images"""

    distance  = (image1 - image2) == 0 #true if two elements in the respective arrays are the same, false otherwise
    distance = distance.tolist() #convert back to simple python list. numpy doesn't support certain boolean operations. 
    count = 0
    for cur_pair in distance:
        if cur_pair == False:
            count += 1
    return count; 



def get_closest_neighbors(dataset, actualLabels, imageOfInterest,  k = 1):
    """
        Return the closest neigbors, using the specified similarity function to find neighbors
    """
    distance_data = []
  
    i = 0
    for cur_neighbor in dataset:
        neighbor_distance = hamming_distance(imageOfInterest, cur_neighbor)
        data = (cur_neighbor, neighbor_distance, actualLabels[i])
        distance_data.append(data) #store the image data of a neighbor, the similarity of the neighbor to our object of interest, and the actual label for the image    
        i = i + 1


    distance_data.sort(key=operator.itemgetter(1)) #sort based on the distances, with the smallest distances coming first.

    
    #we now have distances for all the images (using the image of interest as our reference point)
    #now choose best k
    best_neighbors_data = distance_data[0:k]
    best_neighbors = []
    for data_pair in best_neighbors_data:
        best_neighbors.append((data_pair[0], data_pair[2])) #store the features of the image and the true label of the image for the nearest neighbors. 
    return best_neighbors
    



def classify_image(best_neighbors):
    """Classify a single image based on the votes of its neighbors. Use majority voting"""
    class_tally = {} #dict to store count of votes for each category (i.e. class)
    for cur_neighbor in best_neighbors:
        category_name = cur_neighbor[1]
        if category_name not in class_tally:
            class_tally[category_name] = 1
        else:
            class_tally[category_name] += 1
    #we now have tallies for each of the classes. return class with most tallies
    best_class = max(class_tally.items(), key=operator.itemgetter(1))[0]

    return best_class




if __name__ == '__main__':
    start = time.time()
    
    # Global variables
    TRAIN_PATH = ('digitdata/optdigits-orig_train.txt')
    TEST_PATH = ('digitdata/optdigits-orig_test.txt')
    
    # Load data sets
    X_train, y_train = utils.load_data(TRAIN_PATH)
    X_test, y_test = utils.load_data(TEST_PATH)
    
    
    # Predict
    predicted_labels = []
    k = 2
    for image in X_test:
        best_neighbors = get_closest_neighbors(X_train, y_train, image, k)
        image_category = classify_image(best_neighbors)
        predicted_labels.append(image_category)
    #we've classified all images. now check accuracy.

    end = time.time()
    knn_accuracy = utils.accuracy(y_test, predicted_labels)
    
    print("Accuracy is {}".format(knn_accuracy))
    
        
    
    # Output
   
  





