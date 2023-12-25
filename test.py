import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments,moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import mode
import pickle
import hw4 as h

def runOCR(testFile, path_to_Ytrue):
    letters = ['a', 'd', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'w']
    trainedFeatures = []
    labels = []
    
    for c in letters:
        filePath = f'./HW4/images/{c}.bmp'
        allFeatures = h.getFeatures(filePath)
        
        for instance in allFeatures:
            trainedFeatures.append(instance)
            labels.append(c)
        
    n = len(trainedFeatures)
    
    trainedFeatures = np.array(trainedFeatures)
    Ylabels = np.array(labels).reshape(n, 1)
    
    #normalizing trainedFeatures
    mean = trainedFeatures.mean()
    std = trainedFeatures.std()

    trainedFeatures = 1/std * (trainedFeatures - mean)
    
    #Features from training files are done being collected
    pkl_file = open(path_to_Ytrue, 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    classes = mydict[b'classes']
    locations = mydict[b'locations']

    
    testFeatures, Ytrue = h.getFeatures_test(testFile, classes, locations)
    testFeatures = 1/std * (testFeatures - mean)
    
    D = cdist(testFeatures, trainedFeatures)

    io.imshow(D)
    plt.title('Distance Matrix')
    io.show()

    Ypred = []
    nearest = []
    
    for row in D:
        D_index = np.argsort(row)
        
        for i in range(1,3):
            potential = Ylabels[D_index][i].flatten()
            nearest.append(potential)

        nearest = np.array(nearest).flatten()
        counts = np.sort(np.unique(nearest, return_counts=True))
        
        p = np.argsort(counts[1])
        predictions = counts[0][p]
        prediction = predictions[0]
        Ypred.append(prediction)
        
        nearest = []

    Ypred = np.array(Ypred).flatten()
    print("test data accuracy: ", h.accuracy(Ypred, Ytrue))


def main():
    runOCR('./HW4/images/test.bmp')


if __name__ == '__main__':
    main()
