import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion, binary_opening, binary_closing, binary_dilation
from skimage.measure import label, regionprops, moments,moments_central, moments_normalized, moments_hu
from skimage import io, exposure
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

def getFeatures(img_path):
    img = io.imread(img_path)
    th = 236          #a 180, o 240
    #if img_path == './HW4/images/o.bmp':
    #    th = 235
    #    img_binary = (img < th).astype(np.double)
    #    img_binary = binary_opening(img_binary) 
    #elif img_path == './HW4/images/d.bmp':
    #    th = 240
    #    img_binary = (img < th).astype(np.double)
    #    img_binary = binary_dilation(img_binary)  
    #elif img_path == './HW4/images/a.bmp':    
    #    img_binary = (img < th).astype(np.double)
    #    img_binary = binary_opening(img_binary)        
    #else:
    #    img_binary = (img < th).astype(np.double) 
    
    img_binary = (img < th).astype(np.double)
    img_label = label(img_binary, background=0)

    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()

    Features = []
    
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        
        height = maxr - minr 
        width = maxc - minc  
        if height < 10 or width < 10:
            continue
        
        ax.add_patch(Rectangle((minc,minr), maxc - minc, maxr - minr, 
                            fill=False,
                            edgecolor = 'red', 
                            linewidth=1))
        
        roi = img_binary[(minr):(maxr+5), (minc):(maxc+5)]
          
        m = moments(roi)
        cc= m[0,1] / m[0, 0]
        cr= m[1,0] / m[0, 0]
        mu = moments_central(roi, center=(cr, cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        
        Features.append(hu)
        
    ax.set_title('Bounding Boxes')
    io.show()
    plt.clf()
    
    return Features
    
def getFeatures_test(img_path, classes, locations):
    Ytrue = []
    img = io.imread(img_path)
    th = 236          #a 180, o 240
    
    img_binary = (img < th).astype(np.double)
    img_label = label(img_binary, background=0)

    regions = regionprops(img_label)
    io.imshow(img_binary)

    ax = plt.gca()

    Features = []
    
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        
        for i in range(len(locations)):
            x = locations[i][0]
            y = locations[i][1]
            
            if x > minc and x < maxc and y > minr and y < maxr:
                trueLabel = classes[i]
                Ytrue.append(trueLabel)
            
        height = maxr - minr 
        width = maxc - minc  
        if height < 10 or width < 10:
            continue
        
        ax.add_patch(Rectangle((minc,minr), maxc - minc, maxr - minr, 
                            fill=False,
                            edgecolor = 'red', 
                            linewidth=1))
        
        roi = img_binary[(minr):(maxr+5), (minc):(maxc+5)]
          
        m = moments(roi)
        cc= m[0,1] / m[0, 0]
        cr= m[1,0] / m[0, 0]
        mu = moments_central(roi, center=(cr, cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        
        Features.append(hu)
        
    ax.set_title('Bounding Boxes')
    io.show()
    
    return Features, Ytrue

def accuracy(Ypred, Ytrue):
    count = 0
    
    for i in range(Ypred.size):
        if Ypred[i] == Ytrue[i]:
            count += 1
            
    return count / Ypred.size














def main():
    letters = ['a', 'd', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'w']
    trainedFeatures = []
    labels = []
    
    for c in letters:
        filePath = f'./HW4/images/{c}.bmp'
        allFeatures = getFeatures(filePath)
        
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
    testFeatures = getFeatures('./HW4/images/test.bmp')

    D = cdist(trainedFeatures, trainedFeatures)
    
    Ytrue = []
    Ypred = []
    nearest =[]
    
    for row in D:
        D_index = np.argsort(row)
        Ytrue.append(Ylabels[D_index][0])
        
        for i in range(1,2):
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
    Ytrue = np.array(Ytrue).flatten()
    
    confM = confusion_matrix(Ytrue, Ypred)
    print(confM)
    
    print("training accuracy: ", accuracy(Ypred, Ytrue))


if __name__ == '__main__':
    main()