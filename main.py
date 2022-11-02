import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# https://www.kaggle.com/code/heeraldedhia/kmeans-clustering-for-customer-data/data


def preprocessing(file_name,K_num,random_seed):
    csv = pd.read_csv(file_name)
    # shuffle row
    np.random.seed(random_seed)
    csv = csv.sample(frac=1)

    feature = csv.iloc[:,1:3] # SepalLengthCm,SepalWidthCm

    return feature, K_num


# choose centroid init randomly
def centroid_init_randomly(feature):
    pick_centroid = []
    for i in range(len(feature.iloc[0])):
        pick_centroid += [np.random.uniform(min(feature.iloc[:,i]), max(feature.iloc[:,i]), K)]

    centroid = np.array(pick_centroid)
    centroid = centroid.T
    return centroid



def L2_distance(centroid,pts):
    return sum((centroid - pts)**2)**(0.5)



# most closer centroid is same => same cluster group

def closest(feature,K,centroid):
    Distance = np.zeros((len(feature),K))
    for i,pts in enumerate(feature.iloc()):
        for j,centro in enumerate(centroid):
            Distance[i][j] = L2_distance(centro,pts.to_numpy())

    closest_index = np.array([[np.argmin(x)] for x in Distance])
    return closest_index

# assign class
def assign_class(feature,closest_index):
    F = feature.to_numpy()
    A = []
    for i in range(len(F)):
        A += [np.append(F[i],closest_index[i])]
    A = np.array(A)
    return A

# visualize  initial centroid
def visualize_init(A,centroid):
    x = A[:,0]
    y = A[:,1]
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('SepalLengthCm')
    plt.ylabel('SepalWidthCm')
    for i in range(len(centroid)):
        plt.scatter(centroid[i][0], centroid[i][1], c= 'red')

    plt.show()

    return 0

# visualize clustering data
def clustering(clustered_data,K,centroid):

    x = []
    y = []
    #c = ['spring', 'summer', 'autumn']
    for pts in clustered_data:
        if int(pts[2]) == K:
            x += [pts[0]]
            y += [pts[1]]

    plt.scatter(x,y, alpha=0.5 )
    plt.scatter(centroid[K][0], centroid[K][1], c = 'red')

    plt.xlabel('SepalLengthCm')
    plt.ylabel('SepalWidthCm')

    return x,y


if __name__ == '__main__':

    feature, K = preprocessing('iris.csv', K_num = 3, random_seed= 3823)
    centroid = centroid_init_randomly(feature)
    closest_index = closest(feature,K,centroid)
    pts_and_class = assign_class(feature, closest_index)
    visualize_init(pts_and_class,centroid)

    centroid_x = []
    centroid_y = []
    centroid2 = []
    for i in range(K):
        x, y = clustering(pts_and_class, i, centroid)
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        centroid2 += [[centroid_x,centroid_y]]
    plt.show()



    #iteration
    for j in range(10):
        centroid = centroid2

        closest_index = closest(feature, K, centroid)
        pts_and_class = assign_class(feature, closest_index)
        #visualize_init(pts_and_class, centroid)

        centroid_x = []
        centroid_y = []
        centroid2 = []
        for i in range(K):
            x, y = clustering(pts_and_class, i, centroid)
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)
            centroid2 += [[centroid_x, centroid_y]]
        plt.show()

        for i in range(K):
            x, y = clustering(pts_and_class, i, centroid2)
        plt.show()



