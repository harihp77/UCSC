#!/usr/bin/env python

import os, struct
import numpy
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from numpy import linalg as LA
from pylab import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import prettyplotlib as ppl



def load_feature_vector(digit, dataset="training"):
    path="."
    fname_image_array = os.path.join(path, 't10k-images-idx3-ubyte')
    fname_label_array = os.path.join(path, 't10k-labels-idx1-ubyte')

    file_label = open(fname_label_array, 'rb')
    magic_nr, size = struct.unpack(">II", file_label.read(8))
    label_array = pyarray("b", file_label.read())
    file_label.close()

    file_image = open(fname_image_array, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", file_image.read(16))
    image_array = pyarray("B", file_image.read())
    file_image.close()

    matching_index_array =[]
    for k in range(size):
        if (label_array[k] == digit):
            matching_index_array.append(k)

    digit_cnt  = len(matching_index_array)

    no_of_features= rows * cols
    images = zeros((digit_cnt, no_of_features), dtype=float)
    labels = zeros((digit_cnt, 1), dtype=int8)

    for i in range(digit_cnt):
        images[i] = numpy.array(image_array[ matching_index_array[i]*rows*cols : (matching_index_array[i]+1)*rows*cols ]).reshape((no_of_features))
        labels[i] = label_array[matching_index_array[i]]
    return images, labels

def display_image(image, is_display):
    if is_display:
        plt.imshow(image)
        plt.show()

def write_image_to_pgm_file(image_array, name, is_write=False):
    if is_write:
        digt_image_text = os.path.join(".", name+'.pgm')
        fdesc = open (digt_image_text, "w")
        fdesc.write(("P2  # Image") + "\n")
        fdesc.write(("28  # Rows") + "\n")
        fdesc.write(("28  # Cols") + "\n")
        fdesc.write(("255 # Max ") + "\n")
        for count in range (len(image_array)):
            fdesc.write(str(image_array[count]) + "\n")
        else :
            print ("Done with image to PGM operation")
        fdesc.close()


def plot_mean_vector(array, is_plot=False):
    if is_plot:
        fg, ax = plt.subplots()
        for i in range(len(array)):
            plt.plot(i, array[i], ".",c="b")
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_title("Mean image plot")
        plt.show()
        fg.savefig('mean_plot.jpg')

def plot_scatter_2d(x_axis, y_axis, digit_1, digit_2, digit_3, digit_1_count, digit_2_count,digit_3_count, is_plot=False):
    digit_names = ["zero", "one", "two", "three", "four","five","six", "seven", "eight", "nine", "ten"]
    digit_1_name = digit_names[digit_1]
    digit_2_name = digit_names[digit_2]
    digit_3_name = digit_names[digit_3]
    if is_plot:
        f, ax = plt.subplots()
        for i in range(digit_1_count):
            plt_1 = ax.scatter(x_axis[i],y_axis[i],c="r", marker='o', label=digit_1_name)
        for i in range(digit_2_count):
            plt_2 = ax.scatter(x_axis[i+digit_1_count],y_axis[i+digit_1_count],c="b", marker='o', label=digit_2_name)
        for i in range(digit_3_count):
            plt_3 = ax.scatter(x_axis[i+digit_1_count+digit_2_count],y_axis[i+digit_1_count+digit_2_count],c="g", marker='o', label=digit_3_name)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        title = "Plot for digits - " +  digit_1_name + ", " + digit_2_name + ", " + digit_3_name
        ax.set_title(title)
        plt.legend([plt_1, plt_2, plt_3], [digit_1_name, digit_2_name, digit_3_name])
        plt.show()
        f.savefig('digits_2d_plot.jpg')

def plot_scatter_3d(x_axis, y_axis, z_axis, digit_1, digit_2, digit_3, digit_1_count, digit_2_count,digit_3_count, is_plot=False):
    digit_names = ["zero", "one", "two", "three", "four","five","six", "seven", "eight", "nine", "ten"]
    digit_1_name = digit_names[digit_1]
    digit_2_name = digit_names[digit_2]
    digit_3_name = digit_names[digit_3]
    if is_plot:
        #f, ax = plt.subplots()
        f = plt.figure()
        ax = f.add_subplot(111, projection = '3d')
        for i in range(digit_1_count):
            plt_1 = ax.scatter(x_axis[i],y_axis[i],z_axis[i],c="r", marker='o', label=digit_1_name)
        for i in range(digit_2_count):
            plt_2 = ax.scatter(x_axis[i+digit_1_count],y_axis[i+digit_1_count],z_axis[i+digit_1_count], c="b", marker='o', label=digit_2_name)
        for i in range(digit_3_count):
            plt_3 = ax.scatter(x_axis[i+digit_1_count+digit_2_count],y_axis[i+digit_1_count+digit_2_count], z_axis[i+digit_1_count+digit_2_count], c="g", marker='o', label=digit_3_name)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_zlabel("z axis")
        title = "Plot for digits - " +  digit_1_name + ", " + digit_2_name + ", " + digit_3_name
        ax.set_title(title)
        plt.legend([plt_1, plt_2, plt_3], [digit_1_name, digit_2_name, digit_3_name])
        plt.show()
        f.savefig('digits_2d_plot.jpg')




def plot_scatter_k_means_2d(n_clusters, clusters, is_plot=False):
    if is_plot:
        class_name = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7"]
        colors = ["lime", "aqua","deeppink", "orangered","dodgerblue"]
        fig, ax = plt.subplots()
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        title = "Plot for K-Means class"
        ax.set_title(title)
        for i in range(0, n_clusters):
            x_axis = tuple(x[0] for x in clusters[i])
            y_axis = tuple(x[1] for x in clusters[i])
            ppl.scatter(ax,x_axis, y_axis, color=colors[i], label=class_name[i])
        ppl.legend(ax)
        plt.show()
        fig.savefig('k_means_classification_plot.jpg')

def plot_scatter_k_means_3d(n_clusters, clusters, is_plot=False):
    if is_plot:
        class_name = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7"]
        colors = ["lime", "aqua","deeppink", "orangered","dodgerblue"]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_zlabel("z axis")
        title = "Plot for K-Means class"
        ax.set_title(title)
        for i in range(0, n_clusters):
            x_axis = tuple(x[0] for x in clusters[i])
            y_axis = tuple(x[1] for x in clusters[i])
            z_axis = tuple(x[2] for x in clusters[i])
            ppl.scatter(ax,x_axis, y_axis, z_axis, color=colors[i], label=class_name[i])
        ppl.legend(ax)
        plt.show()
        fig.savefig('k_means_classification_plot.jpg')


def get_sequenced_matching_digits_array_from_training_set(digit_1 =np.arange(10), digit_2 =np.arange(10), digit_3 =np.arange(10)):
    digit_names = ["zero", "one", "two", "three", "four","five","six", "seven", "eight", "nine", "ten"]
    digit_name_1 = digit_names[digit_1]
    digit_name_2 = digit_names[digit_2]
    digit_name_3 = digit_names[digit_3]
    print  "Constructing training set for digits :", digit_name_1, digit_name_2, digit_name_3
    # Get X - The training set
    images_1, labels_1 = load_feature_vector(digit_1, 'training')
    no_of_samples_1, no_of_cols = np.shape(images_1)

    images_2, labels_2 = load_feature_vector(digit_2, 'training')
    no_of_samples_2, no_of_cols = np.shape(images_2)

    images_3, labels_3 = load_feature_vector(digit_3, 'training')
    no_of_samples_3, no_of_cols = np.shape(images_3)

    total_no_of_samples = no_of_samples_1 + no_of_samples_2 + no_of_samples_3

    images = zeros((total_no_of_samples, no_of_cols), dtype=float)
    labels = zeros((total_no_of_samples, 1), dtype=int8)

    for i in range(no_of_samples_1):
        images[i] = numpy.array(images_1[i]).reshape((no_of_cols))
        labels[i] = numpy.array(labels_1[i])

    for i in range(no_of_samples_2):
        images[no_of_samples_1+i] = numpy.array(images_2[i]).reshape((no_of_cols))
        labels[no_of_samples_1+i] = numpy.array(labels_2[i])

    for i in range(no_of_samples_3):
        images[no_of_samples_1+no_of_samples_2+i] = numpy.array(images_3[i]).reshape((no_of_cols))
        labels[no_of_samples_1+no_of_samples_2+i] = numpy.array(labels_3[i])

    total_no_of_samples, total_features = np.shape(images)
    print "Total samples %s:%d, %s:%d, %s:%d, total_samples:%d, total_features:%d" %(digit_name_1, no_of_samples_1, digit_name_2, no_of_samples_2, digit_name_3, no_of_samples_3, total_no_of_samples, total_features)

    return images, no_of_samples_1, no_of_samples_2, no_of_samples_3




def  get_pca_of_x(X):

    # compute mean;
    mean_vector =  np.mean(X, axis=0)
    write_image_to_pgm_file(mean_vector, "mean_vector", False)
    display_image(mean_vector.reshape(28,28),False)

    # Compute Z  - The Zero matrix
    Z = X - mean_vector
    plot_mean_vector(np.array(mean_vector), False)

    # Find co-variance matrix
    print "Finding COVARIANCE for X"
    C =  np.ma.cov(Z, rowvar= False)

    # Find Eigen value using co-variance
    print "Finding Eigen Vector for X:"
    eg_val,eg_vector = LA.eig(C)

    print "Converting Eigen values which is COMPLEX to REAL values"
    eg_vector_real = np.real(eg_vector)

    #WORKAROUND: Perform double transpose
    # as per the sample, have to perform transpose operation twice
    V = np.matrix(eg_vector_real).transpose()
    TV = np.matrix(V).transpose()

    print  "Finding P"
    P = Z*TV

    P1P2 = P[:,0:2]
    P1P2P3 = P[:,0:3]

    #Following code is needed for assignment-03. Hence commenting here..
    '''
    V1V2 = V[0:2,:]

    print  "Finding PC"
    PC =  np.ma.cov(P, rowvar= False)

    print  "Deriving PC-Eigne Vector"
    w1,v1 = LA.eig(PC)

    # Reconstruct Zero matrix
    #R = P*V
    print  "Construct Reduced set"
    R = P1P2*V1V2
    # Construct N*d matrix from reduced dimensions
    X_NEW = R + mean_vector

    #testing to dump random value in reduced set
    arr_index=210
    write_image_to_pgm_file(X[arr_index], digit_name_1+"_orig_210" )
    display_image(X[arr_index].reshape(28,28),False)
    '''
    return  np.array(P1P2), np.array(P1P2P3)





# k-Means Algorithm
def classify_to_class_cluster(X, mu):
    clusters  = {}
    matching_class_id=[]
    for x in X:
        class_id = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        #class_id = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    #for i in enumerate(mu)], key=lambda (x,y):y))[0]
        #print class_id, x
        try:
            clusters[class_id].append(x)
            matching_class_id.append(class_id)
        except KeyError:
            clusters[class_id] = [x]
    return clusters, matching_class_id


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def find_centroid(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters, class_array  = classify_to_class_cluster(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)


def  get_matching_class_entries_from_x(class_id, X, class_array):
    matching_class_X_entries =[]
    size = np.size(class_array)
    for k in range(size):
        if (class_array[k] == class_id):
            matching_class_X_entries.append(X[k])
    return matching_class_X_entries

def print_random_images(matching_class_entries, class_name, no_of_images):
    rows,cols=np.shape(matching_class_entries)
    for i in range(no_of_images):
        rand_index = random.randint(0, rows)
        file_name = class_name +  "_rand_image_" +  str(i)
        write_image_to_pgm_file(matching_class_entries[rand_index], file_name, True )


###########################################################################
#                        MAIN program starts here                         #
###########################################################################

# Initialize
digit_1 = 2
digit_2 = 3
digit_3 = 5
is_two_dimention = False
# K-Means init
n_clusters = 5

# Parse and get matching digits
X, digit_1_cnt, digit_2_cnt, digit_3_cnt = get_sequenced_matching_digits_array_from_training_set(digit_1, digit_2, digit_3 )

# Find PCA
pca_2d , pca_3d = get_pca_of_x(X)

# scatter plot from TRUE values.
no_r, no_c = np.shape(pca_2d)
x_axis = pca_2d[:,0]
y_axis = pca_2d[:,1]
z_axis = pca_3d[:,2]
if ( is_two_dimention):
    plot_scatter_2d(x_axis, y_axis, digit_1, digit_2, digit_3, digit_1_cnt, digit_2_cnt, digit_3_cnt, True )
else:
    #plot_scatter_3d(x_axis, y_axis, z_axis, 8, 5, 6, digit_1_cnt, digit_2_cnt, digit_3_cnt, True )
    plot_scatter_3d(x_axis, y_axis, z_axis, digit_1, digit_2, digit_3, digit_1_cnt, digit_2_cnt, digit_3_cnt, True )



if ( is_two_dimention):
    X_K = pca_2d
else:
    X_K = pca_3d


# Cluster points
mu, clusters = find_centroid(X_K, n_clusters)

# Validate the correctness
clusters, class_array  = classify_to_class_cluster(X_K, mu)

# Plot each cluster
if (is_two_dimention):
    plot_scatter_k_means_2d(n_clusters, clusters, True)
else:
    plot_scatter_k_means_3d(n_clusters, clusters, True)



# Lets print the image randomly from members of the class
no_of_images_to_print=20
class_name = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", "Class_7"]
for i in range(n_clusters):
    matching_class_entries = get_matching_class_entries_from_x(i, X, class_array)
    print_random_images(matching_class_entries, class_name[i], no_of_images_to_print)


