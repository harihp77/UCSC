#!/usr/bin/env python

import os, struct
import numpy
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from numpy import linalg as LA
from pylab import *
from numpy import *
import xlwt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import class_2d_hist
import scipy.sparse as sp



def load_feature_vector(digit, dataset="training"):
    path="."
    fname_image_array = os.path.join(path, 't10k-images-idx3-ubyte')
    fname_label_array = os.path.join(path, 't10k-labels-idx1-ubyte')

    file_label = open(fname_label_array, 'rb')
    magic_nr, size = struct.unpack(">II", file_label.read(8))
    print "********************************************"
    print "     READING  CLASS LABEL of TRAINING set"
    print "********************************************"
    print "No of class labels :%d" %(size)
    label_array = pyarray("b", file_label.read())
    print " Done reading class label from training set"
    file_label.close()

    file_image = open(fname_image_array, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", file_image.read(16))
    print "********************************************"
    print "     READING  FEATURE VECTOR of TRAINING set"
    print "********************************************"
    print "No-of-records:%d, No-of-rows:%d, No-of-col:%d" %(size, rows, cols)
    print "********************************************"
    image_array = pyarray("B", file_image.read())
    #print len(image_array)
    print " Done reading feature vector from training set"
    file_image.close()

    #ind = [ k for k in range(size) if label_array[k] in digit ]
    # following code replaces the above 1 liner..
    matching_index_array =[]
    for k in range(size):
        if (label_array[k] == digit):
            matching_index_array.append(k)

    digit_cnt  = len(matching_index_array)
    #print "Digit count", digit_cnt
    #print matching_index_array

    #images = zeros((digit_cnt, rows, cols), dtype=uint8)
    feature_vector = rows * cols
    images = zeros((digit_cnt, feature_vector), dtype=float)
    labels = zeros((digit_cnt, 1), dtype=int8)

    for i in range(digit_cnt):
        images[i] = numpy.array(image_array[ matching_index_array[i]*rows*cols : (matching_index_array[i]+1)*rows*cols ]).reshape((feature_vector))
        labels[i] = label_array[matching_index_array[i]]
    #print labels
    return images, labels


def write_image_to_text_file(image_array, digit_name):
    digt_image_text = os.path.join(".", digit_name+'_image_text')
    fdesc = open (digt_image_text, "w")
    for count in range (len(image_array)):
        fdesc.write(str(image_array[count]) + "\n")
    else :
        print ("Done with image to text operation")
    fdesc.close()

def write_complex_to_text_file(image_array, digit_name):
    digt_image_text = os.path.join(".", digit_name+'_r_text')
    fdesc = open (digt_image_text, "w")
    a = np.array(image_array)
    row,col= np.shape(a)
    for j in range(col):
        for i in range (row):
            real_number = a[i][j].real
            fdesc.write(str(real_number) )
        fdesc.write("\n")
    else :
        print ("Done with image to text operation")
    fdesc.close()


def write_p1p2_to_exl_file(image_array, digit_name):
    p1_file_xls = os.path.join(".", digit_name+"_pca.xls")
    book = xlwt.Workbook()
    sheet1=book.add_sheet("pca_data")
    at = np.matrix(image_array).transpose()
    a = np.array(at)
    row,col= np.shape(a)
    for i in range(row):
        for j in range (col):
            real_number = a[i][j].real
            r = sheet1.row(i)
            r.write(j,real_number)
    else :
        print ("Done with PCA to excel operation")
    book.save(p1_file_xls)

def write_cov_to_text_file(arr, digit_name):
    digt_image_text = os.path.join(".", digit_name+'_cov_text')
    fdesc = open ("image_cov.txt", "w")
    for row in range (len(arr)):
        for col in range(len(arr)):
            fdesc.write(str(arr[row][col]) + "\n")
    else :
        print ("Done with cov-matrix to text operation")
    fdesc.close()

def write_mean_image_excel(mu_array,digit_name):
    digt_image_xls = os.path.join(".", digit_name+'_mean.xls')
    book=xlwt.Workbook()
    sheet1=book.add_sheet("mean_plot")
    for num in range(1,len(mu_array)):
        row = sheet1.row(num)
        row.write(0,num)
        row.write(1,mu_array[num])
    book.save(digt_image_xls)
    print ("Done with mean-image to excel write for plotting")

def z_matrix(x):
    mu = np.mean(x,axis=0)
    rows, col = np.shape(x)
    z_matrix = zeros((rows, col), dtype=float)
    for  j in range(col):
        for i in range(rows):
            z_matrix[i][j] = x[i][j]-mu[j]
    print ("Done with populating ZERO matrix")
    return z_matrix


def convert_eg_vector_to_real(eg_vector):
    a = np.array(eg_vector)
    row,col= np.shape(a)
    eg_vector_ret = zeros((row, col), dtype=float)
    for j in range(col):
        for i in range (row):
            real_number = a[i][j].real
            eg_vector_ret[i][j]=real_number
    else :
        print ("Done copying complex to float to eignen vector ")
    return eg_vector_ret



def  get_pca_of_digit(digit_1 =np.arange(10), digit_2 =np.arange(10)):
    digit_names = ["zero", "one", "two", "three", "four","five","six", "seven", "eight", "nine", "ten"]
    digit_name_1 = digit_names[digit_1]
    digit_name_2 = digit_names[digit_2]
    print  "Constructing training set for digits name:", digit_name_1, digit_name_2

    # Get X - The training set
    images_1, labels_1 = load_feature_vector(digit_1, 'training')
    no_of_samples_1, no_of_cols = np.shape(images_1)

    images_2, labels_2 = load_feature_vector(digit_2, 'training')
    no_of_samples_2, no_of_cols = np.shape(images_2)

    total_no_of_samples = no_of_samples_1 + no_of_samples_2

    images = zeros((total_no_of_samples, no_of_cols), dtype=float)
    labels = zeros((total_no_of_samples, 1), dtype=int8)

    for i in range(no_of_samples_1):
        images[i] = numpy.array(images_1[i]).reshape((no_of_cols))
        labels[i] = numpy.array(labels_1[i])

    for i in range(no_of_samples_2):
        images[no_of_samples_1+i] = numpy.array(images_2[i]).reshape((no_of_cols))
        labels[no_of_samples_1+i] = numpy.array(labels_2[i])

    total_no_of_sampes, total_cols = np.shape(images)
    print "Total sampels info:  image_1:%d, image_2:%d, image_total:%d, total_cols:%d" %(no_of_samples_1, no_of_samples_2, total_no_of_sampes, total_cols)

    # Compute Z  - The Zero matrix
    Z = z_matrix(images)
    #print "Z Matrix dtype:", Z.dtype
    #print Z[0]

    # compute mean;  Returning as part of this function
    mean_vector =  np.mean(images, axis=0)

    # For testing.
    # Write mean value of image to text file
    # Alternatively we can pass any row in image[i]
    mean_vector_1 =  np.mean(images_1, axis=0)
    mean_vector_2 =  np.mean(images_2, axis=0)
    write_image_to_text_file(mean_vector_1, digit_name_1)
    write_image_to_text_file(mean_vector_2, digit_name_2)

    # Find co-variance matrix
    print "Finding COVARIANCE for digits:", digit_1, digit_2
    print "......"
    print "......"
    C =  np.ma.cov(Z, rowvar= False)
    write_cov_to_text_file(C, digit_name_1+"_"+digit_name_2)
    C_array = np.array(C)

    # Find Eigen value using co-variance
    print "Finding Eigen Vector for digits:", digit_1, digit_2
    print "......"
    print "......"
    eg_val,eg_vector = LA.eig(C_array)
    #print "Eigen VALUE"
    #print eg_val


    print "Converting Eigen values which is COMPLEX to REAL values"
    eg_vector_real = convert_eg_vector_to_real(eg_vector)
    print "Done with conversion to real values"

    #Testing if Eigen vector is normalized.
    print "Eigen normalized value-1:",LA.norm(eg_vector[[50]])
    print "Eigen normalized value-2:",LA.norm(eg_vector_real[[50]])

    print "Transpose of eigen vector"
    print "......"
    print "......"
    #WORKAROUND:
    # as per the sample, have to perform transpose operation twice
    V = np.matrix(eg_vector_real).transpose()
    TV = np.matrix(V).transpose()
    print  "TV.dtype", TV.dtype
    P = Z*TV

    # Test if the mean of P is approximately ZERO
    #p_mean = np.mean(P,axis=0)
    #print p_mean

    P1P2 = P[:,0:2]
    X_AXIS=P[:,0]
    print "X_AXIS-SHAPE", np.shape(X_AXIS)
    Y_AXIS=P[:,1]
    print "Y_AXIS-SHAPE", np.shape(Y_AXIS)

    PC =  np.ma.cov(P, rowvar= False)
    w1,v1 = LA.eig(PC)
    #print "Eigen values for PC"
    #print w1

    # Reconstruct Zero matrix
    R = P*V
    print "First row in R"
    #print R[0]

    return  np.array(P1P2), no_of_samples_1, mean_vector,eg_vector_real


def find_2_principal_comp(x, mu, eg_val):
    z = x-mu
    PCA  = z * (np.matrix(eg_val).transpose())
    print "Size of 2 principal comp is ", np.shape(PCA)
    P1P2 = np.array(PCA)
    return P1P2[0][0], P1P2[0][1]

def find_regular_probability(d1_count, d2_count):
    d1_prob=0
    d2_prob=0
    t_count = d1_count+d2_count
    if (t_count == 0):
        print "The probability can't be determined with given training set"
        return

    if (d1_count != 0):
        d1_prob =  d1_count/t_count

    if (d2_count != 0):
        d2_prob =  d2_count/t_count

    print "D1-count:%f, D2-count:%f, Total-count:%d" %(d1_count,d2_count,t_count)
    if (d1_prob > d2_prob):
        print "The probability is Digit-1"
    elif (d2_prob > d1_prob):
        print "The probability is Digit-2"
    else:
        print "The probability can't be determined with given training set"
    return



dim_1_2d,sample_1_cnt, dim_1_mean, dim_1_eg_val = get_pca_of_digit(5,6)
#write_image_to_text_file(x_axis_digit_1, "one_x")

# scatter plot
# for plotting;
no_r, no_c = np.shape(dim_1_2d)
x_axis = dim_1_2d[:,0]
y_axis = dim_1_2d[:,1]
f, ax = plt.subplots()
for i in range(sample_1_cnt):
    ax.scatter(x_axis[i],y_axis[i],c="r", marker='o')

remaining_cnt = no_r - sample_1_cnt
for i in range(remaining_cnt):
    ax.scatter(x_axis[i+sample_1_cnt],y_axis[i+sample_1_cnt],c="b", marker='o')

ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
plt.show()



#find min and max
min_x = min(x_axis)
max_x = max(x_axis)
min_y = min(y_axis)
max_y = max(y_axis)
bin_size = 15

d1_inst =  class_2d_hist.histogram_2d(dim_1_2d, min_x, max_x, min_y, max_y, bin_size)
d1_2d_hist =  d1_inst.get_2d_histogram()
print "sizeof d1_2d_hist", np.shape(d1_2d_hist)
print d1_2d_hist


d2_inst =  class_2d_hist.histogram_2d(dim_2_2d, min_x, max_x, min_y, max_y, bin_size)
d2_2d_hist =  d2_inst.get_2d_histogram()
print "sizeof d2_2d_hist", np.shape(d2_2d_hist)
print d2_2d_hist




# TEST input
test_image , test_label =  load_feature_vector(1, 'training')
# Take randomly or in this ase just 100th entry for input
test_image_input =  test_image[100,:]


# Get P1 and P2 from principal component
test_digit_pca_feature_1, test_digit_pca_feature_2 = find_2_principal_comp(test_image_input, dim_1_mean, dim_1_eg_val)
print test_digit_pca_feature_1,test_digit_pca_feature_2

# probablity
d1_count = d1_inst.get_bin_count(test_digit_pca_feature_1, test_digit_pca_feature_2)
d2_count = d2_inst.get_bin_count(test_digit_pca_feature_1, test_digit_pca_feature_2)
find_regular_probability(d1_count, d2_count)

# Bayes classificaton/Gaussian model
# Calculate Male MEAN, VARIANCE etc
d1_mean = np.mean(dim_1_2d, axis=0)
d1_covariance_array  =  np.ma.cov(dim_1_2d, rowvar= False)
d1_row_samples, d1_sample_cols = np.shape(dim_1_2d)
print d1_covariance_array


d2_mean = np.mean(dim_2_2d, axis=0)
d2_covariance_array  =  np.ma.cov(dim_2_2d, rowvar= False)
d2_row_samples, d2_sample_cols = np.shape(dim_2_2d)
print d2_covariance_array


