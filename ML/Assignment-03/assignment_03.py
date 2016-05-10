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



def  get_pca_of_digit(digits=np.arange(10)):
    digit_names = ["one", "two", "three", "four","five","six", "seven", "eight", "nine", "ten"]
    digit_name = digit_names[digits]
    print  "Digit name:", digit_name

    # Get X - The training set
    images, labels = load_feature_vector(digits, 'training')
    print images[0]
    # Compute Z  - The Zero matrix
    Z = z_matrix(images)
    #print "Z Matrix dtype:", Z.dtype
    #print Z[0]

    # compute mean and write in excel.
    mean_vector =  np.mean(images, axis=0)
    mean_image = images.mean(axis=0)
    max_image_val = max(mean_image)
    #print  "MAX value", max_image_val
    #print  "MIN value", min(mean_image)

    # Write mean value of image to text file
    # Alternatively we can pass any row in image[i]
    write_image_to_text_file(mean_vector, digit_name)
    write_mean_image_excel(mean_vector,digit_name)

    # Find co-variance matrix
    print "Finding COVARIANCE for digit:", digits
    print "......"
    print "......"
    C =  np.ma.cov(images, rowvar= False)
    write_cov_to_text_file(C, digit_name)

    # Find Eigen value using co-variance
    print "Finding Eigen Vector for digit:", digits
    print "......"
    print "......"
    eg_val,eg_vector = LA.eig(C)
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
    #P = Z*TV
    P = Z*TV

    # Test if the mean of P is approximately ZERO
    #p_mean = np.mean(P,axis=0)
    #print "P Mean"
    #print p_mean

    P1P2 = P[:,0:2]
    X_AXIS=P[:,0]
    #X_AXIS=P[:,783]
    print "X_AXIS-SHAPE", np.shape(X_AXIS)
    Y_AXIS=P[:,1]
    #Y_AXIS=P[:,782]
    print "Y_AXIS-SHAPE", np.shape(Y_AXIS)

    PC =  np.ma.cov(P, rowvar= False)
    w1,v1 = LA.eig(PC)
    #print "Eigen values for PC"
    #print w1
    X_AXIS_array = np.array(X_AXIS)
    Y_AXIS_array = np.array(Y_AXIS)

    # Reconstruct Zero matrix
    R = P*V
    print "First row in R"
    #print R[0]

    return  X_AXIS_array, Y_AXIS_array, np.array(P1P2)


def find_min_max(a1,a2):
    first_d_min_size = min(a1)
    first_d_max_size = max (a1)
    print "Fist array - MIN:%d, MAX:%d" %(first_d_min_size, first_d_max_size)
    second_d_min_size = min(a2)
    second_d_max_size = max (a2)
    print "Second array MIN:%d, MAX:%d" %(second_d_min_size, second_d_max_size)

    if (first_d_min_size > second_d_min_size):
        min_d = first_d_min_size
    else:
        min_d = second_d_min_size

    if (first_d_max_size > second_d_max_size):
        max_d = first_d_max_size
    else:
        max_d = second_d_max_size
    return min_d, max_d




x_axis_digit_1, y_axis_digit_1, dimention1_2d  = get_pca_of_digit(1)
x_axis_digit_2, y_axis_digit_2, dimention2_2d = get_pca_of_digit(8)
write_image_to_text_file(x_axis_digit_1, "one_x")

# scatter plot
#f, ax = plt.subplots()
#ax.scatter(x_axis_digit_1,y_axis_digit_1,c="r", marker='o')
#ax.scatter(x_axis_digit_2,y_axis_digit_2,c="b", marker='o')
#ax.set_xlabel("x axis")
#ax.set_ylabel("y axis")
#plt.show()



#find min and max
min_d1, max_d1 = find_min_max(x_axis_digit_1, x_axis_digit_2)
min_d2, max_d2 = find_min_max(y_axis_digit_1, y_axis_digit_2)
print "Min:%f and Max:%f of dimention-1" %(min_d1, max_d1)
print "Min:%f and Max:%f of dimention-2" %(min_d2, max_d2)
bin_size = 15

d1_2d_hist_inst =  class_2d_hist.histogram_2d(dimention1_2d, min_d1, max_d1, min_d2, max_d2, bin_size)
d1_2d_hist =  d1_2d_hist_inst.get_2d_histogram()
print "sizeof d1_2d_hist", np.shape(d1_2d_hist)
print d1_2d_hist
