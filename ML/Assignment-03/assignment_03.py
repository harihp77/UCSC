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
    print " Done reading feature vector from training set"
    file_image.close()

    #ind = [ k for k in range(size) if label_array[k] in digit ]
    # following code replaces the above 1 liner..
    matching_index_array =[]
    for k in range(size):
        if (label_array[k] == digit):
            matching_index_array.append(k)

    digit_cnt  = len(matching_index_array)

    feature_vector = rows * cols
    images = zeros((digit_cnt, feature_vector), dtype=float)
    labels = zeros((digit_cnt, 1), dtype=int8)

    for i in range(digit_cnt):
        images[i] = numpy.array(image_array[ matching_index_array[i]*rows*cols : (matching_index_array[i]+1)*rows*cols ]).reshape((feature_vector))
        labels[i] = label_array[matching_index_array[i]]
    return images, labels

def display_image(image, is_display):
    if is_display:
        plt.imshow(image)
        plt.show()

def write_image_to_pgm_file(image_array, digit_name, is_write=False):
    if is_write:
        digt_image_text = os.path.join(".", digit_name+'_image.pgm')
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

def write_cov_to_pgm_file(arr, digit_name, is_write=False):
    if is_write:
        digt_image_text = os.path.join(".", digit_name+'_cov.pgm')
        fdesc = open (digt_image_text, "w")
        fdesc.write(("P2  # Image") + "\n")
        fdesc.write(("784  # Rows") + "\n")
        fdesc.write(("784  # Cols") + "\n")
        fdesc.write(("255 # Max ") + "\n")
        for row in range (len(arr)):
            for col in range(len(arr)):
                fdesc.write(str(arr[row][col]) + "\n")
        else :
            print ("Done with cov-matrix to PGM operation")
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

def get_x_new(x,mu):
    rows, col = np.shape(x)
    x_new= zeros((rows, col), dtype=float)
    for  j in range(col):
        for i in range(rows):
            x_new[i][j] = x[i][j]+mu[j]
    print ("Done with populating reduced dim matrix")
    return x_new


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

def plot_scatter(x_axis, y_axis, x_label, y_label, x_count, y_count, is_plot=False):
    if is_plot:
        f, ax = plt.subplots()
        for i in range(x_count):
            plt_1 = ax.scatter(x_axis[i],y_axis[i],c="r", marker='o', label=x_label)
        for i in range(y_count):
            plt_2 = ax.scatter(x_axis[i+x_count],y_axis[i+x_count],c="b", marker='o', label=y_label)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_title("Digits - Two & Four_plot")
        plt.legend([plt_1, plt_2], ["TWO", "FOUR"])
        plt.show()
        f.savefig('Two_Four_plot.jpg')


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

    # compute mean;  Returning as part of this function
    mean_vector =  np.mean(images, axis=0)
    write_image_to_pgm_file(mean_vector, "mean_vector", False)
    display_image(mean_vector.reshape(28,28),False)

    # Compute Z  - The Zero matrix
    Z = images - mean_vector
    plot_mean_vector(np.array(mean_vector), False)

    # Find co-variance matrix
    print "Finding COVARIANCE for digits:", digit_1, digit_2
    print "......"
    C =  np.ma.cov(Z, rowvar= False)
    write_cov_to_pgm_file(C, digit_name_1+"_"+digit_name_2)

    # Find Eigen value using co-variance
    print "Finding Eigen Vector for digits:", digit_1, digit_2
    print "......"
    eg_val,eg_vector = LA.eig(C)
    #print "Eigen VALUE"
    #print eg_val


    print "Converting Eigen values which is COMPLEX to REAL values"
    eg_vector_real = np.real(eg_vector)

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
    print  "Finding P"
    P = Z*TV

    # Test if the mean of P is approximately ZERO
    #p_mean = np.mean(P,axis=0)
    #print p_mean

    P1P2 = P[:,0:2]
    V1V2 = V[0:2,:]

    print  "Finding PC"
    PC =  np.ma.cov(P, rowvar= False)

    print  "Deriving PC-Eigne Vector"
    w1,v1 = LA.eig(PC)
    #print "Eigen values for PC"
    #print w1

    # Reconstruct Zero matrix
    #R = P*V
    print  "Construct Reduced set"
    R = P1P2*V1V2
    # Construct N*d matrix from reduced dimensions
    #X_NEW = get_x_new(np.array(R),np.array(mean_vector))
    X_NEW = R + mean_vector

    arr_index=210
    x_new_dump = X_NEW[arr_index,:]
    write_image_to_pgm_file(images[arr_index], digit_name_1+"_orig_210" )
    display_image(images[arr_index].reshape(28,28),False)
    return  np.array(P1P2), no_of_samples_1, mean_vector,V


def find_2_principal_comp(x, mu, eg_vec):
    z = x-mu
    PCA  = z * (np.matrix(eg_vec).transpose())
    #print "Size of 2 principal comp is ", np.shape(PCA)
    P1P2 = np.array(PCA)
    return P1P2[0][0], P1P2[0][1]

def find_regular_probability(d1_count, d2_count):
    d1_prob=0
    d2_prob=0
    t_count = d1_count+d2_count
    if (t_count == 0):
        #print "The probability can't be determined with given training set"
        return 3

    if (d1_count != 0):
        d1_prob =  d1_count/t_count

    if (d2_count != 0):
        d2_prob =  d2_count/t_count

    #print "D1-count:%f, D2-count:%f, Total-count:%d" %(d1_count,d2_count,t_count)
    if (d1_prob > d2_prob):
        #print "The probability is Digit-1"
        return 1
    elif (d2_prob > d1_prob):
        #print "The probability is Digit-2"
        return 2
    else:
        #print "The probability can't be determined with given training set"
        return 3
    return

def compute_bayseian_classifier(n, x, y, mean_vector, co_variance_matrix):
    determinant_sigma_co_variance = np.linalg.det(co_variance_matrix)
    sqrt_determinant_sigma_co_variance = math.sqrt(determinant_sigma_co_variance)
    n_by_sqrt_determinant_sigma_co_variance = n/sqrt_determinant_sigma_co_variance
    n_dim_sample_minus_mean_vector =  ((x,y)- mean_vector)
    inverse_sigma_co_variance = np.linalg.inv(co_variance_matrix)
    transpose_n_dim_sample_minus_mean_vector = np.matrix(n_dim_sample_minus_mean_vector).transpose()
    return_value = (n_by_sqrt_determinant_sigma_co_variance * np.exp ((-1/2) * np.matrix(n_dim_sample_minus_mean_vector) * inverse_sigma_co_variance * transpose_n_dim_sample_minus_mean_vector))
    return return_value

def find_probability_by_bayseian(x, y, d1_mean, d1_covariance,d2_mean, d2_covariance, n1, n2):
    #print  "--------------------------------------------"
    #print  "Finding probability through Bayes Classifier"
    #print  "--------------------------------------------"
    bc_d1 = compute_bayseian_classifier(n1, x, y, d1_mean, d1_covariance)
    bc_d2 = compute_bayseian_classifier(n2, x, y, d2_mean, d2_covariance)
    #print "BC-Digit-1-Result:%f, BC-Digit-2-Result:%f" %(bc_d1,bc_d2)
    if (bc_d1 > bc_d2):
        return 1
    elif(bc_d2 > bc_d1):
        return 2
    else:
        return 3

def find_classifier_result(test_image, d1_inst, d2_inst, is_match,  d1_mean, d1_covariance, d2_mean, d2_covariance, d1_n, d2_n, digit = np.arange(10)):
    # Take randomly or in this ase just 100th entry for input
    digit_names = ["zero", "one", "two", "three", "four","five","six", "seven", "eight", "nine", "ten"]
    digit_name_1 = digit_names[digit]
    no_of_entries, no_of_col = np.shape(test_image)

    no_of_bc_true_match=0
    no_of_bc_false_match=0
    no_of_bc_indt = 0
    no_of_prob_true_match=0
    no_of_prob_false_match=0
    no_of_prob_indt = 0

    for i in range(no_of_entries):
        test_image_input =  test_image[i,:]
        # Get P1 and P2 from principal component
        x_test_input, y_test_input = find_2_principal_comp(test_image_input, mean_value, eigen_vector)
        #Bayes classification
        bc_rc = find_probability_by_bayseian(x_test_input, y_test_input, d1_mean, d1_covariance, d2_mean, d2_covariance, d1_n, d2_n)
        if (bc_rc == 1):
            no_of_bc_true_match += 1
        elif (bc_rc == 2):
            no_of_bc_false_match += 1
        else:
            no_of_bc_indt += 1

        # probablity
        d1_count = d1_inst.get_bin_count(x_test_input, y_test_input)
        d2_count = d2_inst.get_bin_count(x_test_input, y_test_input)
        prob_rc = find_regular_probability(d1_count, d2_count)
        if (prob_rc == 1):
            no_of_prob_true_match += 1
        elif (prob_rc == 2):
            no_of_prob_false_match += 1
        else:
            no_of_prob_indt += 1

    print "*******************************************************************"
    print "                RESULTS from classifier"
    print "*******************************************************************"
    if (is_match == 1):
        print "        No of matching entires for testing:", no_of_entries
        print "        EXPECTED 100% MATCH for digit:", digit_name_1
    else:
        print "        No of non-matching entires for testing:", no_of_entries
        print "        EXPECTED 100% NO-MATCH for digit:", digit_name_1
    print "*******************************************************************"
    print "*******************************************************************"
    print "GAUSSIAN-RESULTS,     MATCHING    :", no_of_bc_true_match
    print "GAUSSIAN-RESULTS,     NOT-MATCHING:", no_of_bc_false_match
    print "GAUSSIAN-RESULTS,     INDT        :", no_of_bc_indt
    print "*******************************************************************"
    print "*******************************************************************"
    print "PROB-RESULTS,     MATCHING    :", no_of_prob_true_match
    print "PROB-RESULTS,     NOT-MATCHING:", no_of_prob_false_match
    print "PROB-RESULTS,     INDT        :", no_of_prob_indt
    print "*******************************************************************"



pca_2d ,digit_1_cnt, mean_value, eigen_vector  = get_pca_of_digit(2,4)

# scatter plot
# for plotting;
no_r, no_c = np.shape(pca_2d)
x_axis = pca_2d[:,0]
y_axis = pca_2d[:,1]
digit_2_cnt= no_r -digit_1_cnt
plot_scatter(x_axis, y_axis, "TWO", "FOUR",digit_1_cnt, digit_2_cnt, False)

pca_for_digit_1= pca_2d[0:digit_1_cnt,:]
pca_for_digit_2= pca_2d[(digit_1_cnt+1):no_r,:]

# Bayes classificaton/Gaussian model
# pre-computation for Bayes classification/Gaussian model
d1_mean = np.mean(pca_for_digit_1, axis=0)
d1_covariance =  np.ma.cov(pca_for_digit_1, rowvar= False)
d1_n , d1_sample_cols = np.shape(pca_for_digit_1)
#print d1_covariance_array

d2_mean = np.mean(pca_for_digit_2, axis=0)
d2_covariance =  np.ma.cov(pca_for_digit_2, rowvar= False)
d2_n, d2_sample_cols = np.shape(pca_for_digit_2)

# Pre-computation for histogram
#find min and max
min_x = min(x_axis)
max_x = max(x_axis)
min_y = min(y_axis)
max_y = max(y_axis)
bin_size = 15


d1_inst =  class_2d_hist.histogram_2d(pca_for_digit_1, min_x, max_x, min_y, max_y, bin_size)
d1_2d_hist =  d1_inst.get_2d_histogram()


d2_inst =  class_2d_hist.histogram_2d(pca_for_digit_2, min_x, max_x, min_y, max_y, bin_size)
d2_2d_hist =  d2_inst.get_2d_histogram()




# TEST input
test_image_d1 , test_label_d1 =  load_feature_vector(2, 'testing')
find_classifier_result(test_image_d1, d1_inst, d2_inst,True,d1_mean, d1_covariance, d2_mean, d2_covariance, d1_n, d2_n,2 )
