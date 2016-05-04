#!/usr/bin/env python

# Import MATH library
# Import EXCEL library
import os
import xlrd
import xlwt
import numpy as np
#import numpy.linalg.inv as inv
import math
#import matplotlib.pyplot as plt


# Global variables
no_of_rows = 0;
no_of_cols = 0;


def get_gender_count(xl_file_name):
    no_of_males = 0;
    no_of_females = 0;
    excel = xlrd.open_workbook(xl_file_name)
    for sheet in excel.sheets():
        for  row in range(1,sheet.nrows):
             gender = sheet.cell(row,0).value
             if  (gender == "'Male'"):
                 no_of_males = no_of_males + 1
             else:
                 no_of_females = no_of_females +1
    return (no_of_males, no_of_females)

def get_no_of_rows_cols(xl_file_name):
    excel = xlrd.open_workbook(xl_file_name)
    for sheet in excel.sheets():
        return ((sheet.nrows-1), (sheet.ncols-1))

def  populate_height(xl_file_name, is_sample,combined,male,female):
    book = xlrd.open_workbook(xl_file_name)
    for sheet in book.sheets():
        # Printing Colum header
        if (is_sample == 0):
            nrows = sheet.nrows
        else:
            nrows = is_sample
        mrow=0
        frow=0
        for  row in range(1,nrows):
            gender = sheet.cell(row,0).value
            height = sheet.cell(row,1).value
            handspan = sheet.cell(row,2).value
            combined[row-1]= (height, handspan)
            if (gender == "'Male'"):
                male[mrow]=(height, handspan)
                mrow += 1
            else:
                female[frow]=(height, handspan)
                frow += 1

def get_bin_index(item, bin_size, min_range, max_range):
    return (int (1 + ((bin_size -1)*((item - min_range)/(max_range-min_range)))))

def populate_2d_histogram(table_name, hist_name,  ht_min, ht_max, ha_min, ha_max, bin_size):
    for height,hand  in table_name:
        r = get_bin_index(height, bin_size, ht_min, ht_max)
        c = get_bin_index(hand, bin_size, ha_min, ha_max)
        hist_name[r-1, c-1] += 1


def compute_bayseian_classifier(dimention, x, y, mean_vector, co_variance_matrix):
    #print mean_vector
    two_pi_power_d_by_two = math.pow((2*math.pi),dimention/2)
    #print "Co-Variance"
    #print co_variance_matrix
    determinant_sigma_co_variance = np.linalg.det(co_variance_matrix)
    #print "determinant_sigma_co_variance"
    #print determinant_sigma_co_variance
    n_dim_sample_minus_mean_vector =  ((x,y)- mean_vector)
    #print "n_dim_sample_minus_mean_vector"
    #print n_dim_sample_minus_mean_vector
    inverse_sigma_co_variance = np.linalg.inv(co_variance_matrix)
    #print "inverse_sigma_co_variance"
    #print np.matrix(inverse_sigma_co_variance)
    # Thanks to senthil's code ref.. coverted to MATRIX
    transpose_n_dim_sample_minus_mean_vector = np.matrix(n_dim_sample_minus_mean_vector).transpose()
    #print "transpose_n_dim_sample_minus_mean_vector"
    #print transpose_n_dim_sample_minus_mean_vector
    compute_arg1 = (1/(two_pi_power_d_by_two * math.sqrt(determinant_sigma_co_variance)))
    return_value = (compute_arg1 * np.exp ((1/2) * np.matrix(n_dim_sample_minus_mean_vector) * inverse_sigma_co_variance * transpose_n_dim_sample_minus_mean_vector))
    return return_value

def find_probability_by_bayseian(height, handspan, m_mean_vector, m_covariance_array,f_mean_vector, f_covariance_array):
    print  "--------------------------------------------"
    print  "Finding probability through Bayes Classifier"
    print  "--------------------------------------------"
    bc_male = compute_bayseian_classifier(2, height, handspan, m_mean_vector, m_covariance_array)
    bc_female = compute_bayseian_classifier(2, height, handspan, f_mean_vector, f_covariance_array)
    print "BC-Male-Result:%f, BC-Female-Result:%f" %(bc_male,bc_female)
    if (bc_male > bc_female):
        print "Sample is most likely Male"
    elif(bc_female > bc_male):
        print "Sample is most likely Female"
    else:
        print "Sample is Indeterminant"


def  find_regular_probability(height, handspan, m_hist, f_hist, min_height, max_height, min_handspan, max_handspan, bin_size):
    print  "----------------------------------------------"
    print  "Finding probability through regular probability"
    print  "----------------------------------------------"
    height_bin = get_bin_index(height ,bin_size, min_height, max_height)
    handspan_bin = get_bin_index(handspan,bin_size, min_handspan, max_handspan)
    m_prob  = 0
    f_prob  = 0
    m_count = m_hist[height_bin][handspan_bin]
    f_count = m_hist[height_bin][handspan_bin]
    t_count = m_count + f_count
    if (m_count != 0):
        m_prob =  m_count/t_count

    if (f_count != 0):
        f_prob =  f_count/t_count

    print "Male-count:%f, Female-count:%f, Total-count:%d" %(m_count,f_count,t_count)
    if (m_prob > f_prob):
        print "The probability is MALE"
    elif (f_prob > m_prob):
        print "The probability is FEMALE"
    else:
        print "The probability can't be determined with given training set"





# Get required variable
(no_of_rows, no_of_cols) = get_no_of_rows_cols("Height_Handspan.xlsx")
(male_count, female_count) = get_gender_count("Height_Handspan.xlsx")

# Init 2D array and populate values
combined_2d_feature = np.zeros([no_of_rows, no_of_cols],dtype=float)
male_2d_feature = np.zeros([male_count, no_of_cols],dtype=float)
female_2d_feature = np.zeros([female_count, no_of_cols],dtype=float)
populate_height("Height_Handspan.xlsx",0,combined_2d_feature, male_2d_feature, female_2d_feature)

# Get MAX and MIN of height and handspan
min_height = min(combined_2d_feature[:,0])
max_height = max(combined_2d_feature[:,0])
min_handspan = min(combined_2d_feature[:,1])
max_handspan = max(combined_2d_feature[:,1])
print "Height Min:%d, Max:%d" %(min_height, max_height)
print "Handspan Min:%d, Max:%d" %(min_handspan, max_handspan)


# Set bin size
# log2 167 = 7.39
#bin_size = 7.39
bin_size = 15

# Init 2D histogram
male_2d_histogram = np.zeros([bin_size, bin_size],dtype=float)
female_2d_histogram = np.zeros([bin_size, bin_size],dtype=float)

# pouplate 2d histogram
populate_2d_histogram(male_2d_feature, male_2d_histogram, min_height, max_height, min_handspan, max_handspan,bin_size)
populate_2d_histogram(female_2d_feature, female_2d_histogram, min_height, max_height, min_handspan, max_handspan,bin_size)
print "*****************************************************************************"
print "                             MALE  3D  HISTOGRAM                             "
print "*****************************************************************************"
print male_2d_histogram

print "*****************************************************************************"
print "                             FEMALE  3D  HISTOGRAM                             "
print "*****************************************************************************"
print female_2d_histogram



# Calculate Male MEAN, VARIANCE etc
m_mean_vector =  np.mean(male_2d_feature, axis=0)
m_height_arr = male_2d_feature[:,0]
m_hand_arr = male_2d_feature[:,1]
m_covariance_array  =  np.cov(m_height_arr, m_hand_arr)
print "Male Mean vector"
print m_mean_vector
print "Male covariance matrix"
print m_covariance_array

# co-variance matrix doesnt work on the 2D matrix
#m_covariance_matrix =  np.cov(male_2d_feature)


# Calculate female MEAN, VARIANCE etc
f_mean_vector =  np.mean(female_2d_feature, axis=0)
f_height_arr = female_2d_feature[:,0]
f_hand_arr = female_2d_feature[:,1]
f_covariance_array =  np.cov(f_height_arr, f_hand_arr)
print "Female Mean vector"
print f_mean_vector
print "Female covariance matrix"
print f_covariance_array
# co-variance matrix doesnt work on the 2D feature set.
#f_covariance_matrix =  np.cov(female_2d_feature)

sample_data = [[69,17.5],
               [66,22],
               [70,21.5],
               [69,23.5]]



for  height,hand_span in sample_data:
    print" =============================================================================="
    print "              Sample for HEIGHT:%f, HAND_SPAN:%f" %(height, hand_span)
    print" =============================================================================="
    find_probability_by_bayseian(height, hand_span,m_mean_vector, m_covariance_array,f_mean_vector, f_covariance_array)
    print ""
    print ""
    find_regular_probability(height, hand_span, male_2d_histogram, female_2d_histogram, min_height, max_height, min_handspan, max_handspan, bin_size)
    print ""
    print ""
