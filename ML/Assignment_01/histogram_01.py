#!/usr/bin/env python

# Import MATH library
# Import EXCEL library
import os
import xlrd
import xlwt
import numpy as np
import math

#book = xlrd.open_workbook("Height_Data.xlsx")
# List for maintaining full table
# Combined Height, Gender table
male_height = []
female_height = []
combined_height = []



# Method to populate the above tables, given EXCEL is the input file
def  populate_height(xl_file_name, is_sample):
    book = xlrd.open_workbook(xl_file_name)
    for sheet in book.sheets():
        print "N of cols", sheet.ncols
        print "N of Rows", sheet.nrows

        # Printing Colum header
        col_names = sheet.row(0)
        print col_names[0].value,  col_names[1].value, col_names[2].value
        if (is_sample == 0):
            nrows = sheet.nrows
        else:
            nrows = is_sample

        for  row in range(1,nrows):
            feets = sheet.cell(row,0)
            inches = sheet.cell(row,1)
            gender = sheet.cell(row,2)
            height = (feets.value*12) + inches.value
            combined_height.append(height)
            if  (gender.value == "Male"):
                male_height.append(height)
            else:
                female_height.append(height)

def dump_table(name):
    for item in name:
        print item


#Method to plot histogram
def populate_histogram(table_name, hist_name,  min_range, max_range, max_bin):
    #Init the histogram
    for index in range(0,(max_bin)):
        #print index
        hist_name.insert(index,0)

    for item in table_name:
        bin_index = int(1 + ((max_bin-1)*((item-min_range)/(max_range-min_range))))
        hist_name[bin_index-1] =  hist_name[bin_index-1] +1


#Method for Bayesian Classifier
def compute_bayseian_classifier(no_of_sample, value, mean, std_dev):
    return (no_of_sample *((1/(math.sqrt(2*math.pi)*std_dev))*math.exp(-1/2*(((value-mean)/std_dev)**2))))







# MAIN program starts here...

# If is_sampled is ZERO/FALSE, then full data in the file
is_sampled = 200
populate_height("Height_Data.xlsx",is_sampled)


#debug dump!!
#dump_table(female_height)

# Calculate BIN
# Log2(16700) = 14; + 1
bins = 15

#calculate MIN and MAX from total_height table
max_height =  max(combined_height)
min_height = min(combined_height)
print "Combined height table-MIN :%d and MAX :%d " %(min_height,max_height)
print "Bin range is from 1 to %d" %((max_height - min_height)+1)
max_bins = int((max_height - min_height)+1)

male_hist = []
female_hist = []

####################################################
# Populate Male histogram
####################################################
populate_histogram(male_height, male_hist, min_height, max_height,max_bins)
#dump_table(male_hist)

####################################################
# Populate female histogram
####################################################
populate_histogram(female_height, female_hist, min_height, max_height,max_bins)
dump_table(female_hist)


####################################################
# Calculate MEAN, STD DEVIATION
####################################################
m_np = np.array(male_height)
f_np = np.array(female_height)
m_mean = np.mean(m_np)
f_mean = np.mean(f_np)
m_std_dev = np.std(m_np)
f_std_dev = np.std(f_np)

print  "Mean of MALE:%d" %(m_mean)
print  "STD DEV of MALE:%d" %(m_std_dev)
print  "Mean of FEMALE:%d" %(f_mean)
print  "STD DEV of FEMALE:%d" %(f_std_dev)



####################################################
# Predict from the data set
####################################################

data_list= [55,60,65,70,75,80]

for item in  data_list:
    m_bay_c = compute_bayseian_classifier(len(male_height),item,m_mean, m_std_dev)
    f_bay_c = compute_bayseian_classifier(len(female_height),item,f_mean, f_std_dev)


    if (m_bay_c > f_bay_c):
        print  "Prediction by Bayseian classifier for item:%d is MALE" %(item)
    elif (f_bay_c > m_bay_c):
        print  "Prediction by Bayseian classifier for item:%d is FEMALE" %(item)
    else:
        print  "Prediction by Bayseian classifier for item:%d is MALE/FEMALE" %(item)








