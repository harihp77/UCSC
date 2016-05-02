#!/usr/bin/env python

# Import MATH library
# Import EXCEL library
import os
import xlrd
import xlwt
import numpy as np
import math

# Combined Height, Gender table
male_height = []
female_height = []
combined_height = []

male_handspan = []
female_handspan = []
combined_handspan = []



# Method to populate the above tables, given EXCEL is the input file
def  populate_height(xl_file_name, is_sample):
    book = xlrd.open_workbook(xl_file_name)
    for sheet in book.sheets():

        # Printing Colum header
        col_names = sheet.row(0)
        print col_names[0].value,  col_names[1].value, col_names[2].value
        print "N of cols", sheet.ncols
        print "N of Rows", sheet.nrows
        if (is_sample == 0):
            nrows = sheet.nrows
        else:
            nrows = is_sample

        for  row in range(1,nrows):
            gender = sheet.cell(row,0).value
            height = sheet.cell(row,1).value
            handspan = sheet.cell(row,2).value
            combined_height.append(height)
            combined_handspan.append(handspan)
            if  (gender == "Male"):
                male_height.append(height)
                male_handspan.append(handspan)
            else:
                female_height.append(height)
                female_handspan.append(handspan)

def dump_table(name):
    for item in name:
        print item


#Method to plot histogram
#def populate_histogram(table_name, hist_name,  feature_index, min_range, max_range, max_bin):
    #Init the histogram
    #for index in range(0,(max_bin)):
        ##print index
        #hist_name.insert(index,0)

    #for item in table_name:
        #bin_index = int(1 + ((max_bin-1)*((item-min_range)/(max_range-min_range))))
        #hist_name[bin_index-1] =  hist_name[bin_index-1] +1


#Method for Bayesian Classifier
#def compute_bayseian_classifier(no_of_sample, value, mean, std_dev):
    #return (no_of_sample *((1/(math.sqrt(2*math.pi)*std_dev))*math.exp(-1/2*(((value-mean)/std_dev)**2))))







# MAIN program starts here...
# Create 2D feature vector
excel = xlrd.open_workbook("Height_Handspan.xlsx")
for sheet in excel.sheets():
    no_of_males = 0
    no_of_females = 0
    for  row in range(1,sheet.nrows):
        gender = sheet.cell(row,0).value
        print gender
        if  (gender == "Male"):
            no_of_males += 1
        else:
            no_of_females += 1

    combined_2d_feature = np.zeros([sheet.nrows, sheet.ncols],dtype=float)
    print  "Males:%d, Col:%d"%(no_of_females, sheet.ncols)
    male_2d_feature = np.zeros([no_of_males, sheet.ncols],dtype=float)
    female_2d_feature = np.zeros([no_of_females, sheet.ncols],dtype=float)
print male_2d_feature



# If is_sampled is ZERO/FALSE, then full data in the file
is_sampled = 0
populate_height("Height_Handspan.xlsx",is_sampled)

#debug dump!!
#dump_table(female_height)

# Calculate BIN
# Log2(16700) = 14; + 1
max_bins = 15

#calculate MIN and MAX from total_height table
max_height =  max(combined_height)
min_height = min(combined_height)

max_handspan = max(combined_handspan)
min_handspan = min(combined_handspan)

print "Combined handspan table-MIN :%f and MAX :%f " %(min_handspan,max_handspan)
print "Combined height table-MIN :%f and MAX :%f " %(min_height,max_height)

#declare a 2-D array
hist_2d = np.zeros([max_bins,max_bins],dtype=float)
#print hist_2d



