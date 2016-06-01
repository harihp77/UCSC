#!/usr/bin/env python

import pandas as pd
import numpy as np
import xlwt
import os

df = pd.read_excel("a4.xlsx")
df.insert(0,"DummyWeight", 1)
table = df.as_matrix()
rows, cols = np.shape(table)

trained_xa = table[:, 0:(cols-2)]
print "trained_xa shape", trained_xa.shape
trained_binary_classifier = table[:,cols-2:cols-1]
trained_multi_classifier_nominal = table[:, cols-1:cols]
trained_psedo_inv_xa = np.linalg.pinv(trained_xa)
print "trained_psedo_inv_xa shape", trained_psedo_inv_xa.shape

trained_multi_classifier_ordinal = np.zeros((rows,max(trained_multi_classifier_nominal)+1), dtype=int )
trained_multi_classifier_ordinal.fill(-1)
for i in range(0,rows):
    trained_multi_classifier_ordinal[i][trained_multi_classifier_nominal[i]] = 1

# Binary and Multi class weights
W_Binary =  np.matrix(trained_psedo_inv_xa) * np.matrix(trained_binary_classifier)
print "Binary weight shape", W_Binary.shape
#print W_Binary


W_Multi  =  np.matrix(trained_psedo_inv_xa) * np.matrix(trained_multi_classifier_ordinal)
print "Multiclass weight shape", W_Multi.shape
#print W_Multi


# Testing set
df2 = pd.read_excel("a4.xlsx",sheetname='To be classified')
df2.insert(0,"DummyWeight", 1)
test_row, test_col = df2.shape
test = df2.as_matrix()
testing_xa = test[3:,0:test_col-2]
print "Testing XA shape", testing_xa.shape
test_row, test_col = testing_xa.shape

# Binary classifier result
test_binary_result = np.sign(testing_xa * W_Binary)
#print test_binary_result
#print "Shape of binary result", test_binary_result.shape

#Multiclassifier result
test_multi_class_ordinal_result = (testing_xa * W_Multi)
#print test_multi_class_ordinal_result
#print "Shape of multi result", test_multi_class_ordinal_result.shape

test_multi_class_nominal_result = np.zeros((test_row, 1), dtype = int)
for i in range (0, test_row-1):
    test_multi_class_nominal_result[i] = test_multi_class_ordinal_result[i].argmax()
#print test_multi_class_nominal_result



# Validate Training set
train_validate_binary_classification = np.sign(trained_xa*W_Binary)
train_validate_multi_class_ordinal = (trained_xa*W_Multi)
validation_row, validation_col = train_validate_multi_class_ordinal.shape
multi_cls_validation_nominal= np.zeros((validation_row,1), dtype =int)
for i in range(0,validation_row-1):
    multi_cls_validation_nominal[i]= train_validate_multi_class_ordinal[i].argmax()
#print multi_cls_validation_nominal


# Calculate Performance of Linear Binary classifier
TN=TP=FN=FP=0
for  trained_binary_result, validated_binary_result  in zip(trained_binary_classifier,train_validate_binary_classification):
    if (trained_binary_result[0] == validated_binary_result):
        if trained_binary_result[0] == -1:
            TN += 1
        else:
            TP += 1
    else:
        if  validated_binary_result == -1:
            FN += 1
        else:
            FP += 1

binary_truth_tbl= np.zeros((2,2),dtype=int)
binary_truth_tbl[0][0] = TN
binary_truth_tbl[0][1] = FP
binary_truth_tbl[1][0] = FN
binary_truth_tbl[1][1] = TP

accuracy = ((TP + TN)/float(TP + FN +  FP +  TN))
sensitivity = (TP )/float(TP + FN )
specificity = (TN)/float(FP + TN)
ppv = (TP )/float(FP + TP)
binary_metrics = np.zeros((4,1),dtype=float)
binary_metrics [0][0]=accuracy
binary_metrics [1][0]=sensitivity
binary_metrics [2][0]=specificity
binary_metrics [3][0]=ppv


print "TP, TN, FP, FN" , TP, TN, FP, FN
print  "Accuracy of linear binary classifier:" ,((TP + TN)/float(TP + FN +  FP +  TN))
print  "Sensitivity of linear binary classifier:",  (TP )/float(TP + FN )
print  "Specificity of linear binary classifier:",  (TN)/float(FP + TN)
print  "Positive Predictive Value of linear binary classifier:",  (TP )/float(FP + TP)



# Calculate performance of linear mulit-class classifier

print "Shape of multi_cls_validation_nominal", multi_cls_validation_nominal.shape
print "Shape of trained_multi_classifier_nominal", trained_multi_classifier_nominal.shape
multi_class_validation_result = np.zeros((6,6),dtype=int)
print "Shape of multi_class_validation_result", multi_class_validation_result.shape
for  trained_multi_class_result, validated_multi_class_result in zip( trained_multi_classifier_nominal,multi_cls_validation_nominal):
    multi_class_validation_result[trained_multi_class_result[0]][validated_multi_class_result[0]] += 1

multi_cls_metrics = np.zeros((6,1),dtype=float)
row, col = multi_class_validation_result.shape
array_multi_class_validation_result = np.array(multi_class_validation_result)
for j in range(0,col):
    for i in range(0,row):
        multi_cls_metrics[j] = multi_cls_metrics[j] + array_multi_class_validation_result[i][j]
    multi_cls_metrics[j] = array_multi_class_validation_result[j][j]/multi_cls_metrics[j]






def write_to_exl_file(book, m, sheet_name):
    sheet1=book.add_sheet(sheet_name)
    a = np.array(m)
    row,col= np.shape(a)
    for i in range(row):
        for j in range (col):
            value = a[i][j]
            r = sheet1.row(i)
            r.write(j,value)


book = xlwt.Workbook()
write_to_exl_file(book, W_Binary, "W_Binary")
write_to_exl_file(book, W_Multi, "W_Multi")
write_to_exl_file(book, test_binary_result, "binary_cls_res")
write_to_exl_file(book, binary_truth_tbl, "binary_truth_tbl")
write_to_exl_file(book, binary_metrics, "binary_metrics")
write_to_exl_file(book, test_multi_class_nominal_result, "multi_cls_res")
write_to_exl_file(book, multi_class_validation_result, "multi_validation_res")
write_to_exl_file(book, multi_cls_metrics, "multi_cls_metrics")
book.save("output_file.xls")
