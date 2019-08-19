
import os
import sys
import csv
import copy
import math
import random
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,auc,average_precision_score
from getdata import get_testval,get_train
from dataprocess import convertRawToXY
from model import OnehotNetwork,OtherNetwork,PhysicochemicalNetwork,HydrophobicityNetwork,CompositionNetwork,BetapropensityNetwork,AlphaturnpropensityNetwork
import json


gpu_id = '5'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')


def calculate_performance(test_num,labels,predict_y,predict_score):
	tp=0
	fp=0
	tn=0
	fn=0
	for index in range(test_num):
		if(labels[index]==1):
			if(labels[index] == predict_y[index]):
				tp += 1
			else:
				fn += 1
		else:
			if(labels[index] == predict_y[index]):
				tn += 1
			else:
				fp += 1
	acc = float(tp+tn)/test_num
	precision = float(tp)/(tp+fp+ sys.float_info.epsilon)
	sensitivity = float(tp)/(tp+fn+ sys.float_info.epsilon)
	specificity = float(tn)/(tn+fp+ sys.float_info.epsilon)
	f1 = 2 * precision * sensitivity / (precision + sensitivity + sys.float_info.epsilon)
	mcc = float(tp*tn-fp*fn)/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + sys.float_info.epsilon)
	aps = average_precision_score(labels,predict_score)
	fpr,tpr,_ = roc_curve(labels,predict_score)
	aucResults = auc(fpr,tpr)
    
	strResults = 'tp '+ str(tp) + ' fn ' + str(fn) + ' tn ' + str(tn) + ' fp ' + str(fp)
	strResults = strResults + ' acc ' + str(acc) + ' precision ' + str(precision) + ' sensitivity ' + str(sensitivity)
	strResults = strResults + ' specificity ' + str(specificity) + ' f1 ' + str(f1) + ' mcc ' + str(mcc)
	strResults = strResults + ' aps ' + str(aps) + ' auc ' + str(aucResults)
    
	return strResults


def normalization_softmax(array):

	normal_array = []
	de = 0
	for i in array:
		de += math.exp(i)
	for i in array:
		normal_array.append(math.exp(i)/de)

	return normal_array


def NewshufflePosNeg(data2):

	data2_over=[]
	index = [i for i in range(len(data2))]
	random.shuffle(index)
	
	data2_over = data2.as_matrix()[index]
	data2_over = pd.DataFrame(data2_over)

	return data2_over


def Newshufflewrr(data1_pos,data1_neg):
	##### Create an index with nummber of posnum #####
	index = [i for i in range(len(data1_pos))] 
	random.shuffle(index)
	data1_pos = pd.DataFrame(data1_pos)
	data1_pos = data1_pos.as_matrix()[index]
	data1_pos_ss = pd.DataFrame(data1_pos)

	index = [i for i in range(len(data1_neg))]
	random.shuffle(index)
	data1_neg = pd.DataFrame(data1_neg)
	data1_neg = data1_neg.as_matrix()[index]
	data1_neg_ss = pd.DataFrame(data1_neg)

	return data1_pos_ss, data1_neg_ss


def get_matrix(windows_pos,windows_neg):
	windows_pos = pd.DataFrame(windows_pos)
	windows_neg = pd.DataFrame(windows_neg)	
	windows_all = pd.concat([windows_pos,windows_neg])
	windows_all = windows_all.as_matrix()	
	del windows_pos,windows_neg
	return windows_all


all_train_windows_pos,all_train_windows_neg = get_train(r'data/1train.fasta')
val_windows_pos, val_windows_neg = get_testval(r'data/1val.fasta')
test_windows_pos,test_windows_neg = get_testval(r'data/1test.fasta')
test_windows_all = get_matrix(test_windows_pos,test_windows_neg)

test_oneofkeyX,testY = convertRawToXY(test_windows_all,codingMode=0)
test_oneofkeyX.shape = (test_oneofkeyX.shape[0],test_oneofkeyX.shape[2],test_oneofkeyX.shape[3])
test_physicalXo,_ = convertRawToXY(test_windows_all,codingMode=9)
test_physicalXo.shape = (test_physicalXo.shape[0],test_physicalXo.shape[2],test_physicalXo.shape[3])
test_physicalXp,_ = convertRawToXY(test_windows_all,codingMode=10)
test_physicalXp.shape = (test_physicalXp.shape[0],test_physicalXp.shape[2],test_physicalXp.shape[3])
test_physicalXh,_ = convertRawToXY(test_windows_all,codingMode=11)
test_physicalXh.shape = (test_physicalXh.shape[0],test_physicalXh.shape[2],test_physicalXh.shape[3])
test_physicalXc,_ = convertRawToXY(test_windows_all,codingMode=12)
test_physicalXc.shape = (test_physicalXc.shape[0],test_physicalXc.shape[2],test_physicalXc.shape[3])
test_physicalXb,_ = convertRawToXY(test_windows_all,codingMode=13)
test_physicalXb.shape = (test_physicalXb.shape[0],test_physicalXb.shape[2],test_physicalXb.shape[3])
test_physicalXa,_ = convertRawToXY(test_windows_all,codingMode=14)
test_physicalXa.shape = (test_physicalXa.shape[0],test_physicalXa.shape[2],test_physicalXa.shape[3])
del test_windows_pos,test_windows_neg,test_windows_all
print("Test data coding finished!")

val_windows_all = get_matrix(val_windows_pos,val_windows_neg)

val_oneofkeyX,valY = convertRawToXY(val_windows_all,codingMode=0)
val_oneofkeyX.shape = (val_oneofkeyX.shape[0],val_oneofkeyX.shape[2],val_oneofkeyX.shape[3])
val_physicalXo,_ = convertRawToXY(val_windows_all,codingMode=9)
val_physicalXo.shape = (val_physicalXo.shape[0],val_physicalXo.shape[2],val_physicalXo.shape[3])
val_physicalXp,_ = convertRawToXY(val_windows_all,codingMode=10)
val_physicalXp.shape = (val_physicalXp.shape[0],val_physicalXp.shape[2],val_physicalXp.shape[3])
val_physicalXh,_ = convertRawToXY(val_windows_all,codingMode=11)
val_physicalXh.shape = (val_physicalXh.shape[0],val_physicalXh.shape[2],val_physicalXh.shape[3])
val_physicalXc,_ = convertRawToXY(val_windows_all,codingMode=12)
val_physicalXc.shape = (val_physicalXc.shape[0],val_physicalXc.shape[2],val_physicalXc.shape[3])
val_physicalXb,_ = convertRawToXY(val_windows_all,codingMode=13)
val_physicalXb.shape = (val_physicalXb.shape[0],val_physicalXb.shape[2],val_physicalXb.shape[3])
val_physicalXa,_ = convertRawToXY(val_windows_all,codingMode=14)
val_physicalXa.shape = (val_physicalXa.shape[0],val_physicalXa.shape[2],val_physicalXa.shape[3])
del val_windows_pos,val_windows_neg,val_windows_all
print("Val data coding finished!")

iteration_times = 14
for t in range(0,iteration_times):
	print("iteration_times: %d"%t)
	train_windows_pos, train_windows_neg = Newshufflewrr(all_train_windows_pos, all_train_windows_neg)
	ff = int(len(train_windows_pos))  #Fractional factor
	train_windows_pos = train_windows_pos[0:ff]
	train_windows_neg = train_windows_neg[0:ff]

	train_windows_all = pd.concat([train_windows_pos,train_windows_neg])
	train_windows_all = NewshufflePosNeg(train_windows_all)
	train_windows_all = pd.DataFrame(train_windows_all)
	matrix_train_windows_all = train_windows_all.as_matrix()

	train_oneofkeyX,trainY = convertRawToXY(matrix_train_windows_all,codingMode=0)
	train_oneofkeyX.shape = (train_oneofkeyX.shape[0],train_oneofkeyX.shape[2],train_oneofkeyX.shape[3]) 
	train_physicalXo,_ = convertRawToXY(matrix_train_windows_all,codingMode=9)
	train_physicalXo.shape = (train_physicalXo.shape[0],train_physicalXo.shape[2],train_physicalXo.shape[3]) 
	train_physicalXp,_ = convertRawToXY(matrix_train_windows_all,codingMode=10)
	train_physicalXp.shape = (train_physicalXp.shape[0],train_physicalXp.shape[2],train_physicalXp.shape[3])
	train_physicalXh,_ = convertRawToXY(matrix_train_windows_all,codingMode=11)
	train_physicalXh.shape = (train_physicalXh.shape[0],train_physicalXh.shape[2],train_physicalXh.shape[3]) 	
	train_physicalXc,_ = convertRawToXY(matrix_train_windows_all,codingMode=12)
	train_physicalXc.shape = (train_physicalXc.shape[0],train_physicalXc.shape[2],train_physicalXc.shape[3]) 	
	train_physicalXb,_ = convertRawToXY(matrix_train_windows_all,codingMode=13)
	train_physicalXb.shape = (train_physicalXb.shape[0],train_physicalXb.shape[2],train_physicalXb.shape[3])
	train_physicalXa,_ = convertRawToXY(matrix_train_windows_all,codingMode=14)
	train_physicalXa.shape = (train_physicalXa.shape[0],train_physicalXa.shape[2],train_physicalXa.shape[3])
	print("itreation %d times Train data coding finished!" %t)

	if(t==0):
		struct_Onehot_model = OnehotNetwork(train_oneofkeyX,trainY,val_oneofkeyX,valY,train_time=t)
		physical_O_model = OtherNetwork(train_physicalXo,trainY,val_physicalXo,valY,train_time=t)
		physical_P_model = PhysicochemicalNetwork(train_physicalXp,trainY,val_physicalXp,valY,train_time=t)
		physical_H_model = HydrophobicityNetwork(train_physicalXh,trainY,val_physicalXh,valY,train_time=t)
		physical_C_model = CompositionNetwork(train_physicalXc,trainY,val_physicalXc,valY,train_time=t)
		physical_B_model = BetapropensityNetwork(train_physicalXb,trainY,val_physicalXb,valY,train_time=t)
		physical_A_model = AlphaturnpropensityNetwork(train_physicalXa,trainY,val_physicalXa,valY,train_time=t)
		print("itreation %d times training finished!" %t)
	else:
		struct_Onehot_model = OnehotNetwork(train_oneofkeyX,trainY,val_oneofkeyX,valY,train_time=t,compilemodels=struct_Onehot_model)
		physical_O_model = OtherNetwork(train_physicalXo,trainY,val_physicalXo,valY,train_time=t,compilemodels=physical_O_model)
		physical_P_model = PhysicochemicalNetwork(train_physicalXp,trainY,val_physicalXp,valY,train_time=t,compilemodels=physical_P_model)
		physical_H_model = HydrophobicityNetwork(train_physicalXh,trainY,val_physicalXh,valY,train_time=t,compilemodels=physical_H_model)
		physical_C_model = CompositionNetwork(train_physicalXc,trainY,val_physicalXc,valY,train_time=t,compilemodels=physical_C_model)
		physical_B_model = BetapropensityNetwork(train_physicalXb,trainY,val_physicalXb,valY,train_time=t,compilemodels=physical_B_model)
		physical_A_model = AlphaturnpropensityNetwork(train_physicalXa,trainY,val_physicalXa,valY,train_time=t,compilemodels=physical_A_model)
		print("itreation %d times training finished!" %t)

	monitor = 'val_loss'
	weights = []
	with open ('model/loss/'+str(t)+'onehotloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Onetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Pnetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Hnetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Cnetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Bnetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))
	with open ('model/loss/'+str(t)+'Anetloss.json', 'r') as checkpoint_fp:
		weights.append(1/float(json.load(checkpoint_fp)[monitor]))

	weight_array = np.array(weights, dtype= np.float)
	del weights

	#normalize checkpoit data as weights
	weight_array = normalization_softmax(weight_array)
	
	predict_weighted_merge = 0
	#load model weights and checkpoint file
	predict_temp = weight_array[0] * struct_Onehot_model.predict(test_oneofkeyX)
	predict_weighted_merge += predict_temp
	predict_temp = weight_array[1] * physical_O_model.predict(test_physicalXo)
	predict_weighted_merge += predict_temp
	predict_temp = weight_array[2] * physical_P_model.predict(test_physicalXp)
	predict_weighted_merge += predict_temp
	predict_temp = weight_array[3] * physical_H_model.predict(test_physicalXh)
	predict_weighted_merge += predict_temp
	predict_temp = weight_array[4] * physical_C_model.predict(test_physicalXc)
	predict_weighted_merge += predict_temp
	predict_temp = weight_array[5] * physical_B_model.predict(test_physicalXb)
	predict_weighted_merge += predict_temp
	predict_temp = weight_array[6] * physical_A_model.predict(test_physicalXa)
	predict_weighted_merge += predict_temp
	
	predict_classes = copy.deepcopy(predict_weighted_merge[:,1])

	for n in range(len(predict_classes)):
		if predict_classes[n] >= 0.5:
			predict_classes[n] = 1
		else:
			predict_classes[n] = 0

	with open('result/evaluation.txt', mode='a') as resFile:
		resFile.write(str(t)+" "+calculate_performance(len(testY),testY[:,1],predict_classes,predict_weighted_merge[:,1])+'\r\n')
	resFile.close()
	true_label = testY
	result = np.column_stack((true_label[:,1],predict_weighted_merge[:,1]))
	result = pd.DataFrame(result)
	result.to_csv(path_or_buf='result/result'+'-'+str(t)+'.txt',index=False,header=None,sep='\t',quoting=csv.QUOTE_NONNUMERIC)
    