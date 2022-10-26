import os
import sys
import keras
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow.compat.v1 as tf
import numpy as np 
from utils import my_utils as ut
import sklearn
from sklearn.metrics import *
from matplotlib import pyplot as plt
import pandas as pd
import modules.output_util as ou
import modules.dataset_selector as ds
import modules.model_selector as ms
import modules.detectors_selector as des

ut.limit_gpu_usage()


def run_detectors(args):

	#LOAD OUTPUT df_out.csv TO GET SDV SAMPLE FILE LOCATION PARAMETERS___

	#TODO: parametri presi da args
	csv_dir = args[0]
	csv_name = csv_dir + "_output.csv"
	csv_path = "./out_dir/" + csv_dir + "/" + csv_name
	df = pd.read_csv(csv_path)
	dataset_name = df['dataset_name'][0]
	size = df['x_test_size'][0]
	model_name = df['model_name'][0]
	print(df)

	#LOAD DATASET: CIFAR-10______________________________________________

	x_test, y_test, x_train, y_train = ds.select_dataset(dataset_name, size)

	if dataset_name == 'cifar':
		x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
		y = tf.placeholder(tf.float32, shape=(None, 1))
	else:
		x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
		y = tf.placeholder(tf.float32, shape=(None, 1))
		print("mnist")

	#LOAD CLASSIFIER MODEL_______________________________________________

	classifier, model = ms.select_model(dataset_name, model_name)

	#LOAD AND TRAIN DETECTORS____________________________________________

	magnet_detector, squeezer = des.select_detectors(dataset_name, model, model_name)
	magnet_detector.train(x_train, y_train)
	squeezer.train(x_train, y_train)

	y_test_pred, y_pred_score = magnet_detector.test(x_test)
	mag_fpr = ut.calculate_accuracy_bool(y_test_pred)
	print("MagNet FPR calculated on the CIFAR-10 test set", mag_fpr)

	y_test_pred, y_test_pred_score = squeezer.test(x_test)
	squ_fpr = ut.calculate_accuracy_bool(y_test_pred)
	print("Squeezer FPR calculated on the CIFAR-10 test set", squ_fpr)

	#EXTRACT LAYERS: Remove comment to extract the model layers during prediction on the loaded attack set
	#inserire predict
	#output = ut.extract_layers(layers, x_att, "x_att_cifar", 500)

	rows = []

	for index, row in df.iterrows():

		if row['is_succ'] == True:
			
			x_adv = np.load("./out_dir/" + csv_dir + "/" + row['file_name'] + '_sample.npy')
			adv_idx = np.load("./out_dir/" + csv_dir + "/" + row['file_name'] + '_index.npy')
			y_adv = y_test[adv_idx]
			
			y_pred_adv, y_pred_score_adv = magnet_detector.test(x_adv)
			acc = ut.calculate_accuracy_bool(y_pred_adv)
			print("MagNet detection accuracy on the attack set", acc)
			add_row = pd.Series([acc], index=['magnet_det_rate'])
			row = row.append(add_row)

			y_att_pred, y_att_pred_score = squeezer.test(x_adv)
			acc = ut.calculate_accuracy_bool(y_att_pred)
			print("Squeezer detection accuracy on the attack set", acc)
			add_row = pd.Series([acc], index=['squeezer_det_rate'])
			row = row.append(add_row)

			rows.append(row)

	#for r in rows:
		#print(r)

	df_out = ou.create_df_detectors(rows)
	print(df_out)
	df_out.to_csv(csv_path[0:len(csv_path)-4] + '_detector.csv', index=False)


def main():
	
	args = sys.argv[1:]
	run_detectors(args)


if __name__=="__main__":
	
	#SET KERAS SESSION____________________________________________________________________________________________________________________________________________

	tf.disable_v2_behavior() 
	sess = ut.load_tf_session()
	keras.backend.set_learning_phase(0)
	
	main()