import os
import sys
import keras
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


def run_extract(args):

	#LOAD OUTPUT df_out.csv TO GET SDV SAMPLE FILE LOCATION PARAMETERS___

	#TODO: parametri presi da args
	csv_dir = args[0]
	csv_name = args[1] + "_output.csv"
	csv_path = "./out_dir/" + csv_dir + "/" + csv_name
	df = pd.read_csv(csv_path)
	dataset_name = df['dataset_name'][0]
	size = df['x_test_size'][0]
	model_name = df['model_name'][0]
	extract = args[2]
	print(df)

	#LOAD DATASET: CIFAR-10______________________________________________

	x_test, y_test, x_train, y_train = ds.select_dataset(dataset_name, 1000)
	
	#LOAD CLASSIFIER MODEL_______________________________________________

	classifier, model = ms.select_model(dataset_name, model_name)
	layers = model.layers
	
	y_pred = np.argmax(classifier.predict(x_test), axis=1)
	x_test_acc = accuracy_score(y_pred, y_test)
	print("Model accuracy on normal samples: " + str(x_test_acc) + "\n")

	if extract == 'normal':
		ut.extract_layers(layers, x_test, "x_test", "test_folder", 500)
		exit()

	rows = []

	for index, row in df.iterrows():

		if row['is_succ'] == True:

			x_adv = np.load("./out_dir/" + csv_dir + "/" + row['file_name'] + '_sample.npy')
			adv_idx = np.load("./out_dir/" + csv_dir + "/" + row['file_name'] + '_index.npy')
			y_adv = y_test[adv_idx]

			ut.extract_layers(layers, x_adv, row['file_name'], csv_dir, 500)

			#TODO: se si vuole stampare csv di output con qualche informazione 
			#add_row = pd.Series([acc], index=['squeezer_det_rate'])
			#row = row.append(add_row)
			#rows.append(row)

	#df_out = ou.create_df_detectors(rows)
	#print(df_out)
	#df_out.to_csv(csv_path[0:len(csv_path)-4] + '_detector.csv', index=False)


def main():
	
	args = sys.argv[1:]
	run_extract(args)


if __name__=="__main__":

	main()