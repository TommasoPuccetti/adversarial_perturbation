import pandas as pd 
import numpy as np
from data.setup_cifar import CIFAR
import modules.output_util as ou


path = "./out_dir/compose_2/compose_2_output_detector.csv"
df = pd.read_csv(path)

data = CIFAR()
x_test = data.test_data
y_test = np.argmax(data.test_labels, axis=1)
x_train = data.train_data
y_train = np.argmax(data.train_labels, axis=1)

rows = []

print(df)

for index, row in df.iterrows():
		
	x_adv = np.load("./out_dir/" + row['csv_name'] + "/" + row['file_name'] + '_sample.npy')
	adv_idx = np.load("./out_dir/" + row['csv_name'] + "/" + row['file_name'] + '_index.npy')

	l0 = 0
	x_test_succ = x_test[adv_idx]

	for i, x in enumerate(x_adv):

		x_test_lin = np.ndarray.flatten(x_test_succ[i])
		x_adv_lin = np.ndarray.flatten(x_adv[i])
	
		l0 = l0 + np.linalg.norm(x_test_lin - x_adv_lin, ord=0)	
	
	add_row = pd.Series([l0], index=['avg_l0'])
	row = row.append(add_row)
	
	rows.append(row)

df_out = ou.create_df_detectors(rows)
print(df_out)
df_out.to_csv("./out_dir/compose_2" + '_detector_add.csv', index=False)