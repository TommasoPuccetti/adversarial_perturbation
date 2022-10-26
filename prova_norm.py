import numpy as np
from data.setup_cifar import CIFAR
import modules.metrics_evaluator as me
from sewar import *
import sys

data = CIFAR()
x_test = data.test_data
x_test = x_test.astype('float32')
y_test = np.argmax(data.test_labels, axis=1)

succ_idx = np.load("./out_dir/gen_list_jsma_debug/elastic100.5_succ_index.npy")
x_succ_adv = np.load("./out_dir/gen_list_jsma_debug/elastic100.5_succ_sample.npy")

#x_adv_lin = np.ndarray.flatten(x_succ_adv[0])
#x_test_lin = np.ndarray.flatten(x_test[succ_idx[0]])

#uqi(x_test_lin, x_adv_lin)
#qu = msssim(x_test[succ_idx[0]], x_succ_adv[0], MAX=1)

arr = [1,2,3,4,np.NaN,5,6]

linearize = x_test[succ_idx[54]].flatten()
print(linearize)
#print(np.max(x_test[succ_idx[54]]))
#print(np.min(x_test[succ_idx[54]]))
#print(np.isnan(x_test[succ_idx[54]]).any())
print(np.isnan(linearize).any())

erg = [] 

for i in range(0,len(x_succ_adv)):
	qu = ergas(x_test[succ_idx[i]], x_succ_adv[i])
	if(i!=2):
		erg.append(qu)
		print(i)
		print(qu)
	
print(np.mean(erg))


