____INSTALLATION:____

To install the required packages and GPU support please load the Anaconda environment  "adv_gen":

	1) conda env create -f set_conda_environment.yml
	2) conda activate adv_gen
	3) conda env list 


____GENERATE ADVERSARIAL ATTACKS____

The main.py file generate adversarial attacks using the list and the parameters listed in the selected csv file in the "gen_param" folder.
(input_example.csv is given as example). 
The attacks are selected in the ART Toolbox (documentation v1.10.1, https://adversarial-robustness-toolbox.readthedocs.io/en/latest/).
The program exclude the legitimate sample that are missclassified originally by the model.   

- To execute attacks generations loop in main.py:
	
	python main.py -dataset -model -csv_name -img_size -log_all

	Parameters:

		- dataset: name of the dataset

			-cifar
			-mnist

		- model: name of the classifier model

			-carlini
			-clever
			-conv12
			-densenet

		- csv_name: name of the input csv in the gen_params folder
		
		- img_size: size of the starting legitimate sample set
		
		- log_all: save all the attacks sample if True otherwise only the adv_set with selected metrics (TODO leave True) 
			
- The program returns in the out_dir a directory with the same name of the input csv file. The folder contains all the adversarial set generated from the attacks listed in the input csv, the indexes that trace their position in the test set of the selected dataset and a sample adv image for each attack. 
Also an output csv is generated collectiong all the defined metrics for each dataset.  			


____EXERCISE DETECTORS____

The detectors_main.py file run the detectors on the generated attacks in the selected output_folder.
The csv output file in the folder is updated with the performance metrics of the exercised detectors
The available detector are:
	 
| Detector          | # Trainable Parameters                   | 
|-------------------|------------------------------------------|
| MagNet            | https://arxiv.org/abs/1705.09064                                         |
| Feature Squeezing | https://github.com/mzweilin/EvadeML-Zoo  |                                          |


- To exercise MagNet and Squeezer detectors on the generated adversarial samples:
	
	python detectors_main.py -out_folder_main
	
	Parameters:

		- out_folder_main: the name of the output folder with the output csv.   


____TARGET MODELS____

## Dataset: MNIST

| Model Name | # Trainable Parameters  | Testing Accuracy | Alias   |
|------------|-------------------------|------------------|---------|
| Carlini    |  312,202                |     0.9943       | MNIST-1 |
| Cleverhans |  710,218                |     0.9919       | MNIST-2 |

## Dataset: CIFAR-10

|      Model Name     |  # Trainable Parameters  | Testing Accuracy | Alias   |
|---------------------|--------------------------|------------------|---------|
| DenseNet(L=40,k=12) | 1,019,722                |     0.9484       | CIFAR-1 | 
| ConvNet12           | 2,919,082		 |     0.8514	    | CIFAR-2 |



