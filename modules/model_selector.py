import tensorflow.compat.v1 as tf
from art.estimators.classification import TensorFlowV2Classifier
from models import conv12layer as conv
from models.densenet_models import densenet_cifar10_model
from models.densenet_models import get_densenet_weights_path
from models.carlini_models import carlini_mnist_model
from models.cleverhans_models import cleverhans_mnist_model 


def select_cifar(model_name):

	if(model_name == "conv12"):
		model = conv.get_model()
		model.load_weights("./models/trained_models/CIFAR-10_ConvNet12.keras_weights.h5")
		optimizer = 'adam'
	else:
		if(model_name == "densenet"):
			model = densenet_cifar10_model(logits=True, input_range_type=1, pre_filter=lambda x:x)
			model.load_weights(get_densenet_weights_path())
			optimizer = 'sgd'
		else:
			print("\nNo model named " + model_name + " founded \n")
			exit()	

	model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
	print("Model selected: " + model_name + "\n")
	model.summary()

	classifier = TensorFlowV2Classifier(
    	model=model,
    	input_shape=(32,32,3),
    	loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.0,reduction="auto"),
    	clip_values=(0, 1),
    	nb_classes=10
	)

	return classifier, model


def select_mnist(model_name):
	
	if(model_name == "carlini"):
		model = carlini_mnist_model(logits=True, input_range_type=1, pre_filter=lambda x:x)
		model.load_weights("./models/trained_models/MNIST_carlini.keras_weights.h5")
	else:
		if(model_name == "clever"):
			model = cleverhans_mnist_model(logits=True, input_range_type=1, pre_filter=lambda x:x)
			model.load_weights("./models/trained_models/MNIST_cleverhans_retrain.keras_weights.h5")	
		else:
			print("\nNo model named " + model_name + " founded \n")
			exit()		

	model = model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
	print("Model selected: " + model_name + "\n")
	model.summary()

	classifier = TensorFlowV2Classifier(
	    model=model,
	    input_shape=(28,28,1),
	    loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.0,reduction="auto"),
	    clip_values=(0, 1),
	    nb_classes=10
	)

	return classifier, model


def select_model(dataset, model_name):

	if(dataset == "cifar"):	
		classifier, model = select_cifar(model_name)
	else:
		classifier, model = select_mnist(model_name)

	return classifier, model
