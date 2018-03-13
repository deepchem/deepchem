import deepchem as dc
import tensorflow as tensorflow
from deepchem.models.tensorgraph.layers import Layer,Input,Dense,Hingeloss,Label,Feature
from deepchem.models.tensorgraph.layers import ReduceMean,InputFifoQueue,ReLU

class Scscore(TensorGraph):
	tg=dc.models.TensorGraph(tensorboard=True, model_dir='/tmp/scscore',use_queue=True)


	Input1=InputFifoQueue()

	hidden_1=Dense(out_channels=300,in_layer=Input1,activation_fn=tf.nn.relu)
	hidden_2=Dense(out_channels=300,in_layer=hidden_1,activation_fn=tf.nn.relu)
	hidden_3=Dense(out_channels=300,in_layer=hidden_2,activation_fn=tf.nn.relu)
	hidden_4=Dense(out_channels=300,in_layer=hidden_3,activation_fn=tf.nn.relu)
	hidden_5=Dense(out_channels=300,in_layer=hidden_4,activation_fn=tf.nn.relu)

	label=Label(shape=(300,))

	loss=Hingeloss(in_layers=[label,hidden_5])
	tg.set_loss(loss)
	output=Sigmoid(in_layers=[hidden_5])
	tg.add_output(output)