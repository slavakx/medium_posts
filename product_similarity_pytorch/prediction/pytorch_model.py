import torch
from torch.functional import norm
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
import torch.optim as optim

import pytorch_lightning as pl

from typing import List, Tuple, Dict, Union


import torchmetrics as tm

from IPython.core.debugger import set_trace

def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low

class SigmoidRange(nn.Module):
    "Sigmoid module with range `(low, high)`"

    def __init__(self, low, high):
        super(SigmoidRange, self).__init__()

        self.low = low
        self.high = high

    def forward(self, x):
        return sigmoid_range(x, self.low, self.high)



#taken from fastai: LinBnDrop
#extended with other types of normalization (NormLayer) and order of normalization vs activation
class LinearNormDropActivation(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, normalization:str = "BatchNorm1d", dropout=0., activation=None, normalization_before_activation = True):
        norm = getattr(nn, normalization) if normalization is not None else None
        is_batch_norm = norm == nn.BatchNorm1d
        
        #if batch norm don't use bias -> check if this should be always the case
        
        is_bias = True
        if is_batch_norm:
            if normalization_before_activation:
                is_bias = False
        
        layers = [nn.Linear(n_in, n_out, bias= is_bias)]


        if normalization_before_activation:
            if norm is not None:
                layers.append(norm(n_out))
            if activation is not None: layers.append(activation)
        else:
            if activation is not None: layers.append(activation)
            if norm is not None:
                layers.append(norm(n_out))
        if dropout != 0: layers.append(nn.Dropout(dropout)) 

        super().__init__(*layers)


#from Inside Deep Learning book
class SkipDenseConnection(nn.Module):
    def __init__(self, n_layers:int, n_in:int, n_out:int, normalization:str = "BatchNorm1d", dropout:float = 0.0, activation = nn.ReLU()):
        """
        n_layers: how many hidden layers for this block of dense skip connections
        n_in: how many features are coming into this layer
        n_out: how many features should be used for the final layer of this block.  
        normalization: normalization to use can be None, BatchNorm1d, LayerNorm
        dropout: default 0, dropout rate
        activation: initialized activation
        """
        super().__init__()
        
        self.norm = getattr(nn, normalization) if normalization is not None else None
        self.activation = activation

        #The final layer will be treated differently, so lets grab it's index to use in the next two lines
        n_internal_layers = n_layers-1
        #The linear and batch norm layers are stored in `layers` and `bns` respectively. A list comprehensive creates all the layers in one line each. The `if i == l` allows us to single out the last layer which needs to use `out_size` instead of `in_size`
        #self.layers = nn.ModuleList([nn.Linear(n_in * n_internal_layers, n_out) if i == l else nn.Linear(n_in, n_out) for i in range(n_layers)])
        #self.bns = nn.ModuleList([nn.BatchNorm1d(n_out) if i == l else nn.BatchNorm1d(n_in) for i in range(n_layers)])
        #Since we are writing our own `forward` function instead of using nn.Sequential, we can just use one activation object multiple times. 
        #self.activation = nn.LeakyReLU(leak_rate)

        #the last layer has the size of n_in * n_internal_layers because all previous outputs a concatenated and inputed into the last layer
        self.skip_layers = nn.ModuleList([
            LinearNormDropActivation(n_in * n_internal_layers, n_out, normalization=normalization, dropout=dropout, activation = activation, normalization_before_activation=True) 
            if i == n_internal_layers 
            else 
            LinearNormDropActivation(n_in, n_in, normalization=normalization, dropout=dropout, activation = activation, normalization_before_activation=True) for i in range(n_layers)
        ])
    
    def forward(self, x):
        #First we need a location to store the activations from every layer (except the last one) in this block. 
        # All the activations will be combined as the input to the last layer, which is what makes the skips! 
        #activations = []
        
        #zip the linear and normalization layers into paired tuples, using [:-1] to select all but the last item in each list.
        #for layer, bn in zip(self.layers[:-1], self.bns[:-1]):
        #    x = self.activation(bn(layer(x)))
        #    activations.append( x )
        #concatenate the activations together, this makes the input for the last layer
        #x = torch.cat(activations, dim=1)
        #Now manually use the last linear and batch-norm layer on this concatenated input, giving us the result.
        #return self.activation(self.bns[-1](self.layers[-1](x)))

        x_outputs = []
        for layer in self.skip_layers[:-1]:
            x = layer(x)
            x_outputs.append(x)

        x = torch.cat(x_outputs, dim = 1)

        last_layer  = self.skip_layers[-1]
        x = last_layer(x)

        return x



class LastTimeStep(nn.Module):
    """
    A class for extracting the hidden activations of the last time step following
    the output of a PyTorch RNN module.
    """
    def __init__(self, rnn_layers=1, bidirectional=False):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        if bidirectional:
            self.num_driections = 2
        else:
            self.num_driections = 1

    def forward(self, input):
        #Result is either a tuple (out, h_t)
        #or a tuple (out, (h_t, c_t))
        rnn_output = input[0]
        last_step = input[1] #this will be h_t
        if(type(last_step) == tuple):#unless it's a tuple,
            last_step = last_step[0]#then h_t is the first item in the tuple
        
        batch_size = last_step.shape[1] #per docs, shape is: '(num_layers * num_directions, batch, hidden_size)'
        #print(last_step.shape)

        #reshaping so that everything is separate
        last_step = last_step.view(self.rnn_layers, self.num_driections, batch_size, -1)
        
        #We want the last layer's results
        last_step = last_step[self.rnn_layers-1]
        #Re order so batch comes first
        last_step = last_step.permute(1, 0, 2)
        #Finally, flatten the last two dimensions into one
        return last_step.reshape(batch_size, -1)

class TextRecurrentLayer(nn.Module):
    """
    A class for text modeling including Embeddings + GRU + Last layer extraction
    """
    
    def __init__(self, 
                 text_name, 
                 text_vocabulary_size: int, 
                 text_embedding_dimension: int = 20,
                 text_bidirectional = True, 
                 text_recurrent_hidden_size = 20, 
                 text_recurrent_layers = 1, 
                 text_rnn = "GRU"):
        super(TextRecurrentLayer, self).__init__()
        
        self.text_name = text_name
        self.embedding = nn.Embedding(num_embeddings=text_vocabulary_size, 
                                      embedding_dim = text_embedding_dimension,
                                      padding_idx=0)

        #initialize RNN/LSTM/GRU with getattr and provide necessary inputs
        self.rnn = getattr(nn, text_rnn)(input_size = text_embedding_dimension, 
                                         hidden_size = text_recurrent_hidden_size, 
                                         num_layers = text_recurrent_layers, 
                                         bidirectional = text_bidirectional, 
                                         batch_first = True)

        self.last_time_step = LastTimeStep(rnn_layers = text_recurrent_layers, 
                                           bidirectional = text_bidirectional)
        
    def forward(self, input):
        x = self.embedding(input)
        x = self.rnn(x)
        x = self.last_time_step(x)
        
        return x
        


class PytorchModel(pl.LightningModule):
    def __init__(self,  
                 target_encoder = None, 
                 is_target_log = True,
                 optimizer:str = "Adam",
                 metric_to_monitor = "RMSLE", 
                 numerical_input_size:int = 0, 
                 numerical_batch_normalization:bool = True,
                 categorical_embedding_size: List[Tuple[int, int]] = None, 
                 categorical_embedding_dropout:float = 0, 
                 text_as_embedding_bag: bool = False,
                 text_as_embedding_bag_mode: str = "mean", #sum, mean, max
                 text_vocabulary_size: Dict[str, int] = None,
                 text_embedding_dimension: int = 20,
                 text_bidirectional:bool = True, 
                 text_recurrent_hidden_size:int = 20, 
                 text_recurrent_layers:int = 1, 
                 text_rnn:Union[str, None] = "GRU",
                 char_vocabulary_size: Dict[str, int] = None,
                 char_embedding_dimension: int = 20,
                 char_bidirectional:bool = True, 
                 char_recurrent_hidden_size:int = 20, 
                 char_recurrent_layers:int = 1, 
                 char_rnn:Union[str, None] = "GRU",
                 linear_layer_skip_connections:Union[None, Tuple[int, Tuple[List[int], List[float]]]] = None, #n_layers, [(n_out, dropouts)]
                 linear_layers:Tuple[List[int], List[float]] = ([1024],[0.2]), 
                 linear_layer_normalization:Union[str, None] = "BatchNorm1d",
                 normalization_before_activation:bool = False, 
                 linear_layer_activation = nn.ReLU(inplace=True), 
                 final_linear_layer:bool = True,
                 final_normalization: bool = False, 
                 target_range: Tuple[int, int] = None, 
                 loss_function = nn.MSELoss(),
                 learning_rate:float = 0.001, 
                 pretrained_hparams: bool = False, 
                 verbose = False
                ):
        super(PytorchModel, self).__init__()
        
        self.optimizer = optimizer
        self.metric_to_monitor = metric_to_monitor

        #add specific metrics:
        #calculates metric for each step/epoch
        
        self.target_encoder = target_encoder
        self.metric = tm.MeanSquaredError(squared=False) if is_target_log else tm.MeanSquaredLogError()
        
        self.save_hyperparameters()

        if verbose:
            print(f"{self.hparams}")
        #required by pytorch lighgning

        if pretrained_hparams: 
            print(f"pretrained hparams: {self.hparams}")
        else:
            self.loss_function = loss_function
            self.learning_rate = learning_rate

            if linear_layers is not None:
                linear_layer_sizes, linear_layer_dropouts = linear_layers
            else:
                linear_layer_sizes, linear_layer_dropouts = [], []

            if len(linear_layer_sizes) != len(linear_layer_dropouts):
                raise Exception(f"number of linear layers {linear_layer_sizes} does not correspond to number of linear layer dropouts {linear_layer_dropouts}")

            self.embeds = nn.ModuleList([nn.Embedding(num_embeddings=ni, embedding_dim=nf, padding_idx = 0) for ni,nf in categorical_embedding_size])
            self.categorical_dropout = nn.Dropout(categorical_embedding_dropout)
            
            n_categorical_embeddings = sum(e.embedding_dim for e in self.embeds)
            self.categorical_embeddings_size = n_categorical_embeddings
            
            self.batch_normalization_numerical = nn.BatchNorm1d(numerical_input_size) if numerical_batch_normalization else None
            self.numerical_input_size = numerical_input_size
            

            self.text_vocabulary_size =  text_vocabulary_size
            self.text_as_embedding_bag = text_as_embedding_bag

            print("processing text")
            #dictionary of text feature and its vocabulary
            if text_vocabulary_size is not None: 
                if text_as_embedding_bag == False and text_rnn is not None:
                    self.text_embeddings = nn.ModuleList([TextRecurrentLayer(text_name = text_name, 
                                                                            text_vocabulary_size = vocabulary_size,
                                                                            text_embedding_dimension = text_embedding_dimension,
                                                                            text_bidirectional = text_bidirectional, 
                                                                            text_recurrent_hidden_size = text_recurrent_hidden_size,
                                                                            text_recurrent_layers = text_recurrent_layers, text_rnn=text_rnn) for text_name, vocabulary_size in text_vocabulary_size.items()])
                    #number of text features multiplied by recurrent hidden size multiplied by 2 or 1 (if not bidirectional)
                    n_text_embeddings = len(text_vocabulary_size)*text_recurrent_hidden_size * (2 if text_bidirectional else 1)
                elif text_as_embedding_bag == True:

                    #https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
                    self.text_embeddings = nn.ModuleList([
                                                            nn.EmbeddingBag(num_embeddings = vocabulary_size, 
                                                                            embedding_dim = text_embedding_dimension, 
                                                                            mode = text_as_embedding_bag_mode, 
                                                                            padding_idx=0)  for text_name, vocabulary_size in text_vocabulary_size.items()
                                                        ])

                    #total length is the number of text variables multiplied by text_embedding_dimension
                    n_text_embeddings = len(text_vocabulary_size) * text_embedding_dimension

            else:
                self.text_embeddings = []
                n_text_embeddings = 0


            print("processing chars")
            if char_vocabulary_size is not None and char_rnn is not None:
                self.char_embeddings = nn.ModuleList([TextRecurrentLayer(text_name = char_name, 
                                                                        text_vocabulary_size = vocabulary_size,
                                                                        text_embedding_dimension = char_embedding_dimension,
                                                                        text_bidirectional = char_bidirectional, 
                                                                        text_recurrent_hidden_size = char_recurrent_hidden_size,
                                                                        text_recurrent_layers = char_recurrent_layers, text_rnn=char_rnn) for char_name, vocabulary_size in char_vocabulary_size.items()])
                n_char_embeddings = len(char_vocabulary_size)*char_recurrent_hidden_size * (2 if char_bidirectional else 1)
            else:
                self.char_embeddings = []
                n_char_embeddings = 0
            
            
            
            self.n_text_embeddings = n_text_embeddings
            self.n_char_embeddings = n_char_embeddings

                       
            sizes_print = [[f"cat: {self.categorical_embeddings_size}"] + [f"num: {numerical_input_size}"] + [f"text: {n_text_embeddings}"] + [f"char:{n_char_embeddings}"]]

            sizes = [self.categorical_embeddings_size + numerical_input_size + n_text_embeddings + n_char_embeddings]

            #linear layers
            #number of activations is equal to the length of linear_layer_sizes + last linear layer 
            
            linear_layers = nn.ModuleList()
            if linear_layer_skip_connections is not None:
                skip_n_layers, (skip_n_out_list, skip_dropout_list) = linear_layer_skip_connections
                
                for i, (out, p) in enumerate(zip(skip_n_out_list, skip_dropout_list)):
                    n_in = sizes[-1]
                    skip_connection_module = SkipDenseConnection(n_layers=skip_n_layers, 
                                                                 n_in = n_in,
                                                                 n_out = out, 
                                                                 normalization=linear_layer_normalization, 
                                                                 dropout = p, 
                                                                 activation = linear_layer_activation) 
                    linear_layers.append(skip_connection_module)

                    sizes_print = sizes_print + [f"skip: {out}"]
                    sizes = sizes + [out]
                    

            
            if len(linear_layer_sizes) > 0:
                activations = [linear_layer_activation for _ in range(len(linear_layer_sizes))] 

                temp_layers = []
                for i, (out, p,activation) in enumerate(zip(linear_layer_sizes, linear_layer_dropouts,activations)):
                    n_in = sizes[-1]

                    linear_layer_module = LinearNormDropActivation(n_in=n_in,
                                                                   n_out = out, 
                                                                   normalization=linear_layer_normalization if (linear_layer_normalization is not None) and (i!=len(activations) or final_normalization) else None, 
                                                                   dropout=p, 
                                                                   activation=activation, 
                                                                   normalization_before_activation=normalization_before_activation)
                    temp_layers.append(linear_layer_module)        

                    sizes_print = sizes_print + [f"lin: {out}"]
                    sizes = sizes + [out]

                
                linear_layers.extend(temp_layers)

            #linear dense layers
            self.linear_layers = nn.Sequential(*linear_layers)

            if final_linear_layer == True:

                final_layer = nn.ModuleList()
                
                final_layer.append(LinearNormDropActivation(n_in = sizes[-1], n_out = 1, normalization=linear_layer_normalization if linear_layer_normalization is not None and final_normalization else None))
                sizes_print = sizes_print + [1]
                sizes = sizes + [1]

                if target_range is not None: final_layer.append(SigmoidRange(*target_range))

            self.final_layer = nn.Sequential(*final_layer)

            self.sizes = sizes
            if verbose:
                print(f"sizes detailed: {sizes_print}")
                print(f"sizes: {sizes}")
        

    def forward(self, x_input):
        
        #set_trace()

        #numerical data shape [batch_size, num_features]
        numerical_data = x_input["numerical_data"]

        #shape: [batch_size, num_features]
        categorical_data = x_input["categorical_data"]

        #dictionary str, tensor of shape [batch_size, max_length]
        #max length - the max length of sentence within batch_size
        text_data = x_input["text_data"]
        char_data = x_input["char_data"]
        
        text_embedding_bag_data = x_input["text_embedding_bag_data"]
        text_embedding_bag_offset_data = x_input["text_embedding_bag_offset_data"]

        #start with categorical
        if self.categorical_embeddings_size != 0:
            x = [embedding(categorical_data[:,i]) for i, embedding in enumerate(self.embeds)]
            x = torch.cat(x, dim = 1)
            x = self.categorical_dropout(x)

        #continue with numerical
        if self.numerical_input_size != 0:
            if self.batch_normalization_numerical is not None: numerical_data = self.batch_normalization_numerical(numerical_data)
            x = torch.cat([x, numerical_data], 1) if self.categorical_embeddings_size != 0 else numerical_data
            
        #check if there is text
        if len(self.text_embeddings) > 0:  
            if self.text_as_embedding_bag == False:
                x_text = [text_recurrent_layer(text_data[text_recurrent_layer.text_name]) for i, text_recurrent_layer in enumerate(self.text_embeddings)]
                x_text = torch.cat(x_text, dim = 1)
            else:
                x_text = [embedding_bag(input = text_embedding_bag_data[text_name], 
                                        offsets = text_embedding_bag_offset_data[text_name]) for embedding_bag, (text_name, vocab_size) in zip(self.text_embeddings, self.text_vocabulary_size.items())]
                #set_trace()
                x_text = torch.cat(x_text, dim = 1)
            
            x = torch.cat([x, x_text], dim = 1)

        #check if there is char text
        if len(self.char_embeddings) > 0:    
            x_text = [char_recurrent_layer(char_data[char_recurrent_layer.text_name]) for i, char_recurrent_layer in enumerate(self.char_embeddings)]
            x_text = torch.cat(x_text, dim = 1)
            
            x = torch.cat([x, x_text], dim = 1)
        #use the last layer before linear layer in other tasks
        dense_vectors = self.linear_layers(x)

        y_final = self.final_layer(dense_vectors)

        #x is a concatenated vector of all categoricals + numericals + text before it goes through dense layers
        return y_final, x

    #pytorch lightning
    def training_step(self, batch, batch_idx):

        #target shape [batch_size, 1]
        inputs, target = batch
        
        y_hat, _ = self(inputs)
        loss = self.loss_function(y_hat, target)

        #need to pass batch_size because inputs has nested dictionary and I get warning
        #check this issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/10349
        #https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/data.py#L45
        batch_size = len(target)
        self.log("loss", loss, batch_size = batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        
        
        y_hat, _ = self(inputs)
        val_loss = self.loss_function(y_hat, target)

        batch_size = len(target)

        #https://forums.pytorchlightning.ai/t/understanding-logging-and-validation-step-validation-epoch-end/291/2
        #https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/data.py#L45
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        target_temp = target.detach().cpu()
        pred_temp = y_hat.detach().cpu()

        if self.target_encoder is not None:
            #set_trace()
            target_temp = self.target_encoder.inverse_transform(target_temp)
            target_temp = torch.FloatTensor(target_temp)

            pred_temp = self.target_encoder.inverse_transform(pred_temp)
            pred_temp = torch.FloatTensor(pred_temp)
            
        self.metric(pred_temp, target_temp)

        self.log(self.metric_to_monitor, self.metric, on_step=False, on_epoch = True, prog_bar=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer)([ param for param in self.parameters() if param.requires_grad == True], lr = self.learning_rate)
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, patience=4)
        #related to issue: 
        # MisconfigurationException: `configure_optimizers` must include a monitor 
        # when a `ReduceLROnPlateau` scheduler is used. For example: {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': self.metric_to_monitor}
