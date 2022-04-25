import numpy as np
import pandas as pd


import torch
from torch.utils.data import DataLoader

from typing import List

import pytorch_dataset
import pytorch_model

from IPython.core.debugger import set_trace

def get_product_embeddings_from_hidden_layer(model: pytorch_model.PytorchModel, 
                                             dataset: pd.DataFrame, 
                                             train_reference: pytorch_dataset.PytorchDataset,
                                             hidden_layer_size:int, 
                                             item_column:str = "name"
                                             ):
    """Constucts product embeddings from the hidden layer

    Args:
        model (PytorchModel): [description]
        dataset (pd.DataFrame): [description]
        train_refernce (pytorch_dataset.PytorchDataset): [description]
        hidden_layer_size (int): [description]
        item_column (str, optional): [description]. Defaults to "name".
        advertising_theme_column (str, optional): [description]. Defaults to "advertising_theme_name".

    Returns:
        [type]: [description]
    """
    
    print(f"dataset size: {len(dataset)}, unique products: {dataset[item_column].nunique()}")

    product_name_2_vector_dictionary_train = {}
    index_2_product_name_dictionary_train = {}
    product_hidden_layer_matrix_train = np.zeros(shape = (len(dataset), hidden_layer_size))

    model.eval()
    with torch.no_grad():
        for global_product_index, product_name in enumerate(dataset[item_column].values):

            #set_trace()
            ##########################################################################################
            if global_product_index % 10000 == 0:
                print(f"processed {global_product_index}")
            
            
            #allocate matrix
            #temp_hidden_layer_matrix_train = np.zeros(shape = (1, hidden_layer_size))
            
            product_by_index_df = dataset.iloc[global_product_index: (global_product_index + 1), :]
            dd_product_by_index = pytorch_dataset.build_test_dataset(train_reference, product_by_index_df, verbose = False)

            dd_product_by_index_loader = DataLoader(dd_product_by_index, shuffle = False, batch_size = 1, collate_fn = pytorch_dataset.pytorch_collate_fn)
            product_by_index_torch = next(iter(dd_product_by_index_loader))

            #special case coverage when text column does not have any single word common with training set. 
            #in this case it will be tensor([], size=(1, 0), dtype=torch.int64) and will throw exception. 
            #we need to substitute 0 index embedding for "no word"
            #shape of invalid tensor is (1,0), but it needs to be (1,1)
            #size_0 = product_by_index_torch[0]["text_data"][item_column].shape[0]

            #iterate along all text columns
            for text_column in product_by_index_torch[0]["text_data"].keys():
                if product_by_index_torch[0]["text_data"][text_column].shape[1] == 0:
                    product_by_index_torch[0]["text_data"][text_column] = torch.zeros(size = (1, 1), dtype = torch.int64)
            

            #use hidden_layer as a product embedding
            final_output, hidden_layer = model(product_by_index_torch[0])
            hidden_layer = hidden_layer.numpy()
            #collect hidden layers corresponding to this product
            #average
            num_vector = hidden_layer.mean(axis = 0)
            #print(f"{global_product_index} => {num_vector}")

            key = f"{product_name}_{global_product_index}"

            product_hidden_layer_matrix_train[global_product_index, :] = num_vector
            product_name_2_vector_dictionary_train[key] = num_vector
            index_2_product_name_dictionary_train[global_product_index] = product_name

    result = {}
    result["matrix"] = product_hidden_layer_matrix_train
    result["product2vector"] = product_name_2_vector_dictionary_train
    result["index2product"] = index_2_product_name_dictionary_train
    return result


def get_product_embeddings_from_embedding_layer(model: pytorch_model.PytorchModel, 
                                                unique_products: List[str], 
                                                train_reference: pytorch_dataset.PytorchDataset,
                                                product_embedding_index = 0, 
                                                item_column:str = "name"
                                                ):
    
    model.eval()
    with torch.no_grad():
        print(f"products : {len(unique_products)}")
        products_train_vector_dict = {}
        
        emb_dim = model.text_embeddings[product_embedding_index].embedding.weight.data.shape[1]
        products_train_embedding_matrix = np.zeros((len(unique_products), emb_dim))

        unknown_products = []
        
        for i, product_name in enumerate(unique_products):
            if i % 10000 == 0:
                print(f"processed {i}")
            #get product name and convert it into embedding vectors
            #product = average of all its words in the embedding matrix
            product_indices = train_reference.get_encoder_text()[item_column].texts_to_sequences([product_name])[0]
            if len(product_indices) > 0:
                product_vector = model.text_embeddings[product_embedding_index].embedding.weight.data.numpy()[product_indices].mean(axis = 0)
                products_train_vector_dict[product_name] = product_vector

                products_train_embedding_matrix[i, :] = product_vector
            else:
                unknown_products.append(product_name)

    result = {}
    result["matrix"] = products_train_embedding_matrix
    result["product2vector"] = products_train_vector_dict
    result["unknown"] = unknown_products

    return result
        
        
        
def get_categorical_embedding(model: pytorch_model.PytorchModel, 
                              train_set: pd.DataFrame, 
                              embedding_category_index: int, 
                              train_set_category_column: str):
    embeddings = model.embeds[embedding_category_index].weight.data.numpy()
    #remove first which corresponds to unknown
    embeddings = embeddings[1:, :]
    print(embeddings.shape)
    n_count = train_set[train_set_category_column].nunique()
    print(f"train set {train_set_category_column}: {n_count}")
    
    return embeddings


def get_categorical_embedding_mapping(dd_train: pytorch_dataset.PytorchDataset, 
                                      embedding_category_index):
    encoder_categorical = dd_train.get_encoder_categorical()
    mapping = encoder_categorical.category_mapping[embedding_category_index]["mapping"].reset_index().drop(columns = 0).rename(columns = {"index": "mapping"}).reset_index().rename(columns = {"level_0": "index"})
    return mapping


def predict(model, data_loader: DataLoader, is_log, target_encoder = None):
    preds = []
    targets = []
    
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    with torch.no_grad():
        for (inputs, target) in data_loader:

            pred, _ = model(inputs)
            
            preds.extend(pred.numpy())
                            
            targets.extend(target.numpy())

    targets = np.asarray(targets)
    preds = np.asarray(preds)
    
    if target_encoder is not None:
        preds = target_encoder.inverse_transform(preds)
        targets = target_encoder.inverse_transform(targets)
                            
    preds = preds.ravel()
    targets = targets.ravel()
    
    if is_log:
        preds = np.expm1(preds)
        targets = np.expm1(targets)

    assert(len(targets) == len(preds))
    
    return {"y_true": targets, "y_pred": preds} 