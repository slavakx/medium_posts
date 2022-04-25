#pytorch
import torch 

from torch.nn.utils.rnn import pad_sequence


#keras tokenizer
from keras.preprocessing import text

import category_encoders as ce

from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import warnings

from typing import List, Dict



class PytorchDataset:
    """Pytorch Dataset"""

    def __init__(self, 
                 data_pd, 
                 column_target: str, 
                 columns_categorical: str = None, 
                 columns_text: List[str] = None,
                 columns_char: List[str] = None, 
                 encoder_numerical: Pipeline = None, 
                 tokenizer_text_params = {'filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                    'lower': True,
                                    'split': ' ',
                                    'char_level': False},
                 tokenizer_char_params = {'filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                    'lower': True,
                                    'split': ' ',
                                    'char_level': True},       
                 is_train: bool = True, 
                 encoder_text: Dict[str, text.Tokenizer] = None,
                 encoder_char: Dict[str, text.Tokenizer] = None, 
                 encoder_categorical: ce.OrdinalEncoder = None,
                 encoder_target = None, 
                 verbose = True):
        """
        
        """
        
        self.column_target = column_target
        self.encoder_target = encoder_target
        
        self.target = data_pd[column_target].values.reshape(-1, 1)
        
        self.data = data_pd.reset_index()
        
        self.columns_categorical = columns_categorical 
        self.columns_text = columns_text
        self.columns_char = columns_char
        
        self.encoder_numerical = encoder_numerical
        
        
        self.is_train = is_train
        
        if is_train:
            if verbose: print("train set mode")
            
            if self.encoder_target is not None:
                if verbose: print("=> target encoding")
                self.target = self.encoder_target.fit_transform(self.target)

            
            if self.encoder_numerical is not None:
                if verbose: print("=> numerical encoding")
                self.data_numerical = self.encoder_numerical.fit_transform(self.data)

            if self.columns_categorical is not None:
                if verbose: print(f"=> categorical encoding")
                self.encoder_categorical = ce.OrdinalEncoder(handle_missing="return_nan", handle_unknown='return_nan')
                self.data_categorical = self.encoder_categorical.fit_transform(self.data[columns_categorical])
                #don't forget that OrdinalEncoder begins with 1, so the uniques is always 1 less than vocabulary size including 0 (for out of value)
                #self.data_categorical_uniques = self.data_categorical.nunique(axis = 0).to_list()
                self.data_categorical_uniques  = [(self.data_categorical.iloc[:, i].nunique() + 1, 
                                                   self.get_recommended_embedded_dimensions(self.data_categorical.iloc[:, i].nunique() + 1)) for i, j in enumerate(columns_categorical)]
                
            
            self.encoder_text = {}
            self.text_vocabulary_size = {}
            if columns_text is not None:
                for column in columns_text:
                    if verbose: print(f"=> tokenizing {column}")
                    keras_tokenizer = text.Tokenizer(**tokenizer_text_params)
                    keras_tokenizer.fit_on_texts(self.data[column])
                    self.encoder_text[column] = keras_tokenizer
                    vocabulary_size = len(keras_tokenizer.word_counts) + 1
                    self.text_vocabulary_size[column] = vocabulary_size
                    if verbose: print(f"==> {column} vocabulary size {vocabulary_size} ")

            self.encoder_char = {}
            self.char_vocabulary_size = {}
            if columns_char is not None:
                for column in columns_char:
                    if verbose: print(f"=> tokenizing chars {column}")
                    keras_tokenizer = text.Tokenizer(**tokenizer_char_params)
                    keras_tokenizer.fit_on_texts(self.data[column])
                    self.encoder_char[column] = keras_tokenizer
                    vocabulary_size = len(keras_tokenizer.word_counts) + 1
                    self.char_vocabulary_size[column] = vocabulary_size
                    if verbose: print(f"==> {column} vocabulary size {vocabulary_size} ")
        else:
            if verbose: print("test set mode")
            
            if self.encoder_target is not None:
                if verbose: print("=> target encoding")
                self.target = self.encoder_target.transform(self.target)
            
            if self.encoder_numerical is not None:
                if verbose: print("=> numerical encoding")
                self.data_numerical = self.encoder_numerical.transform(self.data)

            if self.columns_categorical is not None:
                if verbose: print(f"=> categorical encoding")
                self.encoder_categorical = encoder_categorical
                self.data_categorical = self.encoder_categorical.transform(self.data[columns_categorical])
                #replace missing values 
                self.data_categorical.fillna(0, inplace = True)
                
            
            #assign text tokenizers
            self.encoder_text = encoder_text
            if columns_text is not None:
                for column in columns_text:
                    keras_tokenizer = self.encoder_text[column]
                    vocabulary_size = len(keras_tokenizer.word_counts) + 1
                    if verbose: print(f"{column} vocabulary size {vocabulary_size}")

            self.encoder_char = encoder_char
            if columns_char is not None:
                for column in columns_char:
                    keras_tokenizer = self.encoder_char[column]
                    vocabulary_size = len(keras_tokenizer.word_counts) + 1
                    if verbose: print(f"{column} vocabulary size {vocabulary_size}")

        #calculate min, max target range
        self.target_min = np.min(self.target)
        self.target_max = np.max(self.target)

        if verbose: print(f"target min, max range ({self.target_min}, {self.target_max})")

    def get_target_range(self): return (self.target_min, self.target_max)
                    
    def get_data_categorical_embedding_sizes(self):
        if self.is_train == False:
            warnings.warn("This is a Test Data. categorical embedding sizes are in Train Data")
            return -1
        else:
            return self.data_categorical_uniques
        
    def get_text_vocabulary_size(self):
        if self.is_train == False:
            warnings.warn("This is a Test Data. text vocabulary size embedding sizes are in Train Data")
            return -1
        else:
            return self.text_vocabulary_size

    def get_char_vocabulary_size(self):
        if self.is_train == False:
            warnings.warn("This is a Test Data. char vocabulary size embedding sizes are in Train Data")
            return -1
        else:
            return self.char_vocabulary_size
    
    def get_recommended_embedded_dimensions(self, n_cat: int):
        return min(600, int(round(1.6 * n_cat**0.56)))
            
    def get_encoder_numerical(self) -> Pipeline: return self.encoder_numerical
    
    def get_encoder_text(self) -> Dict[str, text.Tokenizer]: return self.encoder_text

    def get_encoder_char(self) -> Dict[str, text.Tokenizer]: return self.encoder_char
    
    def get_encoder_categorical(self): return self.encoder_categorical
    
    def get_encoder_target(self): return self.encoder_target
    
    def get_target_name(self): return self.column_target

    def get_data_numerical(self): return self.data_numerical
    
    def get_data_categorical(self): return self.data_categorical
    
    def get_columns_text(self): return self.columns_text

    def get_columns_char(self): return self.columns_char

    def get_columns_categorical(self): return self.columns_categorical
    
    def get_data_text(self, column_name: str, text: List[str]):
        return self.encoder_text[column_name].texts_to_sequences(text)

    def get_data_char(self, column_name: str, text: List[str]):
        return self.encoder_char[column_name].texts_to_sequences(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        numeric_data = self.data_numerical[idx]
        categorical_data = self.data_categorical.iloc[idx].values
        
        #decorate as list because self.data[column].iloc[idx] returns a string, but texts_to_sequences expects a list of strings
        if self.columns_text is not None:
            text_data = {column: self.encoder_text[column].texts_to_sequences([self.data[column].iloc[idx]])[0] for column in self.columns_text}
        else:
            text_data = None

        if self.columns_char is not None:
            char_data = {column: self.encoder_char[column].texts_to_sequences([self.data[column].iloc[idx]])[0] for column in self.columns_char}
        else:
            char_data = None
        
        #when encoder applied it is a numpy array so pay attention. 
        target_data = self.target[idx]

        sample = {
            'numerical_data': numeric_data,
            'categorical_data': categorical_data,
            'text_data': text_data,
            'char_data': char_data, 
            'target': target_data
            }

        return sample
    
    
def pytorch_collate_fn(batch):
    """
    used by DataLoader instead of default collate function

    Args:
        batch (Dict): contains numerical_data, categorical_data, text_data, char_data, target keys

    Returns:
        [type]: [description]
    """
    numerical_data_list = []
    categorical_data_list = []
    text_dict = {}
    char_dict = {}
    
    target_data_list = []


    #for EmbeddingBag
    #holds 
    text_embedding_bag_index_dict = {}
    text_embedding_bag_offset_dict = {}
    


    for record in batch:
        numerical_data = record["numerical_data"]
        categorical_data = record["categorical_data"]
        text_data = record["text_data"]
        char_data = record["char_data"]

        if text_data is not None:
            text_data_columns = text_data.keys()
        else:
            text_data_columns = []

        if char_data is not None:
            char_data_columns = char_data.keys()
        else:
            char_data_columns = []
        
        
        target_data = record["target"]

    
        numerical_data_list.append(numerical_data)
        categorical_data_list.append(categorical_data)
        target_data_list.append(target_data)
    
        
        #iterate through text columns
        for column in text_data_columns:
            if column not in text_dict:
                text_dict[column] = []
                text_embedding_bag_index_dict[column] = []
                text_embedding_bag_offset_dict[column] = []
            text = text_data[column]
            text_length = len(text)
            #set_trace()
            text_dict[column].append(torch.LongTensor(text))
            #text is list of text indices
            text_embedding_bag_index_dict[column].extend(text)

            #text_length is just a number: length of each product name
            text_embedding_bag_offset_dict[column].append(text_length)

        #iterate through char columns
        for column in char_data_columns:
            if column not in char_dict:
                char_dict[column] = []
            char_dict[column].append(torch.LongTensor(char_data[column]))



    #iterte through text data intended for EmbeddingBag implementation and convert it to tensor
    for column in text_data_columns:
        temp = text_embedding_bag_index_dict[column]

        #shape: 1D vector of indices for all words in a batch
        text_embedding_bag_index_dict[column] = torch.tensor(temp, dtype=torch.long)

        offset = text_embedding_bag_offset_dict[column]
        #insert 0 length in the first position
        offset.insert(0, 0)

        offset_cumsum = torch.tensor(offset[:-1]).cumsum(dim = 0)

        #https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
        #shape: 1D vector of size (batch_size - 1): holds length indices 
        text_embedding_bag_offset_dict[column] = offset_cumsum

    #batch[0]- first batch
    #batch[0]["text_data"] - just get the text columns from the first batch




    if len(text_dict) > 0:
        text_encodings = {column: pad_sequence(text_dict[column], padding_value=0, batch_first = True) for column in batch[0]["text_data"].keys()}
    else:
        text_encodings = None

    if len(char_dict) > 0:
        char_encodings = {column: pad_sequence(char_dict[column], padding_value=0, batch_first = True) for column in batch[0]["char_data"].keys()}
    else:
        char_encodings = None

    #pads = 
    return ({"numerical_data": torch.as_tensor(numerical_data_list),
            "categorical_data": torch.as_tensor(categorical_data_list).long(), 
            "text_data": text_encodings, 
            "char_data": char_encodings, 
            "text_embedding_bag_data": text_embedding_bag_index_dict, 
            "text_embedding_bag_offset_data": text_embedding_bag_offset_dict,
            }, 
            torch.as_tensor(target_data_list).float())


def build_pytorch_dataset(train_df:pd.DataFrame, 
                         test_df:pd.DataFrame, 
                         encoder_numerical, 
                         categorical_names:List[str], 
                         text_names:List[str],
                         char_names:List[str], 
                         target_name:str, 
                         encoder_target = None, 
                         verbose = True):

    if verbose: print(f"target: {target_name}")
    if verbose: print(f"train: {train_df.shape}")
    if verbose: print(f"test: {test_df.shape}")
    
    dd_train  = PytorchDataset(train_df, 
                              column_target=target_name,
                              encoder_numerical = encoder_numerical, 
                              columns_categorical = categorical_names, 
                              columns_text = text_names, 
                              columns_char = char_names, 
                              encoder_target=encoder_target, 
                              is_train = True, 
                              verbose = verbose)


    dd_test = PytorchDataset(test_df, 
                            column_target = target_name, 
                            encoder_numerical=dd_train.get_encoder_numerical(),
                            columns_categorical = categorical_names, 
                            columns_text = text_names,
                            columns_char = char_names, 
                            is_train = False, 
                            encoder_text = dd_train.get_encoder_text(), 
                            encoder_char = dd_train.get_encoder_char(),
                            encoder_categorical = dd_train.get_encoder_categorical(), 
                            encoder_target=dd_train.get_encoder_target(), 
                            verbose = verbose
                        )

    return (dd_train, dd_test)


def build_test_dataset(train_dd: PytorchDataset, 
                       test_df:pd.DataFrame, 
                       verbose = True):

    target_name = train_dd.get_target_name()
    if verbose: print(f"target: {target_name}")
    if verbose: print(f"train: {len(train_dd)}")
    if verbose: print(f"test: {test_df.shape}")


    dd_test = PytorchDataset(test_df, 
                            column_target = target_name, 
                            encoder_numerical=train_dd.get_encoder_numerical(),
                            columns_categorical = train_dd.get_columns_categorical(),
                            columns_text = train_dd.get_columns_text(),
                            columns_char = train_dd.get_columns_char(),
                            is_train = False, 
                            encoder_text = train_dd.get_encoder_text(), 
                            encoder_char = train_dd.get_encoder_char(),
                            encoder_categorical = train_dd.get_encoder_categorical(), 
                            encoder_target=train_dd.get_encoder_target(), 
                            verbose = verbose
                        )

    return dd_test