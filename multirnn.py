import os
import numpy as np
import pandas as pd
from termcolor import colored
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense

class MultiRNN():
    def __init__(self,
                 train:pd.DataFrame, test:pd.DataFrame, 
                 length:int, LSTM_units:int | list, activation:str | list,
                 optimizer:str | list, batch_size:int=1, epochs:int | list=25):
        
        """
        Parameters
        ----------

        train : pandas.DataFrame
            A valid training pandas dataframe
            of original dataset having index as orginal dataset.

        test : pandas.DataFrame
            A valid testing pandas dataframe
            of original dataset having index as orginal dataset.

        length : int > 0
            Positive int, length of the output sequences.

        LSTM_units : int > 0 | list of int > 0
            Positive int, for cells in LSTM of keras or PyTorch
            or list of positive ints for each feature LSTM
            cells needed for this feature that matches the index.

        activation : str | list of str, default = "tanh"
            A valid activation function for RNN LSTM layer,
            or list of activation functions for each feature.
            Options ["tanh", "sigmoid", "softmax", "relu"]

        optimizer : str | list of str, default = "adam"
            A valid optimizer for the output layer,
            or list of optimizers for each feature.
            Options ["adam", "rmsprop", "sgd"]

        batch_size : int > 0, default = 1
            Number of time series samples in each batch
            (except maybe the last one).

        epochs : int > 0 | list of int, default = 25 per feature
            Number of epochs to train the model.
            Could either be a single int or list of ints.
        """

        # check if train is a pandas DataFrame
        if not isinstance(train, pd.DataFrame):
            raise ValueError("'train' must be a pandas DataFrame")
        
        # check if test is a pandas DataFrame
        if not isinstance(test, pd.DataFrame):
            raise ValueError("'test' must be a pandas DataFrame")
        
        # check if the datasets (ie. 'train' and 'test') are ready for processing
        for dataset_name, dataset in [("train", train), ("test", test)]:
            ready_, errors = MultiRNN.ready_for_processing(dataset)
            if not ready_:
                raise ValueError(f"{dataset_name} is NOT ready. Please fix before proceeding.\n" +
                                 f"Errors found : {errors}")
        
        # check if length is a positive integer
        if not isinstance(length, int) or length < 1:
            raise ValueError("'length' must be a positive integer")
        
        # check if LSTM_units is a positive integer or list of positive integers
        if not isinstance(LSTM_units, (int, list)):
            raise ValueError("'LSTM_units' must be a positive integer or list of positive integers")
        
        # check if activation is a string or list of strings with valid activation functions
        # and check if they are in ["tanh", "sigmoid", "softmax", "relu"]
        if not isinstance(activation, (str, list)):
            raise ValueError("'activation' must be a string or list of strings")
        
        # check the elements if it is a list for LSTM_units
        if isinstance(LSTM_units, list):
            if len(LSTM_units) != len(train.columns):
                raise ValueError("Length of 'LSTM_units' must be equal to the number of columns in the dataset")
            
            for unit in LSTM_units:
                if not isinstance(unit, int) or unit < 1:
                    raise ValueError(f"Invalid number of LSTM units '{unit}'")
        
        # check the elements if it is a list
        if isinstance(activation, list):
            for act in activation:
                if act not in ["tanh", "sigmoid", "softmax", "relu"]:
                    raise ValueError(f"Invalid activation function '{act}'")
                
        # check if optimizer is a string or list of strings with valid optimizers
        # and check if they are in ["adam", "rmsprop", "sgd"]
        if not isinstance(optimizer, (str, list)):
            raise ValueError("'optimizer' must be a string or list of strings")
        
        # check the elements if it is a list
        if isinstance(optimizer, list):
            for opt in optimizer:
                if opt not in ["adam", "rmsprop", "sgd"]:
                    raise ValueError(f"Invalid optimizer '{opt}'")
                
        # check if batch_size is a positive integer
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("'batch_size' must be a positive integer")
        
        # check if epochs is a positive integer or list of positive integers
        if not isinstance(epochs, (int, list)):
            raise ValueError("'epochs' must be a positive integer or list of positive integers")
        
        # check the elements if it is a list
        if isinstance(epochs, list):
            for ep in epochs:
                if not isinstance(ep, int) or ep < 1:
                    raise ValueError(f"Invalid number of epochs '{ep}'")
                
        # save the parameters to the object
        self.train = train
        self.test = test
        self.length = length
        self.LSTM_units = LSTM_units
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

        # generate datasets and scalers (method handles all parameter checks)
        # returns 'self.datasets' and 'self.scaled_datasets'
        # this method should be called by method 
        # generate_model_list_per_column_in_dataset()
        # so we need to change this later.
        self.generate_dataset_per_column_with_original_index(self.train, 
                                                             self.test)

        # generate models for each column
        # we also need to add the feature to have a list
        # of lstm units, activations, optimizers, and epochs

        self.model_dict = {} # dictionary to save models

        for idx, col in enumerate(self.datasets["train"].keys()):
            LSTM_units = MultiRNN.get_param_value(self.LSTM_units, idx)
            activation = MultiRNN.get_param_value(self.activation, idx)
            optimizer = MultiRNN.get_param_value(self.optimizer, idx)
            epochs = MultiRNN.get_param_value(self.epochs, idx)

            model, model_name, generator, val_generator = self.build_model_per_column(col, 
                                                                                    self.scaled_datasets["train"][col]["data"],
                                                                                    self.scaled_datasets["test"][col]["data"],
                                                                                    length, LSTM_units, activation, optimizer, 
                                                                                    batch_size, epochs)
            
            self.model_dict[model_name] = {"model" : model,
                                           "generator" : generator,
                                           "val_generator" : val_generator}
            
            # for testing and decrease execution time, should be removed later
            if idx == 1:
                break

    @staticmethod
    def get_param_value(param, idx): # from ChatGPT to decrease redundancy
        if isinstance(param, list):
            return param[idx]
        return param

    @staticmethod
    def ready_for_processing(dataset):
        """
        Static method to check if the dataset is ready to process.
        Checks NaN values, and if the dataset is a pandas DataFrame etc.

        NOTE to self: take inspiration from AutoEDA from ML-APP
        (https://github.com/Astr0Bit/ML_APP, or check folder "from ML_APP/auto_eda.py")

        Parameters
        ----------
        dataset : pandas.DataFrame
            A valid pandas DataFrame.

        Returns True if ready, False if not.
        """
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("'dataset' must be a pandas DataFrame")
        
        ready_ = None # True if ready, False if not

        # Check for errors
        errors = {"missing_values" : None,
                  "duplicates" : None,
                  "outliers" : None}

        if dataset.duplicated().sum() > 0:
            errors["duplicates"] = dataset.duplicated().sum()

        if dataset.isna().sum().sum() > 0:
            errors["missing_values"] = dataset.isna().sum().sum()

        if not all(value is None for value in errors.values()):
            ready_ = False

        else:
            ready_ = True

        # Return ready_ and errors
        return ready_, errors

    # for convenience, this method should be called in the constructor
    def generate_dataset_per_column_with_original_index(self,
                                                        train, test):
        """
        Method that will generate datasets as descibed in C. (check PDF)

        C. Build RNN LSTM model for each generated dataset, so if 
        we have for example 100 features, we shall train 100 
        models over generated train dataset and validate using 
        generated train dataset.

        Optional: can save them to local folder for later use.

        Parameters
        ----------
        train : pandas.DataFrame
            Training data.

        test : pandas.DataFrame
            Testing data.

        save : bool, optional
            Whether to save the datasets and scalers, by default False.

        dir_path : str, optional
            Directory path to save the datasets and scalers, by default "/col_datasets".
        """

        self.datasets = {
            "train" : {col:pd.DataFrame(train[col]) for col in train.columns},
            "test" : {col:pd.DataFrame(test[col]) for col in test.columns}
            } # dictionary to save datasets

        self.scaled_datasets = {"train": {}, "test": {}} # dictionary to save scaled datasets

        # scale the train data
        for col, df in self.datasets["train"].items(): # iterate over columns
            scaler = MinMaxScaler() # create MinMaxScaler object
            scaled_data = scaler.fit_transform(df) # fit and transform data
            self.scaled_datasets["train"][col] = {"data": scaled_data, "scaler": scaler} # save to dictionary

        # next step is to scale the test data
        for col, df in self.datasets["test"].items(): # iterate over columns
            scaler = self.scaled_datasets["train"][col]["scaler"] # get scaler from train data
            scaled_data = scaler.transform(df) # transform data
            self.scaled_datasets["test"][col] = {"data": scaled_data, "scaler": scaler} # save to dictionary

        return self.datasets, self.scaled_datasets
    
    def save_datasets(self, dir_path:str="/col_datasets"):
        """
        Method to save datasets to local folder.

        Parameters
        ----------
        dir_path : str, default="/col_datasets"
            Directory path to save the datasets.
        """

        if dir_path == "../" or dir_path == "/":
            dir_path = dir_path + "col_datasets"

        if not isinstance(dir_path, str): # check if dir_path is a string
            raise ValueError("'dir_path' must be a string")

        if not os.path.exists(dir_path): # create directory if it does not exist
            os.makedirs(dir_path) # create directory

            for key in self.datasets.keys(): # iterate over keys
                # create sub-directory for original data
                path = f"{dir_path}/original/{key}"
                if not os.path.exists(path): # create sub-directory if it does not exist
                    os.makedirs(path) # create sub-directory

                # save original data
                for col, df in self.datasets[key].items(): # iterate over columns
                    path = f"{dir_path}/original/{key}/{key}_{col}.csv"
                    df.to_csv(path) # save to local folder

    def save_scalers(self, dir_path:str="/col_datasets"):
        """
        Method to save scalers to local folder.

        Parameters
        ----------
        dir_path : str, default="/col_datasets"
            Directory path to save the scalers.
        """

        if dir_path == "../" or dir_path == "/":
            dir_path = dir_path + "col_datasets"

        if not isinstance(dir_path, str): # check if dir_path is a string
            raise ValueError("'dir_path' must be a string")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for key in self.datasets.keys(): # iterate over keys
            for col, data in self.scaled_datasets[key].items(): # iterate over columns
                path = f"{dir_path}/scalers" # create sub-directory for scalers
                if not os.path.exists(path): # create sub-directory if it does not exist
                    os.makedirs(path) # create sub-directory

                path = f"{dir_path}/scalers/{col}_scaler.pkl" # save scalers
                dump(data["scaler"], path) # save to local folder

    def save_scaled_datasets(self, dir_path:str="/col_datasets"):
        """
        Method to save scaled datasets to local folder.

        Parameters
        ----------
        dir_path : str, default="/col_datasets"
            Directory path to save the scaled datasets.
        """

        if dir_path == "../" or dir_path == "/":
            dir_path = dir_path + "col_datasets"

        if not isinstance(dir_path, str): # check if dir_path is a string
            raise ValueError("'dir_path' must be a string")
        
        if not os.path.exists(dir_path): # create directory if it does not exist
            os.makedirs(dir_path) # create directory

        for key in self.datasets.keys(): # iterate over keys
            path = f"{dir_path}/scaled/{key}" # create sub-directory for scaled data
            if not os.path.exists(path): # create sub-directory if it does not exist
                os.makedirs(path) # create sub-directory

            for col, data in self.scaled_datasets[key].items(): # iterate over columns
                path = f"{dir_path}/scaled/{key}/scaled_{key}_{col}.csv" # save scaled data
                pd.DataFrame(data["data"], index=self.datasets[key][col].index).to_csv(path) # save to local folder

    # this method should be called in generate_model_list_per_column_in_dataset(),
    # which should be called within the constructor
    def build_model_per_column(self, col:str, scaled_train:pd.DataFrame, 
                               scaled_test:pd.DataFrame, length:int, 
                               LSTM_units:int|list, activation:str|list,
                               optimizer:str|list, batch_size:int=1, 
                               epochs:int|list=25):
        """
        Method that takes inputs and creates RNN LSTM for
        the generated single column dataset, based on other
        parameters provided. This method will do needed
        scaling, generating time series and build the
        RNN LSTM Keras / PyTorch model based on hyper 
        parameters with early stopping based on val_loss and
        patience of max 2 epochs.

        Optional: save losses dataframe as a csv-file to local folder.
        Optional: save model as Keras / PyTorch with .h5

        This method should also evaluate the model, it creates
        the model predictions of the part of train data equal
        to or shorter than test dataset length, and save it as
        DataFrame with original index in test values in order
        to plot both of them.

        Returns model, losses, and related model name.

        NOTE to self: this method should iterate over each column
        and data in dictionary self.scaled_datasets["train"] and
        build model. It should return model, losses, and model name.

        Parameters
        ----------
        col : str
            A valid column name in the original dataset.

        scaled_train : pandas.DataFrame
            Scaled Training data.

        scaled_test : pandas.DataFrame
            Scaled Testing data.

        length : int > 0
            Positive int, length of the output sequences.

        LSTM_units : int > 0 | list of int > 0
            Positive int, for cells in LSTM of keras or PyTorch
            or list of positive ints for each feature LSTM
            cells needed for this feature that matches the index.

        activation : str | list of str
            A valid activation function for RNN LSTM layer,
            or list of activation functions for each feature.
            Options ["tanh", "sigmoid", "softmax", "relu"]

        optimizer : str | list of str
            A valid optimizer for the output layer,
            or list of optimizers for each feature.
            Options ["adam", "rmsprop", "sgd"]

        batch_size : int > 0
            Number of time series samples in each batch
            (except maybe the last one).

        epochs : int > 0 | list of int, default = 25 per feature
            Number of epochs to train the model.
            Could either be a single int or list of ints.

        Returns model, losses, and model name.

        NOTE to self: losses is calculated after the model has been compiled
        example: 
            model.compile(loss="mean_squared_error", optimizer="adam")
            loss = model.evaluate(X, y) # where X for our case is the index, and y is the column
        """
        # as data is already scaled, we can use it directly
        # we can use the same data for training and validation

        model_name = f"{col}_model" # model name

        # create time series data
        generator = TimeseriesGenerator(scaled_train, 
                                        scaled_train, 
                                        length=length, 
                                        batch_size=batch_size)
        
        val_generator = TimeseriesGenerator(scaled_test, 
                                            scaled_test, 
                                            length=length, 
                                            batch_size=batch_size)
        
        # create LSTM model
        model = Sequential()
        model.add(LSTM(LSTM_units, 
                       return_sequences=True,
                       activation=activation, 
                       input_shape=(length, 1)))
        model.add(Dense(1))

        # compile the model
        model.compile(optimizer=optimizer, loss="mse")

        # get losses before training
        print(f"{colored('Loss for', 'green')} '{colored(model_name, 'cyan')}' {colored('before training:', 'green')}")
        losses = model.evaluate(generator)

        return model, model_name, generator, val_generator

    def fit_models(self, fit_on_all_data:bool=False):
        """
        Method to fit all models for each column in the dataset.
        Reads in models from dictionary 'model_dict' and fits them.
        
        ONLY run once your are ready to train the models.

        In order to get output of Keras / PyTorch models, losses DataFrame and model name.
        These will be run on each column in train dataset and collected into dictionary that has
        key as column name, and values is a dictionary that contains model, losses, and model name.
        (it is a dictionary of dictionaries)

        Parameters
        ----------
        fit_on_all_data : bool, default=False
            Whether to fit the model on all data, or only on train data.
            If True, the model will be fitted on the full dataset.
            If False, the model will be fitted on the train dataset.

        Returns the dictionary containing all columns.
        """

        if not isinstance(fit_on_all_data, bool):
                    raise ValueError("'fit_on_all_data' must be a boolean")
        
        self.fit_on_all_data = fit_on_all_data

        # iterate over models and fit them
        for idx, value in enumerate(self.model_dict.items()):
            model_name = value[0] # get model name
            col_name = model_name.split("_")[0] # get column name
            model = value[1]["model"] # get model
        
            if fit_on_all_data:
                self.datasets["full_data"][col_name] = pd.concat([self.datasets["train"][col_name].copy(),
                                                                  self.datasets["test"][col_name].copy()], axis=0) # concatenate train and test data
                
                self.scaled_datasets["full_data"][col_name]["scaler"] = MinMaxScaler() # create MinMaxScaler object
                self.scaled_datasets["full_data"][col_name]["data"] = self.scaled_datasets["full_data"][col_name]["scaler"].fit_transform(self.datasets["full_data"][col_name]) # fit and transform data
                self.model_dict[f"final_{model_name}"]["full_generator"] = TimeseriesGenerator(self.scaled_datasets["full_data"][col_name]["data"],
                                                                                               self.scaled_datasets["full_data"][col_name]["data"],
                                                                                               length=self.length,
                                                                                               batch_size=self.batch_size)
                
                print(f"{colored('Training model:', 'green')} '{colored(model_name, 'cyan')}'") # print model name
                model.fit(self.model_dict[model_name]["full_generator"], 
                          epochs=MultiRNN.get_param_value(self.epochs, idx), verbose=1) # fit model on full data
                self.model_dict[model_name]["final_model"] = model # save model to dictionary
                self.model_dict[model_name]["final_losses_df"] = pd.DataFrame(model.history.history) # save losses to dictionary

            else:
                generator = value[1]["generator"] # get generator
                val_generator = value[1]["val_generator"] # get val_generator
                print(f"{colored('Training model:', 'green')} '{colored(model_name, 'cyan')}'") # print model name
                model.fit(generator, epochs=MultiRNN.get_param_value(self.epochs, idx), 
                          validation_data=val_generator, verbose=1) # fit model

                self.model_dict[model_name]["model"] = model # save model to dictionary
                self.model_dict[model_name]["losses_df"] = pd.DataFrame(model.history.history) # save losses to dictionary

    def save_models(self, dir_path:str="/models"):
        """
        Method to save models to local folder.

        Parameters
        ----------
        dir_path : str, default="/models"
            Directory path to save the models.
        """
        
        if not hasattr(self, "fit_models"):
            raise AttributeError("Models have not been fitted yet. Please fit the models first.")        

        if not os.path.exists(dir_path): # create directory if it does not exist
            os.makedirs(dir_path) # create directory

        for key in self.model_dict.keys(): # iterate over keys
            path = f"{dir_path}/{key}.h5"
            if self.fit_on_all_data:
                path = f"{dir_path}/final"
                if not os.path.exists(path): # create directory if it does not exist
                    os.makedirs(path)
                    path = f"{path}/{key}.h5"
                    self.model_dict[key]["final_model"].save(path) # save model to local folder

            else:
                self.model_dict[key]["model"].save(path) # save model to local folder

    def predict(self):
        """
        Method to generate prediction for all features
        and input dataframe that has columns matching
        the original dataset.

        Returns prediction as a pandas DataFrame

        NOTE to self: is this method really supposed to
        take in a datarow? We have not yet made a test
        prediction, and we are supposed to later plot the
        test prediction against the real test values
        with method "plot_predict_against_test_dataset_per_column()".

        # Approach 1: take in a datset with matching column as original
        sub-dataset and create prediction column for that dataset. Problem
        with this is that the user can give a dataset with index we don't
        have in the original dataset.

        # Approach 2: make a simple forecast by taking in a daterange
        dataset where the column data is empty. Problem with this
        is that the user can give a dataset with index we don't have
        in the original dataset.

        # Approach 3: simply predict on the test dataset, that way
        we will have the same index as the original dataset. This
        is the best approach. We will use this approach.
        """

        self.test_pred_dfs = {} # dictionary to save predictions

        # iterate over models and predict
        for idx, value in enumerate(self.model_dict.items()):
            model_name = value[0]
            col_name = model_name.split("_")[0]
            test_predictions = []
            model = value[1]["model"]
            first_eval_batch = self.scaled_datasets["test"][col_name]["data"][-self.length:]
            current_batch = first_eval_batch.reshape((1, self.length, 1))

            for i in range(len(self.datasets["test"][col_name])):
                current_pred = model.predict(current_batch)[0][0] # another [0] was needed to fix the shape error
                test_predictions.append(current_pred)
                current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

            self.test_pred_dfs[col_name] = self.datasets["test"][col_name].copy() # dictionary to save predictions
            true_predictions = self.scaled_datasets["train"][col_name]["scaler"].inverse_transform(test_predictions)
            self.test_pred_dfs[col_name]["predictions"] = true_predictions

    def plot_predict_against_test_dataset_per_column(self, column:str, 
                                                     figure_width:int=5,
                                                     figure_height:int=5,
                                                     save_plot:bool=False,
                                                     save_plot_name:str="plot_test_vs_predict_"):
        """
        Method to plot the real test values and predicted values on the same plot
        using plot dimensions mentioned and column name. And save the plots.

        Parameters
        ----------
        column : str
            A valid column name in the original dataset.

        figure_width : int > 0
            Positive int, width of the plot.

        figure_height : int > 0
            Positive int, height of the plot.

        save_plot : bool, default=False
            Whether to save the plot or not.

        save_plot_name : str, default="plot_test_vs_predict_"
            Name of the plot to be saved.
        """

        if not hasattr(self, "test_pred_dfs"):
            raise AttributeError("Predictions have not been generated yet. Please generate predictions first using 'predict()' method.")

        if isinstance(column, str) and column in self.test_pred_dfs.keys():
            
            self.test_pred_dfs[column].plot(figsize=(figure_width, figure_height),
                                            title=f"Predictions vs Real values for {column}")

            if isinstance(save_plot_name, str) and save_plot:
                plt.savefig(f"{save_plot_name}_{column}.png")

        else:
            raise ValueError(f"Invalid column name '{column}'")

    def plot_loss_val_loss_per_column(self, column:str, 
                                      figure_width:int=5,
                                      figure_height:int=5,
                                      save_plot:bool=False,
                                      save_plot_name:str="plot_loss_val_loss_"):
        """
        Method to plot loss against val_loss 
        of input column and save it.
        
        Parameters
        ----------
        column : str
            A valid column name in the original dataset.

        figure_width : int > 0
            Positive int, width of the plot.

        figure_height : int > 0
            Positive int, height of the plot.

        save_plot : bool, default=False
            Whether to save the plot or not.

        save_plot_name : str, default="plot_loss_val_loss_"
            Name of the plot to be saved.
        """
        
        if isinstance(column, str):
            model_name = f"{column}_model"
            if model_name in self.model_dict.keys():
                self.model_dict[model_name]["losses_df"].plot(figsize=(figure_width, figure_height),
                                                              title=f"Loss vs Val_loss for {column}")
                if isinstance(save_plot_name, str) and save_plot:
                    plt.savefig(f"{save_plot_name}_{column}.png")

        else:
            raise ValueError(f"Invalid column name '{column}'")