import numpy as np
import tensorflow as tf
import pandas as pd
import time
import pickle
dtype = tf.float32
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy


def r2_score(y_truth, y_pred):
    # Returns the r2 score
    residual = tf.add(y_truth, -y_pred, name='residual')
    return 1-tf.reduce_sum(input_tensor=tf.square(residual)) / tf.reduce_sum(input_tensor=tf.square(y_truth-tf.reduce_mean(input_tensor=y_truth)))

def residual(y_truth, y_pred):
    return tf.add(y_truth, -y_pred)

class ElasticNet(tf.keras.regularizers.Regularizer):
    # Implements group lasso + ridge for the first kernel in the bottleneck
    def __init__(self, l1=1e-8, l2=1.0): 
        self.l1 = l1
        self.l2 = l2

    def __call__(self, x): 
        return self.l2*tf.math.reduce_sum(input_tensor=tf.math.square(x)) + \
                             self.l1*tf.reduce_sum(input_tensor=tf.norm(tensor=x, ord = 2, axis = 1))

    def get_config(self): 
        return {'l1': float(self.l1), 'l2': float(self.l2)}

    
class ClassificationPreTrain:
    """
    Implements a bottleneck neural network with keras that does classication rather than regression. The trained weights can be used
    to subsequently initialize the weights for the target task: regression.
    """
    
    def __init__(self, l1, l2, lr, act, input_dim, output_dim, nodes_list=[512, 128, 2, 128, 512]):
        """
        Constructor.
        :param l1: int, lasso penalty
        :param l2: int, ridge penalty
        :param lr: int, learning rate for Adam
        :param act: string, activation function
        :param input_dim: int, input layer dimensionality
        :param output_dim: int, output layer dimensionality
        :param nodes_list: list, integers denoting # nodes in each layer (optional, default=[512, 128, 2, 128, 512])
        """
        
        self.l1=l1
        self.l2=l2
        self.lr=lr
        self.act=act
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.nodes_list=nodes_list
        
        # Architecture
        keras.backend.clear_session()
        self.m = Sequential()
        self.m.add(Dense(self.nodes_list[0], activation=self.act, input_shape=(self.input_dim, ), \
                    kernel_regularizer=ElasticNet(l1=self.l1, l2=self.l2), \
                    bias_regularizer=regularizers.l2(self.l2), \
                    name = 'W1'))
        self.m.add(Dense(self.nodes_list[1], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                    bias_regularizer=regularizers.l2(self.l2), \
                    name = 'W2'))
        # linear for latent space representation
        self.m.add(Dense(self.nodes_list[2], activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
                    bias_regularizer=regularizers.l2(self.l2), \
                    name='bottleneck'))
        self.m.add(Dense(self.nodes_list[3], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                    bias_regularizer=regularizers.l2(self.l2), \
                    name = 'W4'))
        self.m.add(Dense(self.nodes_list[4], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                    bias_regularizer=regularizers.l2(self.l2), \
                    name = 'W5'))
        self.m.add(Dense(self.output_dim, activation='softmax', \
                    kernel_regularizer=regularizers.l2(self.l2), \
                    bias_regularizer=regularizers.l2(self.l2), \
                    name = 'W6_class'))
        self.m.compile(loss=CategoricalCrossentropy(label_smoothing=0), \
                  optimizer=keras.optimizers.Adam(learning_rate=self.lr), \
                  metrics=[metrics.categorical_crossentropy])
        
    def train(self, x_train, cluster_train, x_test, cluster_test, epochs, bs, patience, \
              cvfold_id=0, l1_id=0, l2_id=0, verbose=1, output_name=None):
        """
        Train the bottleneck.
        
        Parameters
        ----------
        :param x_train, cluster_train: numpy 2D matrix and numpy 1D array, input and output training set
        :param x_test, cluster_test: numpy 2D matrix and numpy 1D array, input and output validation set
        :param epochs: int, # of training iterations
        :param bs: int, batch size
        :param patience: int, early stopping
        :param cvfold_id: int, cross validation set id (optional, default=0), for saving
        :param l1_id: int, lasso penalty id (optional, default=0), for saving
        :param l2_id: int, ridge penalty id (optional, default=0), for saving
        :param verbose: int, print info or not (optional, default=1, i.e. print info -- verbose=0 means not printing info)
        :param output_name: string, if provided we save file with the string in it (optional, default = None)
        
        Returns
        -------
        Training and validation loss R^2 score for all epochs
        """
        
        # Settings for early stopping and saving best model 
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=patience)
        mc = ModelCheckpoint('KerasSavedModels/Classification_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id), \
                             monitor='val_categorical_crossentropy', mode='min', verbose=verbose, save_best_only=True)

        # train the network
        print("[INFO] training network...")
        H = self.m.fit(x_train, to_categorical(cluster_train), batch_size=bs,
            validation_data=(x_test, to_categorical(cluster_test)),
            epochs=epochs, verbose=verbose, callbacks=[es, mc])
        
        if output_name is None:
            self.m.save('KerasSavedModels/Classification_last_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id))
        else:
            self.m.save('KerasSavedModels/{}/Classification_last_weights_{}_{}_{}.h5'.format(output_name, cvfold_id, l1_id, l2_id))
        
        # Retrieve activations and ephys prediction
        saved_model = load_model('KerasSavedModels/Classification_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id), \
                                 custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
        CE_train = saved_model.evaluate(x_train, to_categorical(cluster_train))[0]
        CE_test = saved_model.evaluate(x_test, to_categorical(cluster_test))[0]
        
        return CE_train, CE_test, np.array(H.history['categorical_crossentropy']), np.array(H.history['val_categorical_crossentropy'])

    
    
class StraightRegression:
    """
    Implements a bottleneck neural network for regression with keras that can use pre-trained weights.
    """
    
    def __init__(self, l1, l2, lr, act, input_dim, output_dim, pre_trained_weights = False, pre_trained_weights_h5 = None, \
                 nodes_list=[512, 128, 2, 128, 512]):
        """
        Constructor.
        :param l1: int, lasso penalty
        :param l2: int, ridge penalty
        :param lr: int, learning rate for Adam
        :param act: string, activation function
        :param input_dim: int, input layer dimensionality
        :param output_dim: int, output layer dimensionality
        :param pre_trained_weights: bool, if True we have pre-trained weights we could use for intialisation
                                    (optional, default=False)
        :param pre_trained_weights_h5: .h5 file, pre_trained weights to use as intial weights for training deep regression
                                       (optional, default=None)
        :param nodes_list: list, integers denoting # nodes in each layer (optional, default=[512, 128, 2, 128, 512])
        """
        
        self.l1=l1
        self.l2=l2
        self.lr=lr
        self.act=act
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.pre_trained_weights = pre_trained_weights
        self.nodes_list=nodes_list
        
        # Architecture
        keras.backend.clear_session()
        self.m = Sequential()
        self.m.add(Dense(self.nodes_list[0], activation=self.act, input_shape=(self.input_dim, ), \
            kernel_regularizer=ElasticNet(l1=self.l1, l2=self.l2), \
            bias_regularizer=regularizers.l2(self.l2), \
            name = 'W1'))
        self.m.add(Dense(self.nodes_list[1], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
            bias_regularizer=regularizers.l2(self.l2), \
            name = 'W2'))
        # linear for latent space representation
        self.m.add(Dense(self.nodes_list[2], activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
            bias_regularizer=regularizers.l2(self.l2), \
            name='bottleneck'))
        self.m.add(Dense(self.nodes_list[3], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
            bias_regularizer=regularizers.l2(self.l2), \
            name = 'W4'))
        self.m.add(Dense(self.nodes_list[4], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
            bias_regularizer=regularizers.l2(self.l2), \
            name = 'W5'))
        self.m.add(Dense(self.output_dim, activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
            #bias_regularizer=regularizers.l2(self.l2), \
            name = 'W6_regr'))
        if self.pre_trained_weights:
            self.m.load_weights(pre_trained_weights_h5, by_name=True) 
        self.m.compile(loss='mean_squared_error', \
                       optimizer=keras.optimizers.Adam(learning_rate=lr), \
                       metrics=[r2_score, 'mse'])
        
    def train(self, x_train, y_train, x_test, y_test, epochs, bs, patience, cvfold_id=0, l1_id=0, l2_id=0, verbose=1, \
              prune=False, geneNames=None):
        """
        Train the bottleneck. If you don't prune the training lasts for epochs iterations. If you prune the training lasts for 4*epochs
        iterations in total (2*epochs normal training, 2*epochs pruning).
        
        Parameters
        ----------
        :param x_train, y_train: numpy 2D matrices, input and output training set
        :param x_test, y_test: numpy 2D matrices, input and output validation set
        :param epochs: int, # of training iterations
        :param bs: int, bullshit, ah no, batch size
        :param patience: int, early stopping
        :param cvfold_id: int, cross validation set id (optional, default=0), for saving
        :param l1_id: int, lasso penalty id (optional, default=0), for saving
        :param l2_id: int, ridge penalty id (optional, default=0), for saving
        :param verbose: int, print info or not (optional, default=1, i.e. print info -- verbose=0 means not printing info)
        :param prune: bool, if True we additionally prune the network (new input layer with lower dimensionality)
                      (optional, default=False)
        :param geneNames: numpy 1D array, contains name of the corresponding gene of every input neuron (optional, default=None)
        
        Returns
        -------
        Training and validation loss R^2 score for all epochs
        """
        
        # Settings for early stopping and saving best model
        if not prune:
            es = EarlyStopping(monitor='val_mse', mode='min', verbose=verbose, patience=patience)
        else:
            es = EarlyStopping(monitor='val_mse', mode='min', verbose=verbose, patience=2*patience)
            
        if not self.pre_trained_weights:
            mc = ModelCheckpoint('KerasSavedModels/StraightRegression_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id), \
                             monitor='val_mse', mode='min', verbose=verbose, save_best_only=True)
        else:
            mc = ModelCheckpoint('KerasSavedModels/PreTrRegression_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id), \
                             monitor='val_mse', mode='min', verbose=verbose, save_best_only=True)

        # train the network
        print("[INFO] training network...")
        if not prune:
            H = self.m.fit(x_train, y_train, batch_size=bs,
                validation_data=(x_test, y_test),
                epochs=epochs, verbose=verbose, callbacks = [es, mc])
        else:
            H = self.m.fit(x_train, y_train, batch_size=bs,
                validation_data=(x_test, y_test),
                epochs=2*epochs, verbose=verbose, callbacks = [es, mc])           
        
        train_loss_straight_regr = np.array(H.history["r2_score"])
        val_loss_straight_regr = np.array(H.history["val_r2_score"])
        MSE_tr = np.array(H.history["loss"])
        MSE_val = np.array(H.history["val_loss"])
        
        
        # Retrieve activations and ephys prediction
        if not self.pre_trained_weights:
            saved_model = load_model('KerasSavedModels/StraightRegression_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id), \
                                     custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
        else:
            saved_model = load_model('KerasSavedModels/PreTrRegression_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id), \
                                     custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
        #print('predict: ', saved_model.predict(x_train))
        #print('y_train: ', y_train)
        #print('MSE: ', np.sum((y_train - saved_model.predict(x_train))**2))
        #print('denominator: ', np.sum(y_train**2))
        r2_train = 1-np.sum((y_train - saved_model.predict(x_train))**2) / np.sum(y_train**2)
        r2_test = 1-np.sum((y_test - saved_model.predict(x_test))**2) / np.sum(y_test**2)
        self.m.save('KerasSavedModels/Regression_last_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id))
    
        if prune:
            saved_model_ = load_model('KerasSavedModels/Regression_last_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id), \
                                     custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
            weight = saved_model_.get_weights()[0]
            bias = saved_model_.get_weights()[1]
            ind_genes = np.argsort(np.linalg.norm(weight, ord=2, axis=1))[-25:]
            print('The 25 genes that make it: ', geneNames[ind_genes])
            # Architecture
            keras.backend.clear_session()
            self.m = Sequential()
            self.m.add(Dense(self.nodes_list[0], activation=self.act, input_shape=(25, ), \
                        kernel_regularizer=ElasticNet(l1=0, l2=self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W1_25'))
            self.m.add(Dense(self.nodes_list[1], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W2'))
            # linear for latent space representation
            self.m.add(Dense(self.nodes_list[2], activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2),
                        name='bottleneck'))
            self.m.add(Dense(self.nodes_list[3], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W4'))
            self.m.add(Dense(self.nodes_list[4], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W5'))
            self.m.add(Dense(self.output_dim, activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
                        #bias_regularizer=regularizers.l2(l2_parameter), \
                        name = 'W6_regr'))
            # Load weights from training a previous network on regression
            self.m.load_weights('KerasSavedModels/Regression_last_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id), by_name=True)
            # We transfer the weights of those 25 genes now manually too
            self.m.get_layer('W1_25').set_weights([weight[ind_genes, :], bias])
            self.m.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.lr/2), metrics=[r2_score, 'mse'])

            es = EarlyStopping(monitor='val_mse', mode='min', verbose=verbose, patience=2*patience)
            
            if not self.pre_trained_weights:
                mc = ModelCheckpoint('KerasSavedModels/StraightRegression_weights_after_pruning_{}_{}_{}.h5'.\
                                     format(cvfold_id, l1_id, l2_id), \
                                     monitor='val_mse', mode='min', verbose=verbose, save_best_only=True)
            else:
                mc = ModelCheckpoint('KerasSavedModels/PreTrRegression_weights_after_pruning_{}_{}_{}.h5'.\
                                     format(cvfold_id, l1_id, l2_id), \
                                     monitor='val_mse', mode='min', verbose=verbose, save_best_only=True)                

            # train the network
            print("[INFO] training network...")
            H = self.m.fit(x_train[:, ind_genes], y_train, batch_size=bs,
                validation_data=(x_test[:, ind_genes], y_test),
                epochs=2*epochs, verbose=verbose, callbacks = [es, mc])
            
            MSE_tr = np.concatenate([MSE_tr, np.array(H.history["loss"])])
            MSE_val = np.concatenate([MSE_val, np.array(H.history["val_loss"])])
            train_loss_straight_regr = np.concatenate([train_loss_straight_regr, np.array(H.history["r2_score"])])
            val_loss_straight_regr = np.concatenate([val_loss_straight_regr, np.array(H.history["val_r2_score"])])
            
        if prune:
            if not self.pre_trained_weights:
                saved_model_2 = load_model('KerasSavedModels/StraightRegression_weights_after_pruning_{}_{}_{}.h5'.\
                         format(cvfold_id, l1_id, l2_id), \
                         custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
            else:
                saved_model_2 = load_model('KerasSavedModels/PreTrRegression_weights_after_pruning_{}_{}_{}.h5'.\
                         format(cvfold_id, l1_id, l2_id), \
                         custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet, 'residual': residual})
            r2_after_pruning_train = 1-np.sum((y_train - saved_model_2.predict(x_train[:, ind_genes]))**2) \
                                                                            / np.sum(y_train**2)
            r2_after_pruning_test = 1-np.sum((y_test - saved_model_2.predict(x_test[:, ind_genes]))**2) \
                                                                            / np.sum(y_test**2)
        
        if not prune:
            print('Train R^2: ', r2_train)
            print('Test R^2: ', r2_test)
        else:
            print('Train R^2 before pruning: ', r2_train)
            print('Test R^2 after pruning: ', r2_test)            
            print('Train R^2 after pruning: ', r2_after_pruning_train)
            print('Test R^2 after pruning: ', r2_after_pruning_test)

        
        if not prune:
            return  saved_model.predict(x_train), saved_model.predict(x_test), \
                    r2_train, r2_test, train_loss_straight_regr, val_loss_straight_regr, \
                    MSE_tr, MSE_val
        else:
            return r2_train, r2_test, \
                   r2_after_pruning_train, r2_after_pruning_test, \
                   train_loss_straight_regr, val_loss_straight_regr
    
class FreezeUnfreeze:
    """
    Implements a bottleneck neural network with Keras that can use pre-trained weights and first freezes certain layers before training
    all layers.
    """
    
    def __init__(self, l1, l2, lr, act, input_dim, output_dim, unfreeze, pre_trained_weights = False, pre_trained_weights_h5 = None, \
                 nodes_list=[512, 128, 2, 128, 512]):
        """
        Constructor.
        :param l1: int, lasso penalty
        :param l2: int, ridge penalty
        :param lr: int, learning rate for Adam
        :param act: string, activation function
        :param input_dim: int, input layer dimensionality
        :param output_dim: int, output layer dimensionality
        :param unfreeze: list of bools, if element in list is True that index corresponds to a layer that can be trained
        :param pre_trained_weights: bool, if True we have pre-trained weights we could use for intialisation
                                    (optional, default=False)
        :param pre_trained_weights_h5: .h5 file, pre_trained weights to use as intial weights for training deep regression
                                       (optional, default=None)
        :param nodes_list: list, integers denoting # nodes in each layer (optional, default=[512, 128, 2, 128, 512])
        """
        
        self.l1=l1
        self.l2=l2
        self.lr=lr
        self.act=act
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.unfreeze = unfreeze
        self.pre_trained_weights = pre_trained_weights
        self.nodes_list=nodes_list
        
        # Architecture
        keras.backend.clear_session()
        self.m = Sequential()
        self.m.add(Dense(self.nodes_list[0], activation=self.act, input_shape=(self.input_dim, ), \
                    kernel_regularizer=ElasticNet(l1=self.l1, l2=self.l2), \
                    bias_regularizer=regularizers.l2(self.l2), \
                    trainable = self.unfreeze[0], name = 'W1'))
        self.m.add(Dense(self.nodes_list[1], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                    bias_regularizer=regularizers.l2(self.l2), \
                    trainable = self.unfreeze[1], name = 'W2'))
        # linear for latent space representation
        self.m.add(Dense(self.nodes_list[2], activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
                    bias_regularizer=regularizers.l2(self.l2),
                    trainable = self.unfreeze[2], name='bottleneck'))
        self.m.add(Dense(self.nodes_list[3], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                    bias_regularizer=regularizers.l2(self.l2), \
                    trainable = self.unfreeze[3], name = 'W4'))
        self.m.add(Dense(self.nodes_list[4], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                    bias_regularizer=regularizers.l2(self.l2), \
                    trainable = self.unfreeze[4], name = 'W5'))
        self.m.add(Dense(self.output_dim, activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
                    trainable = self.unfreeze[5], \
                    #bias_regularizer=regularizers.l2(l2_parameter), \
                    name = 'W6_regr'))
        
        # Load weights from training a previous network on classification
        if self.pre_trained_weights:
            self.m.load_weights(pre_trained_weights_h5, by_name=True) 
        self.m.compile(loss='mean_squared_error', \
                       optimizer=keras.optimizers.Adam(learning_rate=lr), \
                       metrics=[r2_score, 'mse'])
        
    def train(self, x_train, y_train, x_test, y_test, epochs, bs, patience, cvfold_id=0, l1_id=0, l2_id=0, verbose=1, \
              prune=False, geneNames=None, citeseq=False, report_individual_ephys_feature_test_R2=False):
        """
        Train the bottleneck. If you don't prune the training lasts for 2*epochs iterations (freezing+unfreezing). If you prune the
        training lasts for 4*epochs (freezing (epochs) + unfreezing (epochs) + pruning (2*epochs)) iterations in total.
        
        Parameters
        ----------
        :param x_train, y_train: numpy 2D matrices, input and output training set
        :param x_test, y_test: numpy 2D matrices, input and output validation set
        :param epochs: int, # of training iterations
        :param bs: int, batch size
        :param patience: int, early stopping
        :param cvfold_id: int, cross validation set id (optional, default=0), for saving
        :param l1_id: int, lasso penalty id (optional, default=0), for saving
        :param l2_id: int, ridge penalty id (optional, default=0), for saving
        :param verbose: int, print info or not (optional, default=1, i.e. print info -- verbose=0 means not printing info)
        :param prune: bool, if True we additionally prune the network (new input layer with lower dimensionality)
                      (optional, default=False)
        :param geneNames: numpy 1D array, contains name of the corresponding gene of every input neuron (optional, default=None)
        :param citeseq: bool, if True we're performing the training procedure on a CITE-seq dataset and will use the best epoch
                        during unfreezing for use afterwards when we prune (optional, default=False)
        :param report_individual_ephys_feature_test_R2: bool, if True save test R^2 scores for every individual ephys feature
        
        Returns
        -------
        Training and validation loss R^2 score for all epochs
        """
        
        # Settings for early stopping and saving best model 
        es = EarlyStopping(monitor='val_mse', mode='min', verbose=verbose, patience=patience)
        if not self.pre_trained_weights:
            mc = ModelCheckpoint('KerasSavedModels/FreezeUnfreeze_weights_before_unfreezing_{}_{}_{}.h5'.\
                                 format(cvfold_id, l1_id, l2_id), \
                                 monitor='val_mse', mode='min', verbose=verbose, save_best_only=True)
        else:
            mc = ModelCheckpoint('KerasSavedModels/PreTrFreezeUnfreeze_weights_before_unfreezing_{}_{}_{}.h5'.\
                                 format(cvfold_id, l1_id, l2_id), \
                                 monitor='val_mse', mode='min', verbose=verbose, save_best_only=True)

        # train the network
        print("[INFO] training network...")
        H = self.m.fit(x_train, y_train, batch_size=bs,
            validation_data=(x_test, y_test),
            epochs=epochs, verbose=verbose, callbacks = [es, mc])
        train_loss_freeze_unfreeze = np.array(H.history["r2_score"])
        val_loss_freeze_unfreeze = np.array(H.history["val_r2_score"])
        
        
        # Now UNFREEZE all layers
        for layer in self.m.layers:
            layer.trainable = True
        if verbose!=0:
            for layer in self.m.layers:
                print(layer, 'trainable?', layer.trainable)
            
        # Since we’ve unfrozen additional layers, we must re-compile the model and let us decrease the learning rate by a half
        self.m.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=self.lr/2), metrics=[r2_score, 'mse'])
            
        # Settings for early stopping and saving best model 
        es = EarlyStopping(monitor='val_mse', mode='min', verbose=1, patience=patience)
        if not self.pre_trained_weights:
            mc = ModelCheckpoint('KerasSavedModels/FreezeUnfreeze_weights_after_unfreezing_{}_{}_{}.h5'.\
                                 format(cvfold_id, l1_id, l2_id), \
                                 monitor='val_mse', mode='min', verbose=verbose, save_best_only=True)
        else:
            mc = ModelCheckpoint('KerasSavedModels/PreTrFreezeUnfreeze_weights_after_unfreezing_{}_{}_{}.h5'.\
                                 format(cvfold_id, l1_id, l2_id), \
                                 monitor='val_mse', mode='min', verbose=verbose, save_best_only=True)
        
        
        # train the network again
        print("[INFO] training network...")
        H = self.m.fit(x_train, y_train, batch_size=bs,
            validation_data=(x_test, y_test),
            epochs=epochs, verbose=verbose, callbacks = [es, mc])
        train_loss_freeze_unfreeze = np.concatenate([train_loss_freeze_unfreeze, np.array(H.history["r2_score"])])
        val_loss_freeze_unfreeze = np.concatenate([val_loss_freeze_unfreeze, np.array(H.history["val_r2_score"])])
        
        # Retrieve activations and ephys prediction
        if not self.pre_trained_weights:
            saved_model = load_model('KerasSavedModels/FreezeUnfreeze_weights_before_unfreezing_{}_{}_{}.h5'.\
                                     format(cvfold_id, l1_id, l2_id), \
                                     custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
            saved_model_2 = load_model('KerasSavedModels/FreezeUnfreeze_weights_after_unfreezing_{}_{}_{}.h5'.\
                                     format(cvfold_id, l1_id, l2_id), \
                                     custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
        else:
            saved_model = load_model('KerasSavedModels/PreTrFreezeUnfreeze_weights_before_unfreezing_{}_{}_{}.h5'.\
                                     format(cvfold_id, l1_id, l2_id), \
                                     custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
            saved_model_2 = load_model('KerasSavedModels/PreTrFreezeUnfreeze_weights_after_unfreezing_{}_{}_{}.h5'.\
                                     format(cvfold_id, l1_id, l2_id), \
                                     custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
        
        r2_before_unfreezing_train = 1-np.sum((y_train - saved_model.predict(x_train))**2) / np.sum(y_train**2)
        r2_before_unfreezing_test = 1-np.sum((y_test - saved_model.predict(x_test))**2) / np.sum(y_test**2)
        r2_after_unfreezing_train = 1-np.sum((y_train - saved_model_2.predict(x_train))**2) / np.sum(y_train**2)
        r2_after_unfreezing_test = 1-np.sum((y_test - saved_model_2.predict(x_test))**2) / np.sum(y_test**2)   
        self.m.save('KerasSavedModels/FreezeUnfreeze_last_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id))
        if prune:
            
            saved_model_ = load_model('KerasSavedModels/FreezeUnfreeze_last_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id), \
                                         custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
            
            weight = saved_model_.get_weights()[0]
            bias = saved_model_.get_weights()[1]
            ind_genes = np.argsort(np.linalg.norm(weight, ord=2, axis=1))[-25:]
            print('The 25 genes that make it: ', geneNames[ind_genes])
            # Architecture
            keras.backend.clear_session()
            self.m = Sequential()
            self.m.add(Dense(self.nodes_list[0], activation=self.act, input_shape=(25, ), \
                        kernel_regularizer=ElasticNet(l1=0, l2=self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W1_25'))
            self.m.add(Dense(self.nodes_list[1], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W2'))
            # linear for latent space representation
            self.m.add(Dense(self.nodes_list[2], activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2),
                        name='bottleneck'))
            self.m.add(Dense(self.nodes_list[3], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W4'))
            self.m.add(Dense(self.nodes_list[4], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W5'))
            self.m.add(Dense(self.output_dim, activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
                        #bias_regularizer=regularizers.l2(l2_parameter), \
                        name = 'W6_regr'))
            # Load weights from training a previous network on regression
            self.m.load_weights('KerasSavedModels/FreezeUnfreeze_last_weights_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id), by_name=True)
            # We transfer the weights of those 25 genes now manually too
            self.m.get_layer('W1_25').set_weights([weight[ind_genes, :], bias])
            self.m.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.lr/2), metrics=[r2_score, 'mse'])

            es = EarlyStopping(monitor='val_mse', mode='min', verbose=verbose, patience=2*patience)
            
            if not self.pre_trained_weights:
                mc = ModelCheckpoint('KerasSavedModels/FreezeUnfreeze_weights_after_unfreezing_and_pruning_{}_{}_{}.h5'.\
                                     format(cvfold_id, l1_id, l2_id), \
                                     monitor='val_mse', mode='min', verbose=verbose, save_best_only=True)
            else:
                mc = ModelCheckpoint('KerasSavedModels/PreTrFreezeUnfreeze_weights_after_unfreezing_and_pruning_{}_{}_{}.h5'.\
                                     format(cvfold_id, l1_id, l2_id), \
                                     monitor='val_mse', mode='min', verbose=verbose, save_best_only=True)                

            # train the network
            print("[INFO] training network...")
            H = self.m.fit(x_train[:, ind_genes], y_train, batch_size=bs,
                validation_data=(x_test[:, ind_genes], y_test),
                epochs=2*epochs, verbose=verbose, callbacks = [es, mc])
            train_loss_freeze_unfreeze = np.concatenate([train_loss_freeze_unfreeze, np.array(H.history["r2_score"])])
            val_loss_freeze_unfreeze = np.concatenate([val_loss_freeze_unfreeze, np.array(H.history["val_r2_score"])])
        
        if prune:
            if not self.pre_trained_weights:
                saved_model_3 = load_model('KerasSavedModels/FreezeUnfreeze_weights_after_unfreezing_and_pruning_{}_{}_{}.h5'.\
                         format(cvfold_id, l1_id, l2_id), \
                         custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
            else:
                saved_model_3 = load_model('KerasSavedModels/PreTrFreezeUnfreeze_weights_after_unfreezing_and_pruning_{}_{}_{}.h5'.\
                         format(cvfold_id, l1_id, l2_id), \
                         custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
            r2_after_unfreezing_and_pruning_train = 1-np.sum((y_train - saved_model_3.predict(x_train[:, ind_genes]))**2) \
                                                                            / np.sum(y_train**2)
            r2_after_unfreezing_and_pruning_test = 1-np.sum((y_test - saved_model_3.predict(x_test[:, ind_genes]))**2) \
                                                                            / np.sum(y_test**2)
            if report_individual_ephys_feature_test_R2:
                if not citeseq:
                    if self.nodes_list[2]==2:
                        np.savez('KerasSavedModels/individual_ephys_feature_test_R2_{}_{}_{}.npz'.format(cvfold_id, l1_id, l2_id), \
                                 R2=np.array([1-np.sum((y_test[:,i] - saved_model_3.predict(x_test[:, ind_genes])[:,i])**2) \
                                                                  / np.sum(y_test[:,i]**2) for i in range(y_test.shape[1])])
                                )
                    else:
                        np.savez('KerasSavedModels/individual_ephys_feature_test_R2_{}_{}_{}_nb.npz'.format(cvfold_id, l1_id, l2_id), \
                                 R2=np.array([1-np.sum((y_test[:,i] - saved_model_3.predict(x_test[:, ind_genes])[:,i])**2) \
                                                                  / np.sum(y_test[:,i]**2) for i in range(y_test.shape[1])])
                                )                        
                else:
                    if self.nodes_list[2]==2:
                        np.savez('KerasSavedModels/individual_ephys_feature_test_R2_{}_{}_{}_citeseq.npz'.format(cvfold_id, l1_id, l2_id), \
                                 R2=np.array([1-np.sum((y_test[:,i] - saved_model_3.predict(x_test[:, ind_genes])[:,i])**2) \
                                                                  / np.sum(y_test[:,i]**2) for i in range(y_test.shape[1])])
                                )
                    else:
                        np.savez('KerasSavedModels/individual_ephys_feature_test_R2_{}_{}_{}_citeseq_nb.npz'.format(cvfold_id, l1_id, \
                                                                                                                    l2_id), \
                                 R2=np.array([1-np.sum((y_test[:,i] - saved_model_3.predict(x_test[:, ind_genes])[:,i])**2) \
                                                                  / np.sum(y_test[:,i]**2) for i in range(y_test.shape[1])])
                                )                        

        print('Train R^2 before unfreezing: ', r2_before_unfreezing_train)
        print('Test R^2 before unfreezing: ', r2_before_unfreezing_test)
        print('Train R^2 after unfreezing: ', r2_after_unfreezing_train)
        print('Test R^2 after unfreezing: ', r2_after_unfreezing_test)
        
        if prune:
            print('Train R^2 after unfreezing and pruning: ', r2_after_unfreezing_and_pruning_train)
            print('Test R^2 after unfreezing and pruning: ', r2_after_unfreezing_and_pruning_test)
        
        if not prune:
            return r2_before_unfreezing_train, r2_before_unfreezing_test, \
                   r2_after_unfreezing_train, r2_after_unfreezing_test, \
                   train_loss_freeze_unfreeze, val_loss_freeze_unfreeze
    
        else:
            return r2_before_unfreezing_train, r2_before_unfreezing_test, \
                   r2_after_unfreezing_train, r2_after_unfreezing_test, \
                   r2_after_unfreezing_and_pruning_train, r2_after_unfreezing_and_pruning_test, \
                   train_loss_freeze_unfreeze, val_loss_freeze_unfreeze
    
    def train_full_dataset(self, x_train, y_train, epochs, bs, patience, cvfold_id=0, l1_id=0, l2_id=0, verbose=1, \
              prune=False, geneNames=None, add_autoencoder=False, output_name=None):
        """
        Train the bottleneck for the full dataset (no validation). If you don't prune the training lasts for 2*epochs iterations
        (freezing+unfreezing). If you prune the training lasts for 4*epochs (freezing (epochs) + unfreezing (epochs) + pruning (2*epochs))
        iterations in total.
        
        Parameters
        ----------
        :param x_train, y_train: numpy 2D matrices, input and output training set
        :param epochs: int, # of training iterations
        :param bs: int, batch size
        :param patience: int, early stopping
        :param cvfold_id: int, cross validation set id (optional, default=0)
        :param l1_id: int, lasso penalty id (optional, default=0)
        :param l2_id: int, ridge penalty id (optional, default=0)
        :param verbose: int, print info or not (optional, default=1, i.e. print info -- verbose=0 means not printing info)
        :param prune: bool, if True we additionally prune the network (new input layer with lower dimensionality)
                      (optional, default=False)
        :param geneNames: numpy 1D array, contains name of the corresponding gene of every input neuron (optional, default=None)
        :param add_autoencoder: bool, if True we add an autoencoder to selected genes from the bottleneck layer
        :param output_name: string, if provided we save file with the string in it (optional, default = None)
        
        Returns
        -------
        Training loss R^2 score for all epochs
        
        """

        # train the network
        print("[INFO] training network...")
        H = self.m.fit(x_train, y_train, batch_size=bs,
            epochs=epochs, verbose=verbose)
        train_loss_freeze_unfreeze = np.array(H.history["r2_score"])
        
        
        # Now UNFREEZE all layers
        for layer in self.m.layers:
            layer.trainable = True
        if verbose!=0:
            for layer in self.m.layers:
                print(layer, 'trainable?', layer.trainable)
            
        # Since we’ve unfrozen additional layers, we must re-compile the model and let us decrease the learning rate by a half
        self.m.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=self.lr/2), metrics=[r2_score, 'mse'])
        
        if output_name is None:
            self.m.save('KerasSavedModels/FreezeUnfreeze_before_unfreezing_full_dataset_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id))
        else:
            self.m.save('KerasSavedModels/{}/FreezeUnfreeze_before_unfreezing_full_dataset_{}_{}_{}.h5'.format\
                                                                         (output_name, cvfold_id, l1_id, l2_id))
        
        # train the network again
        print("[INFO] training ephys prediction network...")
        H = self.m.fit(x_train, y_train, batch_size=bs,
            epochs=epochs, verbose=verbose)
        train_loss_freeze_unfreeze = np.concatenate([train_loss_freeze_unfreeze, np.array(H.history["r2_score"])]) 
        
        if output_name is None:
            self.m.save('KerasSavedModels/FreezeUnfreeze_after_unfreezing_full_dataset_{}_{}_{}.h5'.format(cvfold_id, l1_id, l2_id))
        else:
            self.m.save('KerasSavedModels/{}/FreezeUnfreeze_after_unfreezing_full_dataset_{}_{}_{}.h5'.format\
                                                                          (output_name, cvfold_id, l1_id, l2_id))
        
        if prune:
            if output_name is None:
                saved_model_ = load_model('KerasSavedModels/FreezeUnfreeze_after_unfreezing_full_dataset_{}_{}_{}.h5'.\
                                      format(cvfold_id, l1_id, l2_id), \
                                      custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
            else:
                saved_model_ = load_model('KerasSavedModels/{}/FreezeUnfreeze_after_unfreezing_full_dataset_{}_{}_{}.h5'.\
                                      format(output_name, cvfold_id, l1_id, l2_id), \
                                      custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})                
            weight = saved_model_.get_weights()[0]
            bias = saved_model_.get_weights()[1]
            ind_genes = np.argsort(np.linalg.norm(weight, ord=2, axis=1))[-25:]
            print('The 25 genes that make it: ', geneNames[ind_genes])
            
            # Architecture
            keras.backend.clear_session()
            self.m = Sequential()
            self.m.add(Dense(self.nodes_list[0], activation=self.act, input_shape=(25, ), \
                        kernel_regularizer=ElasticNet(l1=0, l2=self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W1_25'))
            self.m.add(Dense(self.nodes_list[1], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W2'))
            # linear for latent space representation
            self.m.add(Dense(self.nodes_list[2], activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2),
                        name='bottleneck'))
            self.m.add(Dense(self.nodes_list[3], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W4'))
            self.m.add(Dense(self.nodes_list[4], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                        bias_regularizer=regularizers.l2(self.l2), \
                        name = 'W5'))
            self.m.add(Dense(self.output_dim, activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
                        #bias_regularizer=regularizers.l2(l2_parameter), \
                        name = 'W6_regr'))
            # Load weights from training a previous network on regression
            if output_name is None:
                self.m.load_weights('KerasSavedModels/FreezeUnfreeze_after_unfreezing_full_dataset_{}_{}_{}.h5'.\
                                    format(cvfold_id, l1_id, l2_id), by_name=True)
            else:
                self.m.load_weights('KerasSavedModels/{}/FreezeUnfreeze_after_unfreezing_full_dataset_{}_{}_{}.h5'.\
                                    format(output_name, cvfold_id, l1_id, l2_id), by_name=True)            
            # We transfer the weights of those 25 genes now manually too
            self.m.get_layer('W1_25').set_weights([weight[ind_genes, :], bias])
            self.m.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.lr/2), metrics=[r2_score, 'mse'])             
            
            # train the network
            print("[INFO] training pruning network...")
            H=self.m.fit(x_train[:, ind_genes], y_train, batch_size=bs,
                epochs=2*epochs, verbose=verbose)
            if output_name is None:
                self.m.save('KerasSavedModels/FreezeUnfreeze_after_unfreezing_ap_full_dataset_{}_{}_{}.h5'.format\
                            (cvfold_id, l1_id, l2_id))
            else:
                self.m.save('KerasSavedModels/{}/FreezeUnfreeze_after_unfreezing_ap_full_dataset_{}_{}_{}.h5'.format\
                            (output_name, cvfold_id, l1_id, l2_id))                
            train_loss_freeze_unfreeze=np.concatenate([train_loss_freeze_unfreeze, np.array(H.history["r2_score"])])
            
            
            if add_autoencoder:
                # The latent space is the same, but instead of predicting ephys we want to predict selected genes. This leads us to
                # latent space visualisations that can be overlayed with gene model predictions.
                
                # Retrieve bottleneck activations
                if output_name is None:
                    saved_model_AE = load_model('KerasSavedModels/FreezeUnfreeze_after_unfreezing_ap_full_dataset_{}_{}_{}.h5'.\
                                                format(cvfold_id, l1_id, l2_id), \
                                                custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})
                else:
                    saved_model_AE = load_model('KerasSavedModels/{}/FreezeUnfreeze_after_unfreezing_ap_full_dataset_{}_{}_{}.h5'.\
                                                format(output_name, cvfold_id, l1_id, l2_id), \
                                                custom_objects={'r2_score': r2_score, 'ElasticNet': ElasticNet})                
                encoder = Model(saved_model_AE.input, saved_model_AE.get_layer('bottleneck').output)
                latent = encoder.predict(x_train[:, ind_genes])
                
                # Retrieve weights
                W4=saved_model_AE.get_weights()[6]
                W4_bias=saved_model_AE.get_weights()[7]
                W5=saved_model_AE.get_weights()[8]
                W5_bias=saved_model_AE.get_weights()[9]
                
                # Architecture
                keras.backend.clear_session()
                self.m = Sequential()
                self.m.add(Dense(self.nodes_list[3], activation=self.act, input_shape=(self.nodes_list[2], ), \
                            kernel_regularizer=regularizers.l2(self.l2), \
                            bias_regularizer=regularizers.l2(self.l2), \
                            name='W4'))
                self.m.add(Dense(self.nodes_list[4], activation=self.act, kernel_regularizer=regularizers.l2(self.l2), \
                            bias_regularizer=regularizers.l2(self.l2), \
                            name='W5'))
                self.m.add(Dense(25, activation='linear', kernel_regularizer=regularizers.l2(self.l2), \
                            #bias_regularizer=regularizers.l2(self.l2), \
                            name='W6_regr_AE'))
                # Set initial weights
                self.m.get_layer('W4').set_weights([W4, W4_bias])
                self.m.get_layer('W5').set_weights([W5, W5_bias])
                self.m.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.lr/2), metrics=[r2_score, 'mse'])
                
                # train the network
                print("[INFO] training autoencoder network")
                H=self.m.fit(latent, x_train[:,ind_genes], batch_size=bs,
                             epochs=2*epochs, verbose=verbose)
                if output_name is None:
                    self.m.save('KerasSavedModels/FreezeUnfreeze_after_unfreezing_ap_full_dataset_AE_{}_{}_{}.h5'\
                                .format(cvfold_id, l1_id, l2_id))
                else:
                    self.m.save('KerasSavedModels/{}/FreezeUnfreeze_after_unfreezing_ap_full_dataset_AE_{}_{}_{}.h5'\
                                .format(output_name, cvfold_id, l1_id, l2_id))                    
                
                train_loss_freeze_unfreeze_AE=np.array(H.history["r2_score"])
                
            if not add_autoencoder:     
                return train_loss_freeze_unfreeze
            else: return train_loss_freeze_unfreeze, train_loss_freeze_unfreeze_AE
 
    
    
