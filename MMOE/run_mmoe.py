"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""

import random

import pandas as pd
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

from mmoe import *
SEED = 1

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
# tf.random.set_seed(SEED)
tf.set_random_seed(SEED)

# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            train_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            val_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            print('\n')
            print('{}_AUC: train:{} val:{} test:{}' .format(output_name,round(train_auc, 4),round(val_auc, 4),round(test_auc,4)))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def main():
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # Load the dataset in Pandas
    train_df = pd.read_csv('data/census-income.data.gz',delimiter=',',header=None,index_col=None,names=column_names)
    test_df = pd.read_csv('data/census-income.test.gz',delimiter=',',header=None,index_col=None,names=column_names)
    data=pd.concat([train_df,test_df],axis=0)
    data['marital_stat']=data['marital_stat'].apply(lambda x : 0 if x==' Never married' else 1)
    data['income_50k']=data['income_50k'].apply(lambda x : 0 if x==' - 50000.' else 1)

    dense_feat=['age','det_ind_code','det_occ_code','wage_per_hour','capital_gains','capital_losses',
                'stock_dividends','instance_weight','weeks_worked']
    target=['marital_stat','income_50k']
    cate_feat=[i for i in column_names if i not in dense_feat and i not in target]
    print(cate_feat,len(cate_feat))

    data[cate_feat] = data[cate_feat].fillna('-1', )
    data[dense_feat] = data[dense_feat].fillna(0, )

    # label Encoding for sparse features,and do simple Transformation for dense features
    for feat in cate_feat:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_feat] = mms.fit_transform(data[dense_feat])
    
    # count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)for i, feat in enumerate(cate_feat)] \
    + [DenseFeat(feat, 1, ) for feat in dense_feat]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train,test=data.iloc[:199523,:],data.iloc[199523:,:]
    x_train,x_test=train[cate_feat+dense_feat],test[cate_feat+dense_feat]
    y_train,y_test=train[target],test[target]
    x_test,x_val,y_test,y_val= train_test_split(x_test,y_test, test_size=0.5, random_state=2021)
    print(x_train.shape,y_train.shape,x_val.shape,y_val.shape,x_test.shape,y_test.shape)

    train_model_input = {name: x_train[name] for name in feature_names}
    val_model_input = {name: x_val[name] for name in feature_names}
    test_model_input = {name: x_test[name] for name in feature_names}

    train_labels = [y_train[col_name].values for col_name in target]
    val_labels = [y_val[col_name].values for col_name in target]
    test_labels= [y_test[col_name].values for col_name in target]

    # 4.Define Model,train,predict and evaluate
    model = MMOE(dnn_feature_columns, num_tasks=2, expert_dim=8, dnn_hidden_units=(128, 128),
                       tasks=['binary', 'binary'])
    
    model.compile(loss={'marital_stat': 'binary_crossentropy','income': 'binary_crossentropy',},optimizer=Adam(),metrics=['accuracy'])

    print(train_labels,train_labels[0])
    print(model.summary())
    model.fit(x=train_model_input,
              y=train_labels,
              validation_data=(val_model_input,val_labels),
              callbacks=[ROCCallback(training_data=(train_model_input, train_labels),
                                     validation_data=(val_model_input,val_labels),
                                     test_data=(test_model_input,test_labels))],
              epochs=30)


if __name__ == '__main__':
    main()
