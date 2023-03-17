# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import argparse
import datetime
from pathlib import Path
import yaml
import pandas as pd
import mltable
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--train_data", default="../../training-mltable-folder/train_data.csv", type=str)
    parser.add_argument("--validation_data", default="../../validation-mltable-folder/validation_data.csv", type=str)
    parser.add_argument("--preprocessed_train_data", default="../../training-mltable-folder", type=str)
    parser.add_argument("--preprocessed_validation_data", default="../../validation-mltable-folder", type=str)
    parser.add_argument("--target_column", default="Class", type=str)
    # parse args
    args = parser.parse_args()
    print("args received ", args)
    # return args
    return args

def preprocess_smote(dataframe, target_column):
    
    # split into x and y for SMOTE
    x = dataframe.drop(columns=target_column)
    y = dataframe[target_column]

    # perform SMOTE (look into SMOTEnn to check for whether over and under sampling would be more effective)
    sm = SMOTE(random_state = 2)
    x, y = sm.fit_resample(x, y.ravel())

    # Convert back to dataframe
    dataframe = pd.DataFrame.from_records(x)
    dataframe[target_column] = y

    return dataframe


def apply_business_logic(dataframe, fit_scaler):
    
    # normalise the amount column
    dataframe['normAmount'] = fit_scaler.transform(np.array(dataframe['Amount']).reshape(-1, 1))

    # drop Time and Amount columns as they are not relevant for prediction purpose 
    dataframe = dataframe.drop(columns=['Time', 'Amount'], axis = 1)

    return dataframe


def main(args):
    """
    Preprocessing of training/validation data
    """

    # Create path to read in validation data that will be preprocessed
    train_data_path = os.path.join(
        args.train_data, "train_data.csv"
    )

    # Read in training data and pass it to the get_preprocessed_data function to be preprocessed
    train_dataframe = pd.read_csv(train_data_path)

    # Fit the scaler to the training data for use on both training and validation data
    fit_scaler = StandardScaler().fit(np.array(train_dataframe['Amount']).reshape(-1, 1))

    train_dataframe = apply_business_logic(train_dataframe, fit_scaler=fit_scaler)
    preprocessed_train_dataframe = preprocess_smote(train_dataframe, args.target_column)

    # Inverse_transform before writing back to df to have data back in original non-normalized format
    preprocessed_train_dataframe['Amount'] = fit_scaler.inverse_transform(np.array(preprocessed_train_dataframe['normAmount']).reshape(-1, 1))
    preprocessed_train_dataframe = preprocessed_train_dataframe.drop(columns=[ 'normAmount'], axis = 1)

    # Write preprocessed train data in output path
    preprocessed_train_data_path = os.path.join(
        args.preprocessed_train_data, "preprocessed_train_data.csv"
    )
    preprocessed_train_dataframe.to_csv(
        preprocessed_train_data_path,
        index=False,
        header=True,
    )

    # Create path to read in validation data that will be preprocessed
    validation_data_path = os.path.join(
        args.validation_data, "validation_data.csv"
    )

    # Read in validation data and pass it to the get_preprocessed_data function to be preprocessed
    validation_dataframe = pd.read_csv(validation_data_path)
    preprocessed_validation_dataframe = apply_business_logic(validation_dataframe, fit_scaler=fit_scaler)

    # Inverse_transform before writing back to df to have data back in original non-normalized format
    preprocessed_validation_dataframe['Amount'] = fit_scaler.inverse_transform(np.array(preprocessed_validation_dataframe['normAmount']).reshape(-1, 1))
    preprocessed_validation_dataframe = preprocessed_validation_dataframe.drop(columns=[ 'normAmount'], axis = 1)

    # Write preprocessed validation data in output path
    preprocessed_validation_data_path = os.path.join(
        args.preprocessed_validation_data, "preprocessed_validation_data.csv"
    )
    preprocessed_validation_dataframe.to_csv(
        preprocessed_validation_data_path,
        index=False,
        header=True,
    )

    # Write training MLTable yaml for AutoML use
    preprocessed_train_data_mltable_path = os.path.join(
        args.preprocessed_train_data, "MLTable"
    )
    
    train_yaml_file = dict(
        paths = [dict(
            file = './preprocessed_train_data.csv'
        )]
        ,transformations = [dict(
            read_delimited = dict(
                delimiter = ','
                ,encoding = 'utf8'
            )
        )]
    )

    with open(preprocessed_train_data_mltable_path, "w") as file:
         yaml.dump(train_yaml_file, file)

    preprocessed_validation_data_mltable_path = os.path.join(
        args.preprocessed_validation_data, "MLTable"
    )


    # Write validation MLTable yaml for AutoML use
    validation_yaml_file = dict(
        paths = [dict(
            file = './preprocessed_validation_data.csv'
        )]
        ,transformations = [dict(
            read_delimited = dict(
                delimiter = ','
                ,encoding = 'utf8'
            )
        )]
    )

    with open(preprocessed_validation_data_mltable_path, "w") as file:
         yaml.dump(validation_yaml_file, file)


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
