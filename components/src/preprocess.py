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


def get_preprocessed_data(dataframe, target_column):
    
    # normalise the amount column
    dataframe['normAmount'] = StandardScaler().fit_transform(np.array(dataframe['Amount']).reshape(-1, 1))
    print('df normalized')

    # drop Time and Amount columns as they are not relevant for prediction purpose 
    dataframe = dataframe.drop(columns=['Time', 'Amount'], axis = 1)
    print('time and amount dropped')

    # split into x and y for SMOTE
    x = dataframe.drop(columns=target_column)
    y = dataframe[target_column]
    print('dataset split')

    # perform SMOTE
    sm = SMOTE(random_state = 2)
    x, y = sm.fit_resample(x, y.ravel())
    print('SMOTE complete')

    # Convert back to dataframe
    dataframe = pd.DataFrame.from_records(x)
    dataframe[target_column] = y
    print("converted to df")

    #Remerge dataframes for main function to right to files
    #dataframe = pd.merge(x,y, left_index=True, right_index=True)
    #print('DF remerged')

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
    #train_dataframe = pd.read_csv(args.train_data)
    preprocessed_train_dataframe = get_preprocessed_data(train_dataframe, args.target_column)

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
    #validation_dataframe = pd.read_csv(args.validation_data)
    preprocessed_validation_dataframe = get_preprocessed_data(validation_dataframe, args.target_column)

    # Write preprocessed validation data in output path
    preprocessed_validation_data_path = os.path.join(
        args.preprocessed_validation_data, "preprocessed_validation_data.csv"
    )
    preprocessed_validation_dataframe.to_csv(
        preprocessed_validation_data_path,
        index=False,
        header=True,
    )

    # Write MLTable yaml file as well in output folder
    # This example reads and copies an existing yaml, but in this case we use csv files until the data is preprocessed

    # read and write MLModel yaml file for train data

    # preprocessed_train_data_mltable_path = [{'file':preprocessed_train_data_path}] 
    # preprocessed_train_data_mltable = mltable.from_delimited_files(paths=preprocessed_train_data_mltable_path)
    # preprocessed_train_data_mltable.save(os.path.join(args.preprocessed_train_data))
    # print(os.path.join(preprocessed_train_data_path, "MLTable.yml"))

    # preprocessed_validation_data_mltable_path = [{'file':preprocessed_validation_data_path}]
    # preprocessed_validation_data_mltable = mltable.from_delimited_files(paths=preprocessed_validation_data_mltable_path)
    # preprocessed_validation_data_mltable.save(os.path.join(args.preprocessed_validation_data))
    # print(os.path.join(preprocessed_validation_data_path, "MLTable.yml"))


    preprocessed_train_data_mltable_path = os.path.join(
        args.preprocessed_train_data, "MLTable"
    )

    #train_yaml_file = {'Paths:{file: ./preprocessed_train_data.csv}, transformations:{read_delimited:{delimiter: ",",encoding: "ascii"}}'}
    train_yaml_file = dict(
        paths = [dict(
            file = './preprocessed_train_data.csv'
        )]
        ,transformations = [dict(
            read_delimited = dict(
                delimiter = ','
                ,encoding = 'utf8'
                #,headers = 'all_files_same_headers'
                #,support_multi_line = True
            )
        )]
    )

    with open(preprocessed_train_data_mltable_path, "w") as file:
         yaml.dump(train_yaml_file, file)

    preprocessed_validation_data_mltable_path = os.path.join(
        args.preprocessed_validation_data, "MLTable"
    )
    
    validation_yaml_file = dict(
        paths = [dict(
            file = './preprocessed_validation_data.csv'
        )]
        ,transformations = [dict(
            read_delimited = dict(
                delimiter = ','
                ,encoding = 'utf8'
                #,headers = 'all_files_same_headers'
                #,support_multi_line = True
            )
        )]
    )

    with open(preprocessed_validation_data_mltable_path, "w") as file:
         yaml.dump(validation_yaml_file, file)

    # import mltable

    # mltable.load("../../training-mltable-folder")


    
    # preprocessed_validation_data_path = os.path.join(
    #     args.preprocessed_validation_data, "MLTable"
    # )

    # validation_yaml_file = {'Paths:{file: ./preprocessed_validation_data.csv}, transformations:{read_delimited:{delimiter: ",",encoding: "ascii"}}'}

    # with open(preprocessed_validation_data_path, "w") as file:
    #      yaml.dump(validation_yaml_file, file)

    # # read and write MLModel yaml file for validation data
    # validation_data_mltable_path = os.path.join(args.validation_data, "MLTable")
    # preprocessed_validation_data_mltable_path = os.path.join(
    #     args.preprocessed_validation_data, "MLTable"
    # )
    # with open(validation_data_mltable_path, "r") as file:
    #     yaml_file = yaml.safe_load(file)
    # with open(preprocessed_validation_data_mltable_path, "w") as file:
    #     yaml.dump(yaml_file, file)


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
