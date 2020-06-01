import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import os
import sys

from args import get_setup_args

if __name__ == '__main__':

    args = get_setup_args()

    # Read the full csv data
    df = pd.read_csv(os.path.join(args.input_dir, 'data.csv'),
                     dtype=str,
                     index_col=None)

    # Split the full data in train and test
    df_train, df_test = train_test_split(df, test_size=args.test_size)

    # Split the remaining training data into training and validation
    df_train, df_dev = train_test_split(df_train, test_size=args.dev_size)

    # Save each type of the data
    df_train.to_csv(os.path.join(args.input_dir, 'train', 'dataset.csv'), index=False)
    df_dev.to_csv(os.path.join(args.input_dir, 'dev', 'dataset.csv'), index=False)
    df_test.to_csv(os.path.join(args.input_dir, 'test', 'dataset.csv'), index=False)

    # Compute the class weights for the imbalanced training set
    y_train = df_train[df_train.columns[-1]].values
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)

    # Print the class weights to add them as default parameters for the train.py
    print(f'Computed class weights for the training data are: {class_weights}')

    sys.exit(0)