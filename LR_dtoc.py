import pandas as pd
import matplotlib as plt
import pyodbc
import datetime
import seaborn as sns
import pandas.io.sql as psql
import time
import numpy as np
import _pickle as pickle
import tensorflow as tf


def place_holder_inputs(batchsize):
    #is batchsize necessary?
    x_pl = tf.placeholder(tf.float32, [None, 12*100]) # features in one spell, diagnose codes number(12) * embedding size(100)
    y_pl = tf.placeholder(tf.float32, [None, 2]) # prediction target: dtoc or not
    return x_pl, y_pl

def _feed_dict(dataset, spell_pl, target_pl):
    
    pass

def run_training(learning_rate, dataset, training_epochs, batch_size, display_step):

    pass

def main():
    pass


if __name__ == "__main__":
    main()
