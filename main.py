import torch
import torch.nn as nn
from torch.autograd import Function
import utils as ut
import torch.optim as optim
import numpy as np
import torch
import argparse
import sys
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime, timedelta
import os

import models
import train_functions as tf




if __name__ == '__main__':

    start_time_main_ = datetime.now()

    
    #need to be defined variables depending on the environment
    num_cpus = None
    cluster  = False

    #check if running on cluster
    if sys.argv[0] == 'model_dev.py':
        cluster = True
        print('Running on cluster')
    if cluster:
        
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Parse cmd line args')
        parser.add_argument('--cpus', type=int, help='Number of CPUs per task')
        args = parser.parse_args()

        #number of cpus to use for the simulation
        num_cpus = args.cpus



    # Set the device
    device = torch.device("cpu")

    '''Set the variables for the training loop'''
    #working directory
    working_directory = 'wd_dev'
    #archive directory
    archive_directory = 'archive_dev'
    #experiment name for the abaqus simulation
    expname = 'H_10'





    test_description = '''
    input strain tensor in range of 0 to 0.7

    Output is scaled to the range of 0 to 120
    


    '''

    train_loop_description ='''

    input scaled to the range of 0 to 0.7
    output scaled to the range of 0 to 120
    popping all elements with all PEEQ values below 0.005
    popping gradients with a PEEQ below 0.0005
    popping all elements with a positive S33 value
    popping all elements with insignificant S33 contribution (S33 > 2*(S11+S22))
    gradients and strain tensor are shuffled
    
    '''

    #always runs simulation when using the cluster
    if cluster:
        RUN_SIM = True
        print('Running simulation on cluster')
    #number of cpus to use for the simulation off cluster
    if not cluster:
        num_cpus = 4

    #set number of training epochs
    epochs = 500

    #set the random seed
    torch.manual_seed(37)


    # Define the models to test
    model = models.MODEL_PLACEHOLDER()

    # Define the learning rate
    lr = LR_PLACEHOLDER

    # Define the optimizer
    optimizer = OPTIMIZER_PLACEHOLDER

    clipping_rate = CR_PLACEHOLDER

    feauture_selector = FEATURE_SELECTOR_PLACEHOLDER

    loss_inst = LOSS_INST_PLACEHOLDER

    optim.Adam(model.parameters(), lr=lr)


    #wheter to update inp and run simulation (turn off for debugging, Abaqus not installed)
    RUN_SIM = True

    #training loop
    TRAIN = True

    #wheter to clean the working directory after each epoch
    CLEAN = True

    if TRAIN:
        trainer = tf.ModelTrainer(
                 model = model,
                 working_directory = working_directory,
                 archive_directory = archive_directory,
                 experiment_name = expname,
                 num_cpus = num_cpus,
                 n_epochs = epochs,
                 optimizer= optimizer,
                 clean= CLEAN,
                 FeatureSelector= feauture_selector,
                 run_sim = RUN_SIM,
                 LossInst= loss_inst,
                 scheduler=None,
                 clipping_rate = clipping_rate,)


        #run the training loop
        trainer.train_default()
