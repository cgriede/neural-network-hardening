General workflow:

The environment needs to be set up correctly, following packages / programs are used:
Abaqus
Python 3.12:
  numpy
  pytorch
  sklearn
  pandas
  time
  matplotlib (matplotlib.use('Agg')) this argument is specified in utils.py and is necessary for cluster operation
  
the wd_dev directory should contain abaqus simulation input files which are named like the experiment.
the experiment data is contained in exp_data dir.
The file picking algo goes for the naming pattern C_20.csv (exp_data) and C_20_000.inp (abaqus sim).

Using script_generator.py ML test runs may be generated (a complete folder is saved to ML_Tests dir),
where hyperparameters like lr and epoch number or optimizers and models may be specified.

These standalone folders contain all necessary scripts for training on a cluster or locally given the environment is
set up correctly.

Training may be started with the main.py script either directly or using the bash script. For running multiple
training runs the submit_all_jobs.sh may be used.

models.py contains a selection of models to choose from.

When a simulation is run the main script uses the simulator function in utils.py

This function:
1: Writes a new input file with a model prediction
2: sends a cmd to start the sim
3: extracts data from the .odb to an .rpt
4: reads data from the .rpt into multiple dataframes (dictionary of dataframes: {Element ID : Dataframe})
   as well as general force data for the whole experiment
5: returns a tuple containing: Force Dataframe, dictionary of dataframes


the ModelTrainer class may be extended to incorporate different training loops and is defined in train_functions.py

the trainer.train_default() method uses the simulator function to run abaqus simulations and select features
according to the feature selector used. backpropagation is then performed on the selected tensor.
Also feature scaling is performed here.

the checkpoint writer writes a checkpoint in a specified interval or when a new lower loss is found. Generating a csv
containing the hardening law, and force response, also generating plots for these.

If there are any questions or unclarities regarding the code please write an email to cgriede@ethz.ch
