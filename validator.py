import utils as ut
import pandas as pd
import os
import subprocess
import torch

class LowLossSearcher:
    def __init__(self, exclude_model_attr:str =None):
        self.lowest_loss = float('inf')
        self.lowest_loss_dir = None
        self.exclude_modelattr = exclude_model_attr

    def process_models_directory(self, modeldir_path):
        for epochdir in os.listdir(modeldir_path):
            if 'Model' in epochdir and (self.exclude_modelattr is None or self.exclude_modelattr not in epochdir):
                epochdir_path = os.path.join(modeldir_path, epochdir)
                self.process_training_runs_directory(epochdir_path)

        
    def process_training_runs_directory(self, trainingdir_path):
        for epochdir in os.listdir(trainingdir_path):
            if 'nepochs' in epochdir:
                trainingdir_path = os.path.join(trainingdir_path, epochdir)
                self.process_epoch_directory(trainingdir_path)

    def process_epoch_directory(self, epochdir_path):
        for subdir in os.listdir(epochdir_path):
            if 'epoch' in subdir:
                subdir_path = os.path.join(epochdir_path, subdir)
                self.process_sub_epoch_directory(subdir_path)

    def process_sub_epoch_directory(self, subdir_path):
        checkpoint_found = False
        for file in os.listdir(subdir_path):
            if 'checkpoint_log' in file:
                checkpoint_found = True
                file_path = os.path.join(subdir_path, file)
                self.process_checkpoint_log(file_path, subdir_path)
        if not checkpoint_found:
            print(f"No checkpoint log found in: {subdir_path}\n")

    def process_checkpoint_log(self, file_path, subdir_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Loss' in line:
                    try:
                        loss = float(line.split(' ')[-1])
                        if loss < self.lowest_loss:
                            self.lowest_loss = loss
                            self.lowest_loss_dir = subdir_path
                    except ValueError:
                        print(f"Could not convert loss to float in line: {line}")

def import_mat_tensor_from_csv(file_path):
    # Read the CSV file
    mat_df = pd.read_csv(file_path, index_col=0)
    
    # Identify the kNN(strain) column
    kNN_column = [col for col in mat_df.columns if col.startswith('kNN(strain)')][0]
    
    # Extract the relevant columns
    strain_series = mat_df['strain'].values
    stress_series = mat_df[kNN_column].values

    strain_tensor = torch.tensor(strain_series)
    stress_tensor = torch.tensor(stress_series)
    
    # Convert to tensor format (assuming ut.mat_tensor is the function to create the tensor)
    mat_tensor = ut.mat_tensor(strain_tensor=strain_tensor, stress_tensor=stress_tensor)
    
    return mat_tensor

def validator(mat_prop_tensor, working_directory, expname, run_simulation = True, num_cpus = 4):
    """
    input: MatProp Tensor, Name: e.g. H_5, 
    output: dataframe with 'displacement', 'force_target', 'force_sim'
    """

    if run_simulation:
        inp = ut.AbaqusInputFile(working_directory, expname)
        inp.change_plastic(mat_prop_tensor)
        inp.write_file()
        abq = ut.AbaqusFunc(working_directory=working_directory, name=expname, num_cpus=num_cpus)
        abq.run_simulation()

        abq.extract_output()

    outp = ut.AbaqusOutputValidationFile(working_directory, expname)
    df= outp.model_data_df

    return df

if __name__ == '__main__':
    start_dir = os.getcwd()


    validate = True
    if validate:
        validation_dirs = r"D:\Bachelor_Thesis_Cedric_Grieder\results\20240818\low_loss"
        dirs_to_validate = [dir for dir in os.listdir(validation_dirs)]
        os.chdir(start_dir)
        archive_directory = 'validation'
        working_directory = os.path.join(archive_directory, 'work')
        kNN_file = r"D:\Bachelor_Thesis_Cedric_Grieder\Code\testing\Merge\neural-network-hardening\validation\kNN\0815_500RMS_Deep_LR001_MSE.csv"
        mat_tensor = import_mat_tensor_from_csv(kNN_file)
        expnames = [
            #'C_20',
            'H_50',
            ]
        
        force_loss_df = validator(mat_prop_tensor=mat_tensor, working_directory= working_directory, expname = 'H_50', run_simulation = True, num_cpus = 4)
        summary_writer = ut.TestrunSummaryWriter(archive_directory= archive_directory, working_directory= working_directory, validation=True,)
        summary_writer.validation_summary(title = '0815_500RMS_Deep_LR001_MSE', loss_df= force_loss_df)
        searcher = LowLossSearcher()
        lowest_loss_dirs = []
        kNNs = []
        run_names = []

        for dir in dirs_to_validate:
            break
            run_names.append((dir.split('\\'))[-1])
            searcher.process_models_directory(os.path.join(validation_dirs, dir))
            lowest_loss_dirs.append(searcher.lowest_loss_dir)
            if 'kNN_master.csv' in os.listdir(searcher.lowest_loss_dir):
                kNNs.append(os.path.join(searcher.lowest_loss_dir, 'kNN_master.csv'))

        for expname in expnames:
            break
            for kNN in kNNs:
                mat_tensor = import_mat_tensor_from_csv(kNN)
            force_loss_df = validator(mat_prop_tensor=mat_tensor, working_directory= working_directory, expname = expname, run_simulation = True, num_cpus = 4)
            summary_writer = ut.TestrunSummaryWriter(archive_directory= archive_directory, working_directory= working_directory, validation=True,)
            summary_writer.validation_summary(title = '0815_500RMS_Deep_LR001_MSE', loss_df= force_loss_df)