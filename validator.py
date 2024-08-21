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
    mat_df = pd.read_csv(file_path,)
    
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

    make_training_plots = False
    validate_models = False
    make_validation_plots = True

    if make_training_plots:
        _i_o_dir = r"D:\Bachelor_Thesis_Cedric_Grieder\presentation\Presentation_material_models"
        for selector_name in os.listdir(_i_o_dir):
            print(f'Processing {selector_name}')
            selector_path = os.path.join(_i_o_dir, selector_name)
            for model_name in os.listdir(selector_path):
                print(f'Processing {model_name}')
                model_dir = os.path.join(selector_path, model_name)
                for epoch_dir in os.listdir(model_dir):
                    low_loss_epoch_dir = os.path.join(model_dir, epoch_dir)
                    title_force = f'Force Comparison {model_name} {selector_name}'
                    kNN_title = f'kNN {model_name} {selector_name}'
                    k_nn_CSV = os.path.join(low_loss_epoch_dir, 'kNN_master.csv')
                    force_loss_csv = os.path.join(low_loss_epoch_dir, 'force_comparison.csv')
                    output_dir = os.path.join(low_loss_epoch_dir, 'output')
                    plotter = ut.PlotGenerator(output_dir)
                    plotter.force_plot(force_loss_csv, title_force)
                    plotter.kNN_plot(k_nn_CSV, kNN_title)

                    working_directory = os.makedirs(os.path.join(_i_o_dir, selector_name, model_name, low_loss_epoch_dir, 'work'), exist_ok=True)
                    archive_directory = os.makedirs(os.path.join(_i_o_dir, selector_name, model_name, low_loss_epoch_dir, 'output'), exist_ok=True)


    expnames = [
    'C_20',
    'H_50',
    ]
    if validate_models:
        validation_knn_dir = r"D:\Bachelor_Thesis_Cedric_Grieder\Code\testing\Merge\neural-network-hardening\validation\kNN"
        for kNN_csv in os.listdir(validation_knn_dir):
            mat_tensor = import_mat_tensor_from_csv(os.path.join(validation_knn_dir, kNN_csv))
            working_directory = r"D:\Bachelor_Thesis_Cedric_Grieder\Code\testing\Merge\neural-network-hardening\validation\work"
            for expname in expnames:
                force_loss_df = validator(mat_prop_tensor=mat_tensor, working_directory= working_directory, expname = expname, run_simulation = True, num_cpus = 4)
                out_dir = os.path.join(working_directory, f'{expname}{kNN_csv}_force_comparison.csv')
                force_loss_df.to_csv(out_dir, index=False)
    force_csv_dir = r"D:\Bachelor_Thesis_Cedric_Grieder\Code\testing\Merge\neural-network-hardening\validation\force_csv"
    if make_validation_plots:
        for force_csv in os.listdir(force_csv_dir):
            force_csv_path = os.path.join(force_csv_dir, force_csv)
            plotter = ut.PlotGenerator(force_csv_dir)
            filename = force_csv.split('.')[0]
            plotter.force_plot(force_csv_path, filename, filename)

