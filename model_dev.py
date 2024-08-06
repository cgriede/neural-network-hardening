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



"""
here the neural net model layers and activation functions are defined


"""


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(1, 8, dtype=torch.float64)
        self.fc2 = nn.Linear(8, 16, dtype=torch.float64)
        self.fc3 = nn.Linear(16, 8, dtype=torch.float64)
        self.fc4 = nn.Linear(8, 1, dtype=torch.float64)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

                # Apply weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.softplus(x)
        return x

def train_strain_normalized(model,
            working_directory: str,
            expname: str,
            strain_tensor: torch.tensor,
            epoch: int,
            num_cpus: int,
            optimizer: torch.optim.Optimizer,
            run_sim: bool=False,
            clip: float = 10**5):
    
    start_time_training = datetime.now()
    
    #data colloection for debugging
    kNN_epoch = []
    force_error_epoch = []
    loss_epoch = []
    debug_grad_epoch = []
    clips_epoch = 0
    grad_list_unclipped = []
    fail_counter = 0

    SCALE_INPUT = True
    if SCALE_INPUT == True:
        # Assuming you know the min and max of the data you're scaling
        data_min = 0  # Replace with your actual data's min
        data_max = 0.7  # Replace with your actual data's max

        feature_range = (0, 1)  # The desired range after scaling

        # Manually calculate scale and min_
        scale = (feature_range[1] - feature_range[0]) / (data_max - data_min)
        min_ = feature_range[0] - data_min * scale

        # Initialize the scaler
        scaler_input = MinMaxScaler(feature_range=feature_range)

        # Manually set the min_ and scale_ attributes
        scaler_input.min_ = np.array([min_])
        scaler_input.scale_ = np.array([scale])
        scaler_input.data_min_ = np.array([data_min])
        scaler_input.data_max_ = np.array([data_max])
        scaler_input.data_range_ = np.array([data_max - data_min])
        scaler_input.n_samples_seen_ = 1

    SCALE_OUTPUT = True
    if SCALE_OUTPUT == True:
        # Assuming you know the min and max of the data you're scaling
        data_min = 0  # Replace with your actual data's min
        data_max = 120  # Replace with your actual data's max

        feature_range = (0, 1)  # The desired range after scaling

        # Manually calculate scale and min_
        scale = (feature_range[1] - feature_range[0]) / (data_max - data_min)
        min_ = feature_range[0] - data_min * scale

        # Initialize the scaler
        scaler_output = MinMaxScaler(feature_range=feature_range)

        # Manually set the min_ and scale_ attributes
        scaler_output.min_ = np.array([min_])
        scaler_output.scale_ = np.array([scale])
        scaler_output.data_min_ = np.array([data_min])
        scaler_output.data_max_ = np.array([data_max])
        scaler_output.data_range_ = np.array([data_max - data_min])
        

    for n in range(epoch):
        start_time_epoch = datetime.now()
        strain_list = []
        grad_list = []
        sigma_33_list = []
        sigma_vm_list = []
        force_contribution_list = []
        area_xy_list = []
        dS33_dSvm_list = []
        q_list = []
        dErr_dFsim_list = []
        element_list = []
        time_list = []
        
        #Abaqus simulation / 1st Forward pass
        with torch.no_grad():
            strain_scaled = torch.tensor(scaler_input.fit_transform(strain_tensor), dtype=torch.float64)
            stress_scaled = model(strain_scaled)
            stress_real = torch.tensor(scaler_output.inverse_transform(stress_scaled.detach().numpy().reshape(-1, 1)), dtype=torch.float64)
            mat_tensor = ut.mat_tensor(strain_tensor, stress_real)

        #append the material tensor to the kNN list for the summary plot    
        kNN_epoch.append(mat_tensor)
        
        loss_df, element_dict = ut.simulator(mat_tensor, working_directory, expname, run_simulation=run_sim, num_cpus=num_cpus)

        ABORT = False
        abort_file = False
        os.listdir(working_directory)
        for file in os.listdir(working_directory):
            if 'ABORT' in file:
                abort_file = True
        #check for aborted simulation
        if (loss_df.empty, element_dict) == (True, False):
            ABORT = True
        #check for abort file
        if abort_file:
            ABORT = True
        #abort training if conditions are met
        if ABORT:
            print('ABORTING TRAINING')
            return loss_epoch, kNN_epoch, force_error_epoch, debug_grad_epoch
        

        # Record the time for the simulator
        end_time_simulator = datetime.now()
        
        #append the force error to the force error list for the summary plot
        force_error_epoch.append(loss_df)

        #record the time for the simulator
        end_time_simulator = datetime.now()

        #calculate the loss in Newtons (1000 factor for scaling, fexp, fsim in kN)
        mse = (((loss_df['force_target'] - loss_df['force_sim'])) ** 2).mean()
        loss = mse
        # Store the loss for the epoch
        loss_epoch.append(loss)

        clips_epoch = 0

        filtered_elements_epoch = []
        #iterate over displacements
        for index, row in loss_df.iterrows():

            f_t = row['force_target']
            f_sim = row['force_sim']

            #calculate gradient of the loss function in Newtons (1000 factor for scaling)
            dErr_dFsim = -2*(f_t - f_sim) /len(loss_df) 

            grad_displacement = []
            strain_displacement = []   

            filtered_elements_disp = []
            #iterate over elements
            for key, element in element_dict.items():
                #get the plastic scalars dataframe
                plastic_scalars_df = element['plastic_scalars_df']

                # Extract stresses
                sigma_11 = plastic_scalars_df.iloc[index]['S11']
                sigma_22 = plastic_scalars_df.iloc[index]['S22']
                sigma_33 = plastic_scalars_df.iloc[index]['S33']

                #filter for low strains that cause exploding gradients
                strain = plastic_scalars_df.iloc[index]['PEEQ']
                if strain < 0.0005:
                    filtered_elements_disp.append(key)
                    continue

                #filter positive S33 values (only interested in compression)
                if sigma_33 > -0.0001:
                    filtered_elements_disp.append(key)
                    continue
                
                # Ensure significant sigma_33 contribution
                if not (2 * abs(sigma_33) > abs(sigma_11 + sigma_22)):
                    filtered_elements_disp.append(key)
                    continue

                #append the stresses to the stress lists
                sigma_33_list.append(sigma_33)
                sigma_vm = plastic_scalars_df.iloc[index]['MISES']
                sigma_vm_list.append(sigma_vm)

                #append the strain to the strain list
                strain_displacement.append(strain)

                #Area of the element (xy-plane) = dF_element/dS33
                area_xy = plastic_scalars_df.iloc[index]['Area_XY']
                area_xy_list.append(area_xy)

                #total force of time step

                #Force contribution of the element, force in negative z direction
                force_contribution = - area_xy * sigma_33
                force_contribution_list.append(force_contribution)

                #dS33/dSvm
                dS33_dSvm = plastic_scalars_df.iloc[index]['dS33_dSvm']
                dS33_dSvm_list.append(dS33_dSvm)

                #normalisation factor q
                q = 1
                #final gradient applying the chain rule, attention to the negative sign
                #force error is in negatve z direction therefore the gradient is negative
                dFsim_dSvm = -dS33_dSvm*area_xy
                grad = dErr_dFsim*dFsim_dSvm

                #grad_displacement.append(grad)
                grad_list.append(grad)
                strain_list.append(strain)

                #append debugging data
                dErr_dFsim_list.append(dErr_dFsim)
                element_list.append(key)
                time_list.append(plastic_scalars_df.iloc[index]['X'])
                q_list.append(q)

            

        strain = torch.tensor(strain_list, dtype=torch.float64, requires_grad=False).unsqueeze(1)
        strain_scaled = torch.tensor(scaler_input.fit_transform(strain), dtype=torch.float64, requires_grad=True)
        grad = torch.tensor((grad_list), dtype=torch.float64, requires_grad=False).unsqueeze(1)

        # Generate shuffled indices
        shuffled_indices = torch.randperm(strain_scaled.size(0))

        # Use the same shuffled indices to shuffle both tensors
        strain_scaled_shuffled = strain_scaled[shuffled_indices]
        grad_shuffled = grad[shuffled_indices]

        

        optimizer.zero_grad()
        # Forward pass
        stress_scaled = model(strain_scaled_shuffled)

        # Backward pass
        stress_scaled.backward(grad_shuffled)

        optimizer.step()

     

        end_time_epoch = datetime.now()

        print(f'''
        ________________________________________________________________
        gradient interval inspection:
        ''')
        ut.interval_inspection(grad_list, [], strain_list, 10)

        #display the results
        print(f'''
        ________________________________________________________________
        Epoch [{n+1}/{epoch}], Loss: {loss},
        Mean Grad: {np.mean(grad_list)}, Median: {np.median(grad_list)}
        clipped gradients: {clips_epoch}
        %of clipped gradients: {clips_epoch/len(grad_list)*100}
        time: {datetime.now()}
        time total epoch: {end_time_epoch-start_time_epoch}
        time for simulator: {end_time_simulator-start_time_epoch}
        time for backprop: {end_time_epoch-end_time_simulator}
        total training time: {end_time_epoch-start_time_training}
        ________________________________________________________________
        ''')

        #data collection for debugging
        debugging_df = pd.DataFrame({
            'time': time_list,
            'dErr_dFsim': dErr_dFsim_list,
            'Area_XY': area_xy_list,
            'dS33_dSvm': dS33_dSvm_list,
            'strain': strain_list,
            'grad': grad_list,
            'element': element_list,
            'sigma_33': sigma_33_list,
            'sigma_vm': sigma_vm_list,
            'force_contribution': force_contribution_list,
            'q': q_list
            })
        debugging_df.to_csv(f'{working_directory}/debugging_df_{n}.csv')
        debug_grad_epoch.append(debugging_df)

        

        #clean files
        if CLEAN:
            ut.cleaner(working_directory)

    return loss_epoch, kNN_epoch, force_error_epoch, debug_grad_epoch











'''

Here the defined training loop is run for the models defined above

'''








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


    #set strain tensor
    s_xsmall = torch.linspace(0,0.00095, 20, dtype=torch.float64)
    s_small = torch.linspace(0.001,0.0095, 20, dtype=torch.float64)
    s_middle = torch.linspace(0.01, 0.0495, 20, dtype=torch.float64)
    s_large = torch.linspace(0.05, 0.7, 20, dtype=torch.float64)

    strain_tensor = torch.cat((s_xsmall, s_small, s_middle, s_large)).unsqueeze(1)

    # Define the loss criterion
    criterion = torch.nn.MSELoss()




    test_description = '''
    input strain tensor in range of 0 to 0.5

    Output is scaled to the range of 0 to 120
    
    # Define the optimizers to test
    optimizers = {
        'RMSprop': optim.RMSprop,
    }


    models = {
        'SimpleModel': SimpleModel,
        ComplexModel: ComplexModel, has a custom activation function with a dropout layer in between
    }

    # Define the learning rates to test

    '''

    train_loop_description ='''

    input scaled to the range of 0 to 0.7
    output scaled to the range of 0 to 120
    popping all elements with all PEEQ values below 0.005
    popping gradients with a PEEQ below 0.0005
    gradient clipping at 1000
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
    epoch = 1

    #set the random seed
    torch.manual_seed(37)

    # Define the models to test
    model = SimpleModel()
    model_type = '4_layer_tanh_softplus'


    # Define the learning rate
    lr = 0.001

    # Define the optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    optim_type = 'RMSprop'

    #wheter to update inp and run simulation (turn off for debugging, Abaqus not installed)
    RUN_SIM = True

    #training loop
    TRAIN = True

    #wheter to clean the working directory after each epoch
    CLEAN = True


    #set debug mode
    DEBUG = True
    if DEBUG:
        RUN_SIM = False
        CLEAN = False
        epoch = 1    

    
    if TRAIN:
        #SET TEST NAME
        test_name = f'{model_type}_{optim_type}_{lr}_nepochs_{epoch}'

        #main training loop
        
        (
        loss_epoch,
        kNN_epoch,
        force_error_epoch,
        debug_grad_epoch
        ) = train_strain_normalized(model=model,
                                                                                            working_directory=working_directory,
                                                                                            expname=expname,
                                                                                            strain_tensor=strain_tensor,
                                                                                            epoch=epoch,
                                                                                            num_cpus=num_cpus,
                                                                                            optimizer=optimizer,
                                                                                            run_sim=RUN_SIM)
        
        #summary writer
        summary = ut.testrun_summary_writer(loss_epoch, force_error_epoch, kNN_epoch, test_name, test_description,
                                        optimizer, train_loop_description, expname, lr, epoch,
                                        archive_directory,debug_grad_epoch, model)


    



###TEST / DEBUG CODE###

#validate simulation
"""
ut.train_loop_validation(model, working_directory, archive_directory, expname, strain_tensor, epoch, run_sim, num_cpus, optimizer, criterion)
"""

#debug the importer:
"""
importer = ut.AbaqusOutputFile('wd_dev', 'H_10')
importer.create_model_data()
print(importer.model_data_df)
"""

#Adam optimizer
"""
     #optimizer = torch.optim.Adam(model.parameters(), lr=LR_GLOBAL)
"""

#RMSProp optimizer
"""
    WEIGHT_DECAY_GLOBAL = None
    MOMENTUM_GLOBAL = None
    ALPHA_GLOBAL = None

    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR_GLOBAL,
                                    weight_decay = WEIGHT_DECAY_GLOBAL, momentum = MOMENTUM_GLOBAL, alpha = ALPHA_GLOBAL)
"""