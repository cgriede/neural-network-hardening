import torch
import torch.nn as nn
from torch.autograd import Function
import utils as ut
import torch.optim as optim
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime, timedelta
import os


import torch
import torch.nn as nn
import torch.optim as optim

class Loss:
    def __init__(self,):
        self.output = None
        self.target = None
        self.value = None


    def compute_loss(self, output, target):
        self.output = output
        self.target = target
        raise NotImplementedError("Subclasses should implement this method")
    
    def gradient(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def value(self):
        return self.value
    
class MSE(Loss):
    def __init__(self,):
        super().__init__()

    def compute_loss(self, output, target):
        self.output = output
        self.target = target
        self.value = ((self.output - self.target) ** 2).mean()
        return self.value
    
    def gradient(self):
        return 2 * (self.output - self.target) / len(self.output)
    
class WSE(Loss):
    def __init__(self, mul: float, threshold: float):
        super().__init__()
        self.mul = mul
        self.threshold = threshold

    def compute_loss(self, output, target):
        self.output = output
        self.target = target
        errors = (self.output - self.target) ** 2
        threshold_index = int(len(self.output) * self.threshold)
        errors[:threshold_index] *= self.mul
        self.value = errors.mean()
        return self.value
    
    def gradient(self):
        errors = self.output - self.target
        threshold_index = int(len(self.output) * self.threshold)
        errors[:threshold_index] *= self.mul
        return 2 * errors / len(self.output)
    
class FeatureSelector:
    def __init__(self):
        self.filtered_elements = set()
        self.gradient_list = []
        self.input_strain_list = []
        self.info_df = pd.DataFrame()

        self.force_loss_df = None
        self.element_dict = None

    def select_features(self, loss, force_loss_df, element_dict):
        raise NotImplementedError("Subclasses should implement this method")
    
class StandardFilter(FeatureSelector):
    def __init__(self,):
        super().__init__()

    def select_features(self, loss, force_loss_df, element_dict):
        self.loss = loss
        self.force_loss_df = force_loss_df
        self.element_dict = element_dict

        filtered_elements = set()

        sigma_33_list = []
        sigma_vm_list = []
        force_contribution_list = []
        area_xy_list = []
        dS33_dSvm_list = []
        dErr_dFsim_list = []
        element_list = []
        time_list = []
        gradient_list = []
        strain_list = []
        

        dErr_dFsim = self.loss.gradient()
        self.force_loss_df = pd.concat([self.force_loss_df, pd.Series(dErr_dFsim, name='dErr_dFsim')], axis=1)

        #iterate over displacements
        for index, row in self.force_loss_df.iterrows():
            dErr_dFsim = row['dErr_dFsim']
            #iterate over elements
            for key, element in self.element_dict.items():
                #get the plastic scalars dataframe
                plastic_scalars_df = element['plastic_scalars_df']

                # Extract stresses
                sigma_11 = plastic_scalars_df.iloc[index]['S11']
                sigma_22 = plastic_scalars_df.iloc[index]['S22']
                sigma_33 = plastic_scalars_df.iloc[index]['S33']

                #filter for low strains that cause exploding gradients
                strain = plastic_scalars_df.iloc[index]['PEEQ']
                if strain < 0.0005:
                    filtered_elements.add(key)
                    continue

                #filter positive S33 values (only interested in compression)
                if sigma_33 > -0.0001:
                    filtered_elements.add(key)
                    continue
                
                # Ensure significant sigma_33 contribution
                if not (2 * abs(sigma_33) > abs(sigma_11 + sigma_22)):
                    filtered_elements.add(key)
                    continue

                #append the stresses to the stress lists
                sigma_33_list.append(sigma_33)
                sigma_vm = plastic_scalars_df.iloc[index]['MISES']
                sigma_vm_list.append(sigma_vm)

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


                #final gradient applying the chain rule, attention to the negative sign
                #force error is in negatve z direction therefore the gradient is negative
                dFsim_dSvm = -dS33_dSvm*area_xy
                grad = dErr_dFsim*dFsim_dSvm

                #grad_displacement.append(grad)
                gradient_list.append(grad)
                strain_list.append(strain)

                #append debugging data
                dErr_dFsim_list.append(dErr_dFsim)
                element_list.append(key)
                time_list.append(plastic_scalars_df.iloc[index]['X'])

        self.info_df = pd.DataFrame({
            'time': time_list,
            'dErr_dFsim': dErr_dFsim_list,
            'Area_XY': area_xy_list,
            'dS33_dSvm': dS33_dSvm_list,
            'element': element_list,
            'sigma_33': sigma_33_list,
            'sigma_vm': sigma_vm_list,
            'force_contribution': force_contribution_list,
            'grad': gradient_list,
            'strain': strain_list
            })
        self.filtered_elements = filtered_elements
        self.gradient_list = gradient_list
        self.input_strain_list = strain_list


class ModelTrainer:
    def __init__(self,
                 model,
                 working_directory,
                 archive_directory,
                 experiment_name,
                 num_cpus: int,
                 n_epochs: int,
                 optimizer: torch.optim,
                 FeatureSelector: FeatureSelector,
                 clean: bool = False,
                 run_sim: bool=False,
                 LossInst= MSE(),
                 scheduler=None,
                 clipping_rate: float = None,
                 ):
        self.model = model
        self.working_directory = working_directory
        self.archive_directory = archive_directory
        self.experiment_name = experiment_name
        self.num_cpus = num_cpus
        self.optimizer = optimizer
        self.run_sim = run_sim
        self.LossInst = LossInst
        self.scheduler = scheduler
        self.clean = clean
        self.learning_rate = optimizer.param_groups[0]['lr']
        self.n_epochs = n_epochs
        self.clip = clipping_rate
        self.FeatureSelector = FeatureSelector
        self.summary = self.init_summary_writer()

        #initialize the feature selector
        self.features = self.FeatureSelector()
        #initialize the input and output scalers
        self.input_scaler = self.init_input_scaler()
        self.output_scaler = self.init_output_scaler()

        #initialize the strain tensor
        self.strain_tensor = None
        self.set_strain_tensor()

        #initialize the simulation output
        self.loss_df = None
        self.element_dict = None

    def init_summary_writer(self, 
                             test_description:str = None, train_loop_description:str = None):
        
        #SET TEST NAME
        test_name = f'{self.model.name}_{self.optimizer.__class__.__name__}_{self.learning_rate}_nepochs_{self.n_epochs}'

        summary = ut.TestrunSummaryWriter(
        archive_directory= self.archive_directory,
        test_name=test_name,
        test_description=test_description,
        train_loop_description=train_loop_description,
        optimizer=self.optimizer,
        expname=self.experiment_name,
        learning_rate=self.learning_rate,
        number_of_epochs=self.n_epochs,
        working_directory=self.working_directory,
        model=self.model,
    )
        return summary


    def set_strain_tensor(self, strain_tensor=None):
        self.strain_tensor = strain_tensor
        #default strain tensor
        if self.strain_tensor is None:
            s_xsmall = torch.linspace(0,0.00095, 20, dtype=torch.float64)
            s_small = torch.linspace(0.001,0.0095, 20, dtype=torch.float64)
            s_middle = torch.linspace(0.01, 0.0495, 20, dtype=torch.float64)
            s_large = torch.linspace(0.05, 0.7, 20, dtype=torch.float64)

            self.strain_tensor = torch.cat((s_xsmall, s_small, s_middle, s_large)).unsqueeze(1)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_feature_selector(self, FeatureSelector):
        self.FeatureSelector = FeatureSelector

    def check_abort(self):
        ABORT = False
        abort_file = False
        os.listdir(self.working_directory)
        for file in os.listdir(self.working_directory):
            if 'ABORT' in file:
                abort_file = True
        #check for aborted simulation
        if (self.loss_df.empty, self.element_dict) == (True, False):
            ABORT = True
        #check for abort file
        if abort_file:
            ABORT = True
        #abort training if conditions are met
        if ABORT:
            print('ABORTING TRAINING')
        
        return ABORT

    def init_input_scaler(self):
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
        return scaler_input

    def init_output_scaler(self):
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
        return scaler_output
    
    def run_simulator(self):
                #Abaqus simulation / 1st Forward pass
        with torch.no_grad():
            strain_scaled = torch.tensor(self.input_scaler.fit_transform(self.strain_tensor), dtype=torch.float64)
            stress_scaled = self.model(strain_scaled)
            stress_real = torch.tensor(self.output_scaler.inverse_transform(stress_scaled.detach().numpy().reshape(-1, 1)), dtype=torch.float64)
            mat_tensor = ut.mat_tensor(self.strain_tensor, stress_real)

        
        self.loss_df, self.element_dict = ut.simulator(mat_tensor, self.working_directory, self.experiment_name,
                                                        run_simulation=self.run_sim, num_cpus=self.num_cpus)

    def train_default(self,):
    
        start_time_training = datetime.now()

        for current_epoch in range(self.n_epochs):
            start_time_epoch = datetime.now()

            #Abaqus simulation / 1st Forward pass
            self.run_simulator()

            # Record the time for the simulator
            end_time_simulator = datetime.now()

            #calculate the loss in Newtons
            target = self.loss_df['force_target']
            output = self.loss_df['force_sim']

            loss = self.LossInst.compute_loss(output=output, target=target)
            #apply the feature selector (calculate gradients and filter elements)
            self.features.select_features(self.LossInst, self.loss_df, self.element_dict)

            #extract the features
            strain_list = self.features.input_strain_list
            grad_list = self.features.gradient_list

            #convert the lists to tensors and scale them
            strain = torch.tensor(strain_list, dtype=torch.float64, requires_grad=False).unsqueeze(1)
            strain_scaled = torch.tensor(self.input_scaler.fit_transform(strain), dtype=torch.float64, requires_grad=True)
            grad = torch.tensor(grad_list, dtype=torch.float64, requires_grad=False).unsqueeze(1)

            # Generate shuffled indices
            shuffled_indices = torch.randperm(strain_scaled.size(0))

            # Use the same shuffled indices to shuffle both tensors
            strain_scaled_shuffled = strain_scaled[shuffled_indices]
            grad_shuffled = grad[shuffled_indices]

            

            self.optimizer.zero_grad()
            # Forward pass
            stress_scaled = self.model(strain_scaled_shuffled)

            # Backward pass
            stress_scaled.backward(grad_shuffled)

            #apply clipping if specified
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()

            end_time_epoch = datetime.now()

            #display the results
            print(f'''
            ________________________________________________________________
            Epoch [{current_epoch+1}/{self.n_epochs}], Loss: {loss},
            Mean Grad: {np.mean(grad_list)}, Median: {np.median(grad_list)}
            time: {datetime.now()}
            time total epoch: {end_time_epoch-start_time_epoch}
            time for simulator: {end_time_simulator-start_time_epoch}
            time for backprop: {end_time_epoch-end_time_simulator}
            total training time: {end_time_epoch-start_time_training}
            ________________________________________________________________
            ''')


            #calculate the plastic table once again with the new model
            with torch.no_grad():
                strain_scaled = torch.tensor(self.input_scaler.fit_transform(self.strain_tensor), dtype=torch.float64)
                stress_scaled = self.model(strain_scaled)
                stress_real = torch.tensor(self.output_scaler.inverse_transform(stress_scaled.detach().numpy().reshape(-1, 1)), dtype=torch.float64)
                new_mat_tensor = ut.mat_tensor(self.strain_tensor, stress_real)

            self.summary.checkpoint(current_epoch = current_epoch,
                            loss_epoch = loss,
                            removed_elements= self.features.filtered_elements,
                            mat_tensor = new_mat_tensor,
                            model = self.model,
                            loss_df = self.loss_df,
                            backprop_element_df=self.features.info_df)
            
            #clean files
            if self.clean:
                ut.cleaner(self.working_directory)

            ABORT = self.check_abort()
            if ABORT:
                break
        self.summary.write_summary()


class DynamicFilter0005(FeatureSelector):
    def __init__(self,):
        super().__init__()
        self.loss_list = []
        self.cooldown = 1
        self.insignificant_threshold = 2 #2% threshold for insignificant loss decrease (lr 0.0005)
        self.cut_ratio = 0.2

    def select_features(self, loss, force_loss_df, element_dict):
        self.loss = loss
        self.loss_list.append(loss.value)
        self.force_loss_df = force_loss_df
        self.element_dict = element_dict


        if len(self.loss_list) > self.cooldown:
            previous_loss = self.loss_list[-self.cooldown - 1]
            current_loss = self.loss_list[-1]
            percentage_decrease = ((previous_loss - current_loss) / previous_loss) * 100

            if percentage_decrease < self.insignificant_threshold:
                print("Insignificant loss decrease:", percentage_decrease, "%")
                if self.cut_ratio < 0.9:
                    self.cut_ratio += 0.1
                    print(f'increased cut ratio to {self.cut_ratio}')

        use_displacement_index = int(len(self.force_loss_df) * self.cut_ratio)

        filtered_elements = set()

        sigma_33_list = []
        sigma_vm_list = []
        force_contribution_list = []
        area_xy_list = []
        dS33_dSvm_list = []
        dErr_dFsim_list = []
        element_list = []
        time_list = []
        gradient_list = []
        strain_list = []
        

        dErr_dFsim = self.loss.gradient()
        self.force_loss_df = pd.concat([self.force_loss_df, pd.Series(dErr_dFsim, name='dErr_dFsim')], axis=1)

        #iterate over displacements
        for index, row in self.force_loss_df.iterrows():

            #only use the first part of the displacement
            if index > use_displacement_index:
                break

            dErr_dFsim = row['dErr_dFsim']
            #iterate over elements
            for key, element in self.element_dict.items():
                #get the plastic scalars dataframe
                plastic_scalars_df = element['plastic_scalars_df']

                # Extract stresses
                sigma_11 = plastic_scalars_df.iloc[index]['S11']
                sigma_22 = plastic_scalars_df.iloc[index]['S22']
                sigma_33 = plastic_scalars_df.iloc[index]['S33']

                #filter for low strains that cause exploding gradients
                strain = plastic_scalars_df.iloc[index]['PEEQ']
                if strain < 0.0005:
                    filtered_elements.add(key)
                    continue

                #filter positive S33 values (only interested in compression)
                if sigma_33 > -0.0001:
                    filtered_elements.add(key)
                    continue
                
                # Ensure significant sigma_33 contribution
                if not (2 * abs(sigma_33) > abs(sigma_11 + sigma_22)):
                    filtered_elements.add(key)
                    continue

                #append the stresses to the stress lists
                sigma_33_list.append(sigma_33)
                sigma_vm = plastic_scalars_df.iloc[index]['MISES']
                sigma_vm_list.append(sigma_vm)

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


                #final gradient applying the chain rule, attention to the negative sign
                #force error is in negatve z direction therefore the gradient is negative
                dFsim_dSvm = -dS33_dSvm*area_xy
                grad = dErr_dFsim*dFsim_dSvm

                #grad_displacement.append(grad)
                gradient_list.append(grad)
                strain_list.append(strain)

                #append debugging data
                dErr_dFsim_list.append(dErr_dFsim)
                element_list.append(key)
                time_list.append(plastic_scalars_df.iloc[index]['X'])

        self.info_df = pd.DataFrame({
            'time': time_list,
            'dErr_dFsim': dErr_dFsim_list,
            'Area_XY': area_xy_list,
            'dS33_dSvm': dS33_dSvm_list,
            'element': element_list,
            'sigma_33': sigma_33_list,
            'sigma_vm': sigma_vm_list,
            'force_contribution': force_contribution_list,
            'grad': gradient_list,
            'strain': strain_list
            })
        self.filtered_elements = filtered_elements
        self.gradient_list = gradient_list
        self.input_strain_list = strain_list


class DynamicFilter001(DynamicFilter0005):
    def __init__(self,):
        super().__init__()
        self.loss_list = []
        self.cooldown = 5
        self.insignificant_threshold = 1 # 1% threshold for insignificant loss decrease (lr 0.001)
        self.cut_ratio = 0.2