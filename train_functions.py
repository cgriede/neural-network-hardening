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
        if isinstance(self.output, float):
            length = 1
        else:
            length = len(self.output)

        return 2 * (self.output - self.target) / length
    
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
        self.contact_elements = set()
        self.gradient_list = []
        self.input_strain_list = []
        self.info_df = pd.DataFrame()

        self.force_loss_df = None
        self.element_dict = None

    def select_features(self, loss, force_loss_df, element_dict):
        raise NotImplementedError("Subclasses should implement this method")


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
            # Very small strains: 20 points, spacing of 0.00005
            s_very_small = torch.linspace(0, 0.00095, 20, dtype=torch.float64)
            # Small strains: 18 points, spacing of 0.0005 (starts from 0.001, after s_very_small ends)
            s_small = torch.linspace(0.001, 0.0095, 18, dtype=torch.float64)
            # Medium strains: 79 points, spacing of 0.005 (starts from 0.01, after s_small ends)
            s_medium = torch.linspace(0.01, 0.4, 79, dtype=torch.float64)
            # Large strains: 16 points, spacing of 0.02 (starts from 0.42 to avoid overlap with s_medium)
            s_large = torch.linspace(0.42, 0.7, 15, dtype=torch.float64)
            # Concatenate all strain tensors to ensure there are no duplicates
            self.strain_tensor = torch.cat((s_very_small, s_small, s_medium, s_large)).unsqueeze(1)

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

    def train_default(self, save_all_epochs=False):
    
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
                            backprop_element_df=self.features.info_df,
                            save_all_epochs=save_all_epochs,)
            
            #clean files
            if self.clean:
                ut.cleaner(self.working_directory)

            ABORT = self.check_abort()
            if ABORT:
                break
        self.summary.write_summary()

    
class StandardFilter(FeatureSelector):
    def __init__(self,):
        super().__init__()

    def select_features(self, loss, force_loss_df, element_dict):
        self.loss = loss
        self.force_loss_df = force_loss_df
        self.element_dict = element_dict


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



        # Function to calculate dynamic strain threshold
        def dynamic_strain_threshold(X, X_min=0, X_max=1):
            threshold_min = 0.0008
            threshold_max = 0.1
            return threshold_min + ((X - X_min) / (X_max - X_min)) * (threshold_max - threshold_min)

        # Function to calculate dynamic stress threshold
        def dynamic_stress_threshold(X, X_min=0, X_max=1):
            stress_min = -0.5
            stress_max = -5.0
            return stress_min + ((X - X_min) / (X_max - X_min)) * (stress_max - stress_min)

        # Get the minimum and maximum displacement
        X_min = self.force_loss_df['displacement'].min()
        X_max = self.force_loss_df['displacement'].max()


        initial_elements_set = set(element_dict.keys())
        #initialize contact elements
        contact_elements = set()
        contact_elements_list = []
        time_int_list = []
        # Iterate over displacements
        for index, row in self.force_loss_df.iterrows():
            displacement = row['displacement']
            time = row['X']
            time_int = int(time)
            time_int_list.append(time_int)

            dErr_dFsim = row['dErr_dFsim']
            
            # Calculate the current dynamic thresholds for strain and stress
            current_strain_threshold = dynamic_strain_threshold(displacement, X_min, X_max)
            current_stress_threshold = dynamic_stress_threshold(displacement, X_min, X_max)
            
            # Iterate over elements
            filtered_elements = set()
            for key, element in element_dict.items():
                # Get the plastic scalars dataframe
                plastic_scalars_df = element['plastic_scalars_df']

                # Extract stresses
                sigma_11 = plastic_scalars_df.iloc[index]['S11']
                sigma_22 = plastic_scalars_df.iloc[index]['S22']
                sigma_33 = plastic_scalars_df.iloc[index]['S33']

                # Filter based on the dynamic threshold for strain
                strain = plastic_scalars_df.iloc[index]['PEEQ']

                if not (key in contact_elements):
                    if strain < current_strain_threshold:
                        filtered_elements.add(key)
                        continue

                    # Filter based on the dynamic threshold for stress
                    if ((sigma_33 - current_stress_threshold) > 0):
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
                        
            unfiltered_elements_set = initial_elements_set - filtered_elements
            contact_elements = contact_elements.union(unfiltered_elements_set)
            contact_elements_list.append(contact_elements)
        

    #must always return this
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

class EnforceS33Direction(FeatureSelector):
    def __init__(self,):
        super().__init__()

    def select_features(self, loss, force_loss_df, element_dict):
        self.loss = loss
        self.force_loss_df = force_loss_df
        self.element_dict = element_dict


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



        # Function to calculate dynamic strain threshold
        def dynamic_strain_threshold(X, X_min=0, X_max=1):
            threshold_min = 0.0008
            threshold_max = 0.1
            return threshold_min + ((X - X_min) / (X_max - X_min)) * (threshold_max - threshold_min)

        # Function to calculate dynamic stress threshold
        def dynamic_stress_threshold(X, X_min=0, X_max=1):
            stress_min = -0.5
            stress_max = -5.0
            return stress_min + ((X - X_min) / (X_max - X_min)) * (stress_max - stress_min)

        # Get the minimum and maximum displacement
        X_min = self.force_loss_df['displacement'].min()
        X_max = self.force_loss_df['displacement'].max()


        initial_elements_set = set(element_dict.keys())
        #initialize contact elements
        contact_elements = set()
        contact_elements_list = []
        time_int_list = []
        # Iterate over displacements
        for index, row in self.force_loss_df.iterrows():
            displacement = row['displacement']
            time = row['X']
            time_int = int(time)
            time_int_list.append(time_int)

            dErr_dFsim = row['dErr_dFsim']
            
            # Calculate the current dynamic thresholds for strain and stress
            current_strain_threshold = dynamic_strain_threshold(displacement, X_min, X_max)
            current_stress_threshold = dynamic_stress_threshold(displacement, X_min, X_max)
            
            # Iterate over elements
            filtered_elements = set()
            for key, element in element_dict.items():
                # Get the plastic scalars dataframe
                plastic_scalars_df = element['plastic_scalars_df']

                # Extract stresses
                sigma_11 = plastic_scalars_df.iloc[index]['S11']
                sigma_22 = plastic_scalars_df.iloc[index]['S22']
                sigma_33 = plastic_scalars_df.iloc[index]['S33']

                # Filter based on the dynamic threshold for strain
                strain = plastic_scalars_df.iloc[index]['PEEQ']

                if not (key in contact_elements):
                    if strain < current_strain_threshold:
                        filtered_elements.add(key)
                        continue

                    # Filter based on the dynamic threshold for stress
                    if ((sigma_33 - current_stress_threshold) > 0):
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

                if np.sign(sigma_33) == np.sign(dS33_dSvm):
                    grad = dErr_dFsim*dFsim_dSvm

                else:
                    grad = -np.sign(sigma_33)*dErr_dFsim
                #grad_displacement.append(grad)
                gradient_list.append(grad)
                strain_list.append(strain)

                #append debugging data
                dErr_dFsim_list.append(dErr_dFsim)
                element_list.append(key)
                time_list.append(plastic_scalars_df.iloc[index]['X'])
                        
            unfiltered_elements_set = initial_elements_set - filtered_elements
            contact_elements = contact_elements.union(unfiltered_elements_set)
            contact_elements_list.append(contact_elements)
        

    #must always return this
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

class EnforceLight(FeatureSelector):
    def __init__(self,):
        super().__init__()

    def select_features(self, loss, force_loss_df, element_dict):
        self.loss = loss
        self.force_loss_df = force_loss_df
        self.element_dict = element_dict


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



        # Function to calculate dynamic strain threshold
        def dynamic_strain_threshold(X, X_min=0, X_max=1):
            threshold_min = 0.0008
            threshold_max = 0.1
            return threshold_min + ((X - X_min) / (X_max - X_min)) * (threshold_max - threshold_min)

        # Function to calculate dynamic stress threshold
        def dynamic_stress_threshold(X, X_min=0, X_max=1):
            stress_min = -0.5
            stress_max = -5.0
            return stress_min + ((X - X_min) / (X_max - X_min)) * (stress_max - stress_min)

        # Get the minimum and maximum displacement
        X_min = self.force_loss_df['displacement'].min()
        X_max = self.force_loss_df['displacement'].max()


        initial_elements_set = set(element_dict.keys())
        #initialize contact elements
        contact_elements = set()
        contact_elements_list = []
        time_int_list = []
        # Iterate over displacements
        for index, row in self.force_loss_df.iterrows():
            displacement = row['displacement']
            time = row['X']
            time_int = int(time)
            time_int_list.append(time_int)

            dErr_dFsim = row['dErr_dFsim']
            
            # Calculate the current dynamic thresholds for strain and stress
            current_strain_threshold = dynamic_strain_threshold(displacement, X_min, X_max)
            current_stress_threshold = dynamic_stress_threshold(displacement, X_min, X_max)
            
            # Iterate over elements
            filtered_elements = set()
            for key, element in element_dict.items():
                # Get the plastic scalars dataframe
                plastic_scalars_df = element['plastic_scalars_df']

                # Extract stresses
                sigma_11 = plastic_scalars_df.iloc[index]['S11']
                sigma_22 = plastic_scalars_df.iloc[index]['S22']
                sigma_33 = plastic_scalars_df.iloc[index]['S33']

                # Filter based on the dynamic threshold for strain
                strain = plastic_scalars_df.iloc[index]['PEEQ']

                if not (key in contact_elements):
                    if strain < current_strain_threshold:
                        filtered_elements.add(key)
                        continue

                    # Filter based on the dynamic threshold for stress
                    if ((sigma_33 - current_stress_threshold) > 0):
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

                if not (2 * abs(sigma_33) > abs(sigma_11 + sigma_22)):
                    #filtered previously (insignificant S33 contribution)
                    grad = 0
                #grad_displacement.append(grad)
                gradient_list.append(grad)
                strain_list.append(strain)

                #append debugging data
                dErr_dFsim_list.append(dErr_dFsim)
                element_list.append(key)
                time_list.append(plastic_scalars_df.iloc[index]['X'])
                        
            unfiltered_elements_set = initial_elements_set - filtered_elements
            contact_elements = contact_elements.union(unfiltered_elements_set)
            contact_elements_list.append(contact_elements)
        

    #must always return this
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

class Regularization(FeatureSelector):
    def __init__(self,):
        super().__init__()
        self.features_per_displacement = None

    def select_features(self, loss, force_loss_df, element_dict):
        self.loss = loss
        self.force_loss_df = force_loss_df
        self.element_dict = element_dict

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

        def dynamic_strain_threshold(X, X_min=0, X_max=1):
            threshold_min = 0.0008
            threshold_max = 0.1
            return threshold_min + ((X - X_min) / (X_max - X_min)) * (threshold_max - threshold_min)

        def dynamic_stress_threshold(X, X_min=0, X_max=1):
            stress_min = -0.5
            stress_max = -5.0
            return stress_min + ((X - X_min) / (X_max - X_min)) * (stress_max - stress_min)
        
        def find_zero_strain_gradient():

        #lower bound: 5e-05
        #upper bound: 500e-06 = 5e-04
            lower_bound = 5e-05
            upper_bound = 5e-04

            zero_strain_prediction = None
            # Find the zero-strain prediction

            for element in element_dict.values():
                # Get the plastic scalars dataframe
                plastic_scalars_df = element['plastic_scalars_df']
                
                # Filter the dataframe where PEEQ is within the specified bounds and get the MISES value
                zero_strain_prediction = plastic_scalars_df.loc[
                    (plastic_scalars_df['PEEQ'] >= lower_bound) & (plastic_scalars_df['PEEQ'] <= upper_bound), 'MISES'
                ]
                
                if not zero_strain_prediction.empty:
                    print(f'Zero strain prediction found: {zero_strain_prediction.values[0]}')
                    zero_strain_prediction_value = zero_strain_prediction.values[0]
                    break  # Exit after finding the first match

            if zero_strain_prediction is None:
                raise ValueError('No elements found within the specified bounds')
            #gradient = 2*(output - target) = 2*(zero_strain_prediction - 0)
            zero_strain_gradient = 2*zero_strain_prediction_value
            return float(zero_strain_gradient)

        X_min = self.force_loss_df['displacement'].min()
        X_max = self.force_loss_df['displacement'].max()

        initial_elements_set = set(element_dict.keys())
        contact_elements = set()
        contact_elements_list = []
        time_int_list = []

        zero_strain_gradient = find_zero_strain_gradient()

        if self.features_per_displacement is None:
            self.features_per_displacement = len(element_dict)

        for index, row in self.force_loss_df.iterrows():
            displacement = row['displacement']
            time = row['X']
            time_int = int(time)
            time_int_list.append(time_int)

            dErr_dFsim = row['dErr_dFsim']
            current_strain_threshold = dynamic_strain_threshold(displacement, X_min, X_max)
            current_stress_threshold = dynamic_stress_threshold(displacement, X_min, X_max)

            filtered_elements = set()
            feature_count = 0
            for key, element in element_dict.items():
                plastic_scalars_df = element['plastic_scalars_df']
                sigma_11 = plastic_scalars_df.iloc[index]['S11']
                sigma_22 = plastic_scalars_df.iloc[index]['S22']
                sigma_33 = plastic_scalars_df.iloc[index]['S33']
                strain = plastic_scalars_df.iloc[index]['PEEQ']


                if key in contact_elements:
                    sigma_33_list.append(sigma_33)
                    sigma_vm = plastic_scalars_df.iloc[index]['MISES']
                    sigma_vm_list.append(sigma_vm)
                    area_xy = plastic_scalars_df.iloc[index]['Area_XY']
                    area_xy_list.append(area_xy)
                    force_contribution = -area_xy * sigma_33
                    force_contribution_list.append(force_contribution)
                    dS33_dSvm = plastic_scalars_df.iloc[index]['dS33_dSvm']
                    dS33_dSvm_list.append(dS33_dSvm)

                    dFsim_dSvm = -dS33_dSvm * area_xy
                    grad = dErr_dFsim * dFsim_dSvm
                    
                    gradient_list.append(grad)
                    strain_list.append(strain)
                    dErr_dFsim_list.append(dErr_dFsim)
                    element_list.append(key)
                    time_list.append(plastic_scalars_df.iloc[index]['X'])
                    feature_count += 1
                else:
                    if strain < current_strain_threshold:
                        filtered_elements.add(key)
                        continue
                    if ((sigma_33 - current_stress_threshold) > 0):
                        filtered_elements.add(key)
                        continue

            padding_number = self.features_per_displacement - feature_count
            for i in range(padding_number):
                sigma_33_list.append(0)
                sigma_vm_list.append(0)
                area_xy_list.append(0)
                force_contribution_list.append(0)
                dS33_dSvm_list.append(0)

                strain = 0
                #scale down boundary condition by 5% and to force error scale
                _lambda = 0.05
                grad = zero_strain_gradient * _lambda * dErr_dFsim
                
                gradient_list.append(grad)
                strain_list.append(strain)
                dErr_dFsim_list.append(0)
                element_list.append(0)
                time_list.append(row['X'])

            unfiltered_elements_set = initial_elements_set - filtered_elements
            contact_elements = contact_elements.union(unfiltered_elements_set)
            contact_elements_list.append(contact_elements)

        #take last element of the list to get the number of features per displacement
        self.features_per_displacement = len(contact_elements_list[-1])

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
        self.gradient_list = gradient_list
        self.input_strain_list = strain_list
        self.filtered_elements = filtered_elements


class HeavyRegularization(FeatureSelector):
    def __init__(self,):
        super().__init__()
        self.features_per_displacement = None

    def select_features(self, loss, force_loss_df, element_dict):
        self.loss = loss
        self.force_loss_df = force_loss_df
        self.element_dict = element_dict

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

        def dynamic_strain_threshold(X, X_min=0, X_max=1):
            threshold_min = 0.0008
            threshold_max = 0.1
            return threshold_min + ((X - X_min) / (X_max - X_min)) * (threshold_max - threshold_min)

        def dynamic_stress_threshold(X, X_min=0, X_max=1):
            stress_min = -0.5
            stress_max = -5.0
            return stress_min + ((X - X_min) / (X_max - X_min)) * (stress_max - stress_min)
        
        def find_zero_strain_gradient():

        #lower bound: 5e-05
        #upper bound: 500e-06 = 5e-04
            lower_bound = 5e-05
            upper_bound = 5e-04

            zero_strain_prediction = None
            # Find the zero-strain prediction

            for element in element_dict.values():
                # Get the plastic scalars dataframe
                plastic_scalars_df = element['plastic_scalars_df']
                
                # Filter the dataframe where PEEQ is within the specified bounds and get the MISES value
                zero_strain_prediction = plastic_scalars_df.loc[
                    (plastic_scalars_df['PEEQ'] >= lower_bound) & (plastic_scalars_df['PEEQ'] <= upper_bound), 'MISES'
                ]
                
                if not zero_strain_prediction.empty:
                    print(f'Zero strain prediction found: {zero_strain_prediction.values[0]}')
                    zero_strain_prediction_value = zero_strain_prediction.values[0]
                    break  # Exit after finding the first match

            if zero_strain_prediction is None:
                raise ValueError('No elements found within the specified bounds')
            #gradient = 2*(output - target) = 2*(zero_strain_prediction - 0)
            zero_strain_gradient = 2*zero_strain_prediction_value
            return float(zero_strain_gradient)

        X_min = self.force_loss_df['displacement'].min()
        X_max = self.force_loss_df['displacement'].max()

        initial_elements_set = set(element_dict.keys())
        contact_elements = set()
        contact_elements_list = []
        time_int_list = []

        zero_strain_gradient = find_zero_strain_gradient()

        if self.features_per_displacement is None:
            self.features_per_displacement = len(element_dict)

        for index, row in self.force_loss_df.iterrows():
            displacement = row['displacement']
            time = row['X']
            time_int = int(time)
            time_int_list.append(time_int)

            dErr_dFsim = row['dErr_dFsim']
            current_strain_threshold = dynamic_strain_threshold(displacement, X_min, X_max)
            current_stress_threshold = dynamic_stress_threshold(displacement, X_min, X_max)

            filtered_elements = set()
            feature_count = 0
            for key, element in element_dict.items():
                plastic_scalars_df = element['plastic_scalars_df']
                sigma_11 = plastic_scalars_df.iloc[index]['S11']
                sigma_22 = plastic_scalars_df.iloc[index]['S22']
                sigma_33 = plastic_scalars_df.iloc[index]['S33']
                strain = plastic_scalars_df.iloc[index]['PEEQ']


                if key in contact_elements:
                    sigma_33_list.append(sigma_33)
                    sigma_vm = plastic_scalars_df.iloc[index]['MISES']
                    sigma_vm_list.append(sigma_vm)
                    area_xy = plastic_scalars_df.iloc[index]['Area_XY']
                    area_xy_list.append(area_xy)
                    force_contribution = -area_xy * sigma_33
                    force_contribution_list.append(force_contribution)
                    dS33_dSvm = plastic_scalars_df.iloc[index]['dS33_dSvm']
                    dS33_dSvm_list.append(dS33_dSvm)

                    dFsim_dSvm = -dS33_dSvm * area_xy
                    grad = dErr_dFsim * dFsim_dSvm
                    
                    gradient_list.append(grad)
                    strain_list.append(strain)
                    dErr_dFsim_list.append(dErr_dFsim)
                    element_list.append(key)
                    time_list.append(plastic_scalars_df.iloc[index]['X'])
                    feature_count += 1
                else:
                    if strain < current_strain_threshold:
                        filtered_elements.add(key)
                        continue
                    if ((sigma_33 - current_stress_threshold) > 0):
                        filtered_elements.add(key)
                        continue

            padding_number = self.features_per_displacement - feature_count
            for i in range(padding_number):
                sigma_33_list.append(0)
                sigma_vm_list.append(0)
                area_xy_list.append(0)
                force_contribution_list.append(0)
                dS33_dSvm_list.append(0)

                strain = 0
                #scale down boundary condition by 5% and to force error scale
                grad = zero_strain_gradient * 0.05 * abs(dErr_dFsim)
                
                gradient_list.append(grad)
                strain_list.append(strain)
                dErr_dFsim_list.append(0)
                element_list.append(0)
                time_list.append(row['X'])

            unfiltered_elements_set = initial_elements_set - filtered_elements
            contact_elements = contact_elements.union(unfiltered_elements_set)
            contact_elements_list.append(contact_elements)

        #take last element of the list to get the number of features per displacement
        self.features_per_displacement = len(contact_elements_list[-1])

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
        self.gradient_list = gradient_list
        self.input_strain_list = strain_list
        self.filtered_elements = filtered_elements

class LightRegularization(FeatureSelector):
    def __init__(self,):
        super().__init__()
        self.features_per_displacement = None

    def select_features(self, loss, force_loss_df, element_dict):
        self.loss = loss
        self.force_loss_df = force_loss_df
        self.element_dict = element_dict

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

        def dynamic_strain_threshold(X, X_min=0, X_max=1):
            threshold_min = 0.0008
            threshold_max = 0.1
            return threshold_min + ((X - X_min) / (X_max - X_min)) * (threshold_max - threshold_min)

        def dynamic_stress_threshold(X, X_min=0, X_max=1):
            stress_min = -0.5
            stress_max = -5.0
            return stress_min + ((X - X_min) / (X_max - X_min)) * (stress_max - stress_min)
        
        def find_zero_strain_gradient():

        #lower bound: 5e-05
        #upper bound: 500e-06 = 5e-04
            lower_bound = 5e-05
            upper_bound = 5e-04

            zero_strain_prediction = None
            # Find the zero-strain prediction

            for element in element_dict.values():
                # Get the plastic scalars dataframe
                plastic_scalars_df = element['plastic_scalars_df']
                
                # Filter the dataframe where PEEQ is within the specified bounds and get the MISES value
                zero_strain_prediction = plastic_scalars_df.loc[
                    (plastic_scalars_df['PEEQ'] >= lower_bound) & (plastic_scalars_df['PEEQ'] <= upper_bound), 'MISES'
                ]
                
                if not zero_strain_prediction.empty:
                    print(f'Zero strain prediction found: {zero_strain_prediction.values[0]}')
                    zero_strain_prediction_value = zero_strain_prediction.values[0]
                    break  # Exit after finding the first match

            if zero_strain_prediction is None:
                raise ValueError('No elements found within the specified bounds')
            #gradient = 2*(output - target) = 2*(zero_strain_prediction - 0)
            zero_strain_gradient = 2*zero_strain_prediction_value
            return float(zero_strain_gradient)

        X_min = self.force_loss_df['displacement'].min()
        X_max = self.force_loss_df['displacement'].max()

        initial_elements_set = set(element_dict.keys())
        contact_elements = set()
        contact_elements_list = []
        time_int_list = []

        zero_strain_gradient = find_zero_strain_gradient()

        if self.features_per_displacement is None:
            self.features_per_displacement = len(element_dict)

        for index, row in self.force_loss_df.iterrows():
            displacement = row['displacement']
            time = row['X']
            time_int = int(time)
            time_int_list.append(time_int)

            dErr_dFsim = row['dErr_dFsim']
            current_strain_threshold = dynamic_strain_threshold(displacement, X_min, X_max)
            current_stress_threshold = dynamic_stress_threshold(displacement, X_min, X_max)

            filtered_elements = set()
            feature_count = 0
            for key, element in element_dict.items():
                plastic_scalars_df = element['plastic_scalars_df']
                sigma_11 = plastic_scalars_df.iloc[index]['S11']
                sigma_22 = plastic_scalars_df.iloc[index]['S22']
                sigma_33 = plastic_scalars_df.iloc[index]['S33']
                strain = plastic_scalars_df.iloc[index]['PEEQ']


                if key in contact_elements:
                    sigma_33_list.append(sigma_33)
                    sigma_vm = plastic_scalars_df.iloc[index]['MISES']
                    sigma_vm_list.append(sigma_vm)
                    area_xy = plastic_scalars_df.iloc[index]['Area_XY']
                    area_xy_list.append(area_xy)
                    force_contribution = -area_xy * sigma_33
                    force_contribution_list.append(force_contribution)
                    dS33_dSvm = plastic_scalars_df.iloc[index]['dS33_dSvm']
                    dS33_dSvm_list.append(dS33_dSvm)

                    dFsim_dSvm = -dS33_dSvm * area_xy
                    grad = dErr_dFsim * dFsim_dSvm
                    
                    gradient_list.append(grad)
                    strain_list.append(strain)
                    dErr_dFsim_list.append(dErr_dFsim)
                    element_list.append(key)
                    time_list.append(plastic_scalars_df.iloc[index]['X'])
                    feature_count += 1
                else:
                    if strain < current_strain_threshold:
                        filtered_elements.add(key)
                        continue
                    if ((sigma_33 - current_stress_threshold) > 0):
                        filtered_elements.add(key)
                        continue

            padding_number = self.features_per_displacement - feature_count
            for i in range(padding_number):
                sigma_33_list.append(0)
                sigma_vm_list.append(0)
                area_xy_list.append(0)
                force_contribution_list.append(0)
                dS33_dSvm_list.append(0)

                strain = 0
                #scale down boundary condition by 1% and to force error scale
                grad = zero_strain_gradient * 0.01 * abs(dErr_dFsim)
                
                gradient_list.append(grad)
                strain_list.append(strain)
                dErr_dFsim_list.append(0)
                element_list.append(0)
                time_list.append(row['X'])

            unfiltered_elements_set = initial_elements_set - filtered_elements
            contact_elements = contact_elements.union(unfiltered_elements_set)
            contact_elements_list.append(contact_elements)

        #take last element of the list to get the number of features per displacement
        self.features_per_displacement = len(contact_elements_list[-1])

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
        self.gradient_list = gradient_list
        self.input_strain_list = strain_list
        self.filtered_elements = filtered_elements



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

        #TODO
        if False:
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