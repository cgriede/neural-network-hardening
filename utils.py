import matplotlib
matplotlib.use('Agg')

import os
import re
import time
import sys
import torch
import shutil
import fnmatch
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union
from typing import List, Union
from datetime import datetime
from collections import OrderedDict
from multiprocessing import Pool
from matplotlib import cm

def gradient_scaler(grad: np.array) -> np.array:
    """
    Returns array scaled to max absolute value of array scaled on interval [-1,1]
    """

    # Get the max absolute value
    max_abs = np.max(np.abs(grad))

    # Scale the array
    return grad / max_abs

def time_diff(start_time):
    def __repr__ () -> str:
        return f'time difference: {duration}'
    duration = datetime.now() - start_time
    return duration

def train_loop_validation(working_directory, archive_directory,
                          expname, strain_tensor,
                          run_sim, num_cpus):
    #aborts training if set to true for catching errors
    something_not_right = False
    kNN_epoch = []
    force_error_epoch = []
    loss_epoch = []
    #main training loop
    yield_stress = []
    for strain in strain_tensor.squeeze(1).tolist():
        yield_stress.append(two_stage_hardening_model(strain))
    yield_stress = torch.tensor(yield_stress, dtype=torch.float64).unsqueeze(1)
    mat_tensor = mat_tensor(strain_tensor, yield_stress)

    loss_df, element_dict = simulator(mat_tensor, working_directory, expname, run_simulation=run_sim, num_cpus=num_cpus)
    loss_df.set_index('X', inplace=True)
    #save the loss_df to csv file
    loss_df.to_csv(os.path.join(archive_directory, 'force_df.csv'))
    loss_df.drop('force_target', axis=1, inplace=True)
    loss_df.to_csv(os.path.join(archive_directory, 'H_10.csv'))

#Normalize PEEQ for each timestep
def normalize_peeq(element_dict):
    # Combine all PEEQ values for each timestep
    combined_df = pd.concat([element['plastic_scalars_df'] for element in element_dict.values()])
    
    # Normalize PEEQ for each timestep
    for time in combined_df['X'].unique():
        timestep_df = combined_df[combined_df['X'] == time]
        total_PEEQ = timestep_df['PEEQ'].sum()
        
        if total_PEEQ == 0:
            # Handle case where all PEEQ values are zero to avoid division by zero
            for element in element_dict.values():
                element['plastic_scalars_df'].loc[element['plastic_scalars_df']['X'] == time, 'normalized_PEEQ'] = 0
        else:
            for key, element in element_dict.items():
                mask = element['plastic_scalars_df']['X'] == time
                element_max_peeq = element['plastic_scalars_df'].loc[mask, 'PEEQ']
                normalized_peeq = element_max_peeq / total_PEEQ
                element['plastic_scalars_df'].loc[mask, 'normalized_PEEQ'] = normalized_peeq

def get_radius(name: str):
    """
    for input H_5 will return radius 5
    """
    radius = name.split('_')[1]
    return float(radius)

def get_type(name: str):
    """
    for input H_5 will return type H "hemispherical"
    """
    type = name.split('_')[0]
    return type

def mat_tensor(strain_tensor, stress_tensor):
    assert strain_tensor.shape == stress_tensor.shape, (
        f'Shape of strain_tensor :{strain_tensor.shape} does not match stress_tensor {stress_tensor.shape}'
    )

    return torch.stack((stress_tensor, strain_tensor), dim=1)

def file_picker(directory: str, name: str, extension: str = ".example"):
    """
    Picks the newest file from a directory using the base name and file extension.

    directory: Path to directory
    name: Base name of the file (e.g., H_5)
    extension: File extension (e.g., .odb)

    returns: File path (str) to the newest file or None if no such file exists
    """
    # Compile a regex pattern to match files with the given base name and extension
    pattern = re.compile(rf"{re.escape(name)}_(\d{{3}}){re.escape(extension)}")

    # List all files in the specified directory
    files = os.listdir(directory)

    # Filter for files that match the pattern
    matching_files = [file for file in files if pattern.match(file)]

    # If no matching files are found, print a message and return None
    if not matching_files:
        print(f"No files matching {name}_### {extension} found in the directory.")
        return None

    # Sort the matching files based on the numeric suffix in descending order
    matching_files.sort(key=lambda f: int(pattern.search(f).group(1)), reverse=True)

    # Get the newest file (the first one in the sorted list)
    newest_file = matching_files[0]

    # Construct the full path to the newest file
    newest_file_path = os.path.join(directory, newest_file)

    return newest_file_path

def get_next_filename(directory: str, name: str, extension: str = ".example"):
    """
    Generate the next available filename with an incrementing suffix.

    :param directory: The directory where the files are located.
    :param name: The base name of the file.
    :param extension: The file extension (default is '.example').
    :return: The next available filename with the incrementing suffix.
    """
    # Ensure the extension starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension

    # Compile a regex pattern to match files with the given base name and extension
    pattern = re.compile(rf"{re.escape(name)}_(\d{{3}}){re.escape(extension)}")

    # List all files in the specified directory
    files = os.listdir(directory)

    # Filter for files that match the pattern
    matching_files = [file for file in files if pattern.match(file)]

    # Extract the numeric suffixes and find the highest one
    max_suffix = -1
    for file in matching_files:
        match = pattern.search(file)
        if match:
            suffix = int(match.group(1))
            if suffix > max_suffix:
                max_suffix = suffix

    # Determine the next suffix
    next_suffix = max_suffix + 1
    next_suffix_str = f"_{next_suffix:03d}"

    # Generate the next filename
    next_filename = f"{name}{next_suffix_str}{extension}"
    next_filepath = os.path.join(directory, next_filename)

    return next_filepath
        
class AbaqusFunc:
    def __init__(self, working_directory, name, num_cpus = 4):
        self.working_directory = os.path.abspath(working_directory)
        self.name = name
        self.num_cpus = num_cpus

    def run_simulation(self):
        """
        Run an Abaqus simulation based on the given input file.
        """
        input_file = os.path.abspath(file_picker(self.working_directory, self.name, ".inp"))
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"The input file '{input_file}' does not exist.")

        job_name = os.path.splitext(os.path.basename(input_file))[0]
        lock_file_path = os.path.join(self.working_directory, f"{job_name}.lck")

        if os.path.exists(lock_file_path):
            print("There is a running simulation / existing lock file")
            lock = True

        command = f"abaqus job={job_name} input={input_file} double cpus={self.num_cpus} ask_delete=OFF history=odb background"

        try:
            process = subprocess.Popen(command, shell=True, cwd=self.working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()  # Capture output and error streams
            time.sleep(1)  # Wait for the lock file to be created
            if stdout:
                #print(stdout.decode())  # Print stdout
                pass
            else:
                pass#print("No output captured from subprocess.")

            if stderr:
                pass#print(stderr.decode())  # Print stderr

            # Instead of sleeping for a fixed time, check periodically for the lock file
            while os.path.exists(lock_file_path):
                print("Waiting for the lock file to be deleted...")
                time.sleep(5)  # Check every 5 seconds

            return_code = process.poll()  # Check if process has finished
            if return_code is None:
                return_code = process.wait()  # Wait for the process to finish if it hasn't yet

            if return_code != 0:
                print(f"Error: Simulation failed with return code {return_code}")
                return False
            else:
                #print(f"Simulation {job_name} ended successfully.")
                return True
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the simulation: {e}")
            return False


    def extract_output(self):

        odb_file = file_picker(self.working_directory, self.name, ".odb" )
        extract_output = os.path.abspath(f"{self.name}_extract_output.py")
        command = f"abaqus cae noGUI={extract_output} -- {odb_file} {self.working_directory}"

        try:
            result = subprocess.run(command, check=True, shell=True, cwd=self.working_directory, capture_output=True, text=True)
            print(f"Output extraction for {odb_file} completed successfully.")
            print("STDOUT:", result.stdout)  # To see the output
            print("STDERR:", result.stderr)  # To see any error messages
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while extracting output data: {e}")
            print("STDOUT:", e.stdout)  # To see the output
            print("STDERR:", e.stderr)  # To see any error messages
            
def interpolate_nan(tensor):
    for col in range(tensor.shape[1]):  # Iterate over columns
        for row in range(1, tensor.shape[0]):  # Start from the second row
            if torch.isnan(tensor[row, col]):
                # Find the nearest non-NaN value above
                above_row = row - 1
                while above_row >= 0 and torch.isnan(tensor[above_row, col]):
                    above_row -= 1
                
                # Find the nearest non-NaN value below
                below_row = row + 1
                while below_row < tensor.shape[0] and torch.isnan(tensor[below_row, col]):
                    below_row += 1
                
                # Interpolate the NaN value using linear interpolation
                if above_row >= 0 and below_row < tensor.shape[0]:
                    tensor[row, col] = (tensor[above_row, col] + tensor[below_row, col]) / 2
                elif above_row >= 0:
                    tensor[row, col] = tensor[above_row, col]
                elif below_row < tensor.shape[0]:
                    tensor[row, col] = tensor[below_row, col]

class AbaqusInputFile:
    def __init__(self, directory_path, name):
        
        self.directory_path = directory_path
        self.name = name
        self.file_path = file_picker(self.directory_path, name, '.inp')
        self.lines = []
        self.read_file()
        self.output_path = os.path.dirname(os.path.join(directory_path, '.inp'))  # directory of the input file

    def read_file(self):
        """Read the Abaqus input file and store its lines."""
        with open(self.file_path, 'r') as file:
            self.lines = file.readlines()
    
    def get_next_filename(self):
        """Generate the next available filename with an incrementing suffix."""
        base_name, ext = os.path.splitext(self.name)
        i = 0
        while True:
            suffix = f"_{i:03d}" if i > 0 else ""
            new_name = f"{base_name}{suffix}{ext}"  # Adjusted line
            new_path = os.path.join(self.output_path, new_name)
            if not os.path.exists(new_path):
                return new_path
            i += 1

    def write_file(self):
        """Write the current lines to a new Abaqus input file with an incrementing suffix if needed."""
        output_path = get_next_filename(self.directory_path, self.name, '.inp')
        with open(output_path, 'w') as file:
            file.writelines(self.lines)

    def replace_parameter(self, parameter_name, old_value, new_value):
        """Replace a parameter value in the file."""
        for i, line in enumerate(self.lines):
            if parameter_name in line:
                self.lines[i] = line.replace(old_value, new_value)

    def change_plastic(self, tensor):
        # Validate tensor values are positive
        assert (tensor >= 0).all(), "All tensor values must be positive."   

        start_line = 0
        end_line = 0
        n_inserts = 0
        start_found = False
        
        # Search for the start and end lines
        for i, line in enumerate(self.lines):
            if line == "*Plastic\n":
                start_line = i + 1
                start_found = True
            elif line.strip() == "**" and start_found:
                end_line = i - 1
                break  # Stop searching

        # Check if the start line was found
        if not start_found:
            raise ValueError("The '*Plastic' line was not found in the input lines.")

        # Insert or update lines in the range
        for i, ss_pair in enumerate(tensor):
            line = start_line + i
            if line <= end_line:
                self.lines[line] = f"{tensor[i, 0].item()},  {tensor[i, 1].item()}\n"
            else:
                self.lines.insert(line, f"{tensor[i, 0].item()},  {tensor[i, 1].item()}\n")
                n_inserts += 1

        # Ensure the end line plus the number of inserts matches the expected final line
        assert end_line + n_inserts == line, f"Expected end_line to be {line + 1 + 1}, but got {end_line + n_inserts + 1}"



    def add_section(self, section_name, data_lines, before_section=None):
        """Add a new section to the file."""
        new_section = [f'*{section_name}\n'] + [f'{line}\n' for line in data_lines]
        
        if before_section:
            # Find the position to insert the new section
            try:
                insert_position = self.lines.index(f'*{before_section}\n')
            except ValueError:
                raise ValueError(f"Section '{before_section}' not found in the input file.")
            self.lines = self.lines[:insert_position] + new_section + self.lines[insert_position:]
        else:
            # Append the new section at the end
            self.lines.extend(new_section)

    def write_contact_element_set_input(contact_elements_list, time_int_list, time_intervals, filename='contact_element_sets.inp'):
        """
        Writes contact element sets to an input file for specified time intervals.

        Parameters:
        contact_elements_list (list of sets): List of sets containing contact elements for each time step.
        force_loss_df (DataFrame): DataFrame containing the simulation data with a column 'X' representing time.
        time_intervals (list of int): List of time intervals at which to write the element sets.
        filename (str): Name of the output file.
        """
        with open(filename, 'w') as file:
            for index, time in enumerate(time_int_list):
                if time not in time_intervals:
                    continue
                # Find the corresponding index for the time interval
                contact_elements = contact_elements_list[index]
                
                # Write the header for the element set
                file.write(f"*Elset, elset=ContactElements_{time}\n")
                
                # Write the elements in rows of 8
                elements = list(contact_elements)
                for i in range(0, len(elements), 8):
                    line = ', '.join(f'{el:6}' for el in elements[i:i+8])
                    file.write(f'  {line}\n')



class ExperimentData:
    def __init__(self, directory_path, name):
        self.name = name
        self.directory_path = directory_path
        self.df = None
        self.folder_name = 'exp_data'
        self.read_csv()

    def __repr__(self):
        return (f"\n"
                f"ExperimentData(Name: {self.name}, "
                f"folder_name={self.folder_name}, "
                f"data_shape={self.df.shape}")

    def read_csv(self):
        # Construct the file path
        file_path = os.path.join(self.directory_path, self.folder_name, self.find_matching_file())
        # Read CSV file
        self.df = pd.read_csv(file_path, header=None)
        
        # Keep only the first three columns
        self.df = self.df.iloc[:, :3]
        
        self.df.columns = ['X', 'displacement', 'force_target']
        
        # Normalize the time column (let time start at 0, for reading convenience)
        self.df['X'] = self.df['X'] - self.df['X'].iloc[0]

        #use positive displacements (same coordinate system as abaqus)
        self.df['displacement'] = self.df['displacement']
        
        
        
    def find_matching_file(self):
        exp_data = os.listdir(os.path.join(self.directory_path, self.folder_name))
        for file in exp_data:
            if file == f'{self.name}.csv':
                return file
class AbaqusOutputFile:
    def __init__(self, directory_path, name):

        self.directory_path = directory_path  # Folder path
        self.name = name

        self.rpt_path = file_picker(directory_path, self.name, extension=".rpt")
        self.nmp_path = file_picker(directory_path, self.name, extension=".nmp")
        
        self.lines = []
        self.data_sections = {}

        self.initial_distance = 0.1
        
        
        time_parse_start = datetime.now()
        self.read_file()
        self.parse_sections()
        print(f"Time to parse sections: {datetime.now() - time_parse_start}")

        self.model_data_df = None
        """
        creates a dataframe with columns: X, displacement, force_sim, force_target

        where X stands for time in seconds, used in the simulation, force_sim is interpolated from experiment data along the 
        displacement values, force_target is the real force from experiment data

        assuming quasistatic behaviour interpolating along displacement values is a valid approach
        (ref from tancogne-dejean, experiment data)
        time is only kept as a reference for the simulation

        the chosen amplitude of the abaqus simulation is a smooth step,
        (sinusoidal behaviour is not considered in this approach)

        it is to be determined whether the linear adjustemnt of displacement or time values
        is necessary for an effective training procedure of the neural network model, as
        there might be a bias to specific timepoints in the simulation if the time values are kept
        unevenly spaced
        
        """
        self.model_data_df = self.create_model_data()

        time_node_data_start = datetime.now()
        #Node displacement df
        self.node_data = {}
        self.create_node_data()
        print(f"Time to create node data: {datetime.now() - time_node_data_start}")

        self.element_data = {}

        #create df with information about the elements and make the dict entry for element_id, 'plastic_scalars_df'
        time_element_data_start = datetime.now()
        self.create_element_df()
        print(f"Time to create element data: {datetime.now() - time_element_data_start}")

        time_calc_partial_derivative_start = datetime.now()
        self.calc_partial_derivative()
        print(f"Time to calculate partial derivative: {datetime.now() - time_calc_partial_derivative_start}")
        self.sort_dict_keys()

        #2create element dict
        self.get_element_info()

        time_calc_area = datetime.now()
        self.calc_area2()
        print(f"Time to calculate area: {datetime.now() - time_calc_area}")

        #align element and model data (plastic scalars / force-displacement)
        self.align_model_element()

        #cut 0 force rows to have meaningful data only (no data where indenter is "above" battery)
        self.cut_zero_force_rows()
    
    @staticmethod
    def is_number(s):
        """Check if the string s is a number."""
        try:
            float(s)
            return True
        except ValueError:
            return False
  
    

    
    @staticmethod
    def dS33_dSvm(S11, S22, S33, Svm):
        """
        returns dS33 / dSvm (sensitivity for S33 for change in Von-mises-equivalent stress)
        """
        q = (2*S33 - S11 -S22)
        if q != 0 : return 2*(Svm) / q
        else: return 0
    

    def __repr__(self):
        return f'AbaqusOutputFile: {self.name}'

    def read_file(self):
        """Read the Abaqus output file and store its lines."""
        with open(self.rpt_path, 'r') as file:
            self.lines = file.readlines()

    def get_element_info(self):
            # Read the content of the output file
            with open(self.nmp_path, 'r') as file:
                element_node_info = file.readlines()


            # Iterate over each line in the output file content
            for line in element_node_info:
                # Parse the line to extract element and node information
                #sample line = 'Element 2: Node 139 (Initial Position: [27.904762 29.772728  5.07    ])'

                parts = line.split(': ')

                #returns element id
                el_pat = re.compile(r'Element (\d+)')
                el_id = re.match(el_pat, parts[0])
                if el_id: el_id = int(el_id.group(1))

                
                #returns node id
                node_pat = re.compile(r'Node (\d+)')
                node_id = re.match(node_pat, parts[1])
                if node_id: node_id = int(node_id.group(1))

                
                #returns list of coordinates [x,y,z]
                ncoord_pat = re.compile(r'\[(.*?)\]')
                ncoord = re.match(ncoord_pat, parts[2])
                if ncoord: ncoord = ncoord.group(1)
                ncoord = re.split(r'\s+', ncoord)


                valid_coords = []
                for coord in ncoord:
                    try:
                        valid_coords.append(float(coord))
                    except ValueError:
                        pass  # Ignore non-float coordinates

                assert len(valid_coords) == 3, f'expected 3 coordinates not {len(ncoord)} for element: {el_id}, node: {node_id}'
                for coord in valid_coords:
                    assert isinstance(coord, (float, np.float64)), "coord must be a float or numpy float"
                

                # Assign the node coordinates to the specific node ID
                if el_id in self.element_data:
                    self.element_data[el_id]['nodes'][node_id] = {'node_coord': valid_coords}
                
    def sort_dict_keys(self):
        #sort the keys in ascending order for correct Area calculation
        for el_id in self.element_data:
            # Sort the nodes by key (node_id)
            sorted_nodes = OrderedDict(sorted(self.element_data[el_id]['nodes'].items()))

            # Update the element_data dictionary with the sorted nodes
            self.element_data[el_id]['nodes'] = sorted_nodes
            
    def parse_sections(self):
        """Parse the file into different sections based on five consecutive new lines."""
        section_data = []
        empty_line_count = 0
        section_index = 0
        in_section = False

        for line in self.lines:
            if line.strip() == "":
                empty_line_count += 1
            else:
                empty_line_count = 0
                if not in_section:
                    in_section = True
                    section_data = []

            if empty_line_count < 5:
                if in_section:
                    section_data.append(line.strip())
            else:
                if in_section:
                    # Save the previous section
                    self.data_sections[f'Section {section_index}'] = section_data
                    section_index += 1
                    in_section = False

        # Add the last section if it exists
        if section_data:
            self.data_sections[f'Section {section_index}'] = section_data
    
    def cut_zero_force_rows(self):
            cut_index= 0
            for index, value in self.model_data_df['force_sim'].items():
                if value <= 0:
                    cut_index = index
                else:
                    break
            #adjust model data
            self.model_data_df = self.model_data_df.iloc[cut_index:].reset_index(drop=True)
            #adjust element data
            for element in self.element_data.values():
                element['plastic_scalars_df'] = element['plastic_scalars_df'].iloc[cut_index:].reset_index(drop=True)

    def create_model_data(self):
        """Create a DataFrame for the model data based on the first relevant section."""
        for section_data in self.data_sections.values():
            if 'displacement' in section_data[0] and 'reaction-force' in section_data[0]:
                # Extract the header and data lines
                data = []
                for line in section_data[2:]:  # Skip the first two lines (header and underline)
                    columns = re.split(r'\s+', line)
                    if len(columns) == 3:
                        x_str, displacement_str, force_str = columns

                        # Convert to appropriate values, handling 'NoValue'
                        x = float(x_str) if self.is_number(x_str) else float('nan')
                        displacement = float(displacement_str.replace(',', '')) if 'NoValue' not in displacement_str else float('nan')
                        force = float(force_str.replace(',', '')) if 'NoValue' not in force_str else float('nan')
                        force *= -4  # Scaling factor for the force due to symmetry, Reaction force is in the negative direction
                        data.append((x, displacement, force))

                # Create DataFrame
                model_data_df = pd.DataFrame(data, columns=["X", "displacement", "force_sim"])

                # Retrieve experimental data for interpolation
                exp_data = ExperimentData(self.directory_path, self.name)
                exp_df = exp_data.df

                # Convert to N
                exp_df['force_target'] = 1000*exp_df['force_target']  

                interpolated_forces = []

                #convert to positive displacements
                model_data_df['displacement'] = -model_data_df['displacement']
                #adjust for initial distance between indenter and battery model 0.1mm
                model_data_df['displacement'] = model_data_df['displacement'] - self.initial_distance

                # Match experimental forces to output
                for disp in model_data_df['displacement']:
                    # Find the two closest displacements in exp_df
                    closest_indices = np.argsort(np.abs(exp_df['displacement'] - disp))[:2]
                    closest_displacements = exp_df.iloc[closest_indices]

                    if len(closest_displacements) < 2:
                        interpolated_force = closest_displacements['force_target'].values[0]
                    else:
                        x1, x2 = closest_displacements['displacement'].values
                        y1, y2 = closest_displacements['force_target'].values

                        # Linear interpolation formula
                        interpolated_force = y1 + (y2 - y1) * (disp - x1) / (x2 - x1)

                    interpolated_forces.append(interpolated_force)

                # Add the interpolated forces to model_data_df
                model_data_df['force_target'] = interpolated_forces

                # Return the final DataFrame with model data
                return model_data_df

        # If no relevant section found, return an empty DataFrame
        return pd.DataFrame(columns=["X", "displacement", "force_sim", "force_target"])

    def create_element_df(self):
        """Create DataFrames and nested dictionaries for each element."""
        for section_name, section_data in self.data_sections.items():
            # Check the content of section_data for element information
            element_match = None
            for line in section_data:
                element_match = re.search(r'Elem: (\d+)', line)
                if element_match:
                    break

            if element_match:
                # Extract the element ID
                element_id = int(element_match.group(1))

                # Process plasticity scalars
                data = []
                for line in section_data:
                    columns = re.split(r'\s+', line.strip())
                    if all(self.is_number(col) for col in columns):
                        data.append([float(col) for col in columns])
                if data:
                    df = pd.DataFrame(data, columns=["X", "PEEQ", "MISES", "S11", "S22", "S33", "S12", "S13", "S23"])
                    
                    #Drop elements with all strains strictly lower than 0.005, fiter for non-zero strain data
                    threshold = 0.005
                    if df['PEEQ'].max() > threshold:
                        self.element_data[element_id] = {'plastic_scalars_df': df, 'nodes': {}}


    def calc_area2(self):
        # Calculate area for each element and timestep
        for element_id, data in self.element_data.items():
            plastic_scalars_df = data['plastic_scalars_df']
            nodes = data['nodes']
            
            # Precompute node coordinates
            node_coords = {node_id: np.array(coords['node_coord']) for node_id, coords in nodes.items()}
            
            # Initialize result DataFrame
            result_df = pd.DataFrame(index=plastic_scalars_df.index)
            
            # Extract displacements for all nodes and timepoints
            displacements = {node_id: self.node_data[node_id][['U1', 'U2']].values for node_id in nodes.keys()}
            
            # Iterate over timepoints
            for index in plastic_scalars_df.index:
                x_list = []
                y_list = []

                # Compute new coordinates for each node
                for node_id, coords in node_coords.items():
                    displacement = displacements[node_id][index]
                    x, y = coords[:2] + displacement
                    x_list.append(x)
                    y_list.append(y)

                assert len(x_list) == 4, f"element: {element_id} has not 4 nodes, but {len(x_list)} nodes"

                # Convert lists to NumPy arrays for vectorized operations
                x_array = np.array(x_list)
                y_array = np.array(y_list)

                # Sort points counterclockwise
                x_array, y_array = self.sort_points_counterclockwise2(x_array, y_array)

                # Calculate area using vectorized operations
                area_xy = self.calculate_quadrilateral_area2(x_array, y_array)
                result_df.loc[index, 'Area_XY'] = area_xy
            
            # Add area DataFrame to plastic_scalars DataFrame
            self.element_data[element_id]['plastic_scalars_df'] = pd.concat([plastic_scalars_df, result_df], axis=1)

    def sort_points_counterclockwise2(self, x_array, y_array):
        # Calculate the centroid
        centroid_x = np.mean(x_array)
        centroid_y = np.mean(y_array)
        
        # Calculate the angle of each point relative to the centroid
        angles = np.arctan2(y_array - centroid_y, x_array - centroid_x)
        
        # Sort points by angle
        sorted_indices = np.argsort(angles)
        return x_array[sorted_indices], y_array[sorted_indices]

    def calculate_quadrilateral_area2(self, x_array, y_array):
        # Assuming the points are ordered correctly, calculate the area
        # Using the Shoelace formula for a quadrilateral
        return 0.5 * np.abs(
            x_array[0] * y_array[1] + x_array[1] * y_array[2] + 
            x_array[2] * y_array[3] + x_array[3] * y_array[0] -
            (y_array[0] * x_array[1] + y_array[1] * x_array[2] + 
            y_array[2] * x_array[3] + y_array[3] * x_array[0])
        )


    def calc_partial_derivative(self):
    #calculate dS33/dSvm and insert to df
        for element_id, data in self.element_data.items():
            # Get the plastic scalars DataFrame
            df = data['plastic_scalars_df']
            # Calculate dS33_dSvm for each row and store it in a new column
            df['dS33_dSvm'] = df.apply(lambda row: self.dS33_dSvm(row['S11'], row['S22'], row['S33'], row['MISES']), axis=1)

    def create_node_data(self):
        """Create DataFrames and nested dictionaries for each node."""
        for section_name, section_data in self.data_sections.items():
            # Check the content of section_data for node information
            node_match = None
            for line in section_data:
                node_match = re.search(r'node: (\d+)', line)
                if node_match:
                    break

            if node_match:
                # Extract the node ID
                node_id = int(node_match.group(1))

                # Process displacements
                data = []
                for line in section_data:
                    columns = re.split(r'\s+', line.strip())
                    if all(self.is_number(col) for col in columns):
                        data.append([float(col) for col in columns])
                if data:
                    df = pd.DataFrame(data, columns=["X", "U1", "U2", "U3"])
                    self.node_data[node_id] = df
 
    def align_model_element(self):
            #sort the element data dict by ascending element ids
            self.element_data = OrderedDict(self.element_data)

            first_key = next(iter(self.element_data))
            df2 = self.element_data[first_key]['plastic_scalars_df'].copy()

            # Step 1: Set index to 'X' for both DataFrames
            self.model_data_df.set_index('X', inplace=True)
            df2.set_index('X', inplace=True)

            # Step 2: Interpolate df1 to match df2's index (time intervals)
            df1_interpolated = self.model_data_df.interpolate(method='index')

            # Step 3: Align df1_interpolated with df2 and reset index to match original structure
            self.model_data_df = df1_interpolated.reindex(df2.index).reset_index()
            df2 = df2.reset_index()

class AbaqusOutputValidationFile(AbaqusOutputFile):
    def __init__(self, directory_path, name):
        self.directory_path = directory_path  # Folder path
        self.name = name

        self.rpt_path = file_picker(directory_path, self.name, extension=".rpt")

        self.lines = []
        self.data_sections = {}

        self.initial_distance = 0.1
        
        
        time_parse_start = datetime.now()
        self.read_file()
        self.parse_sections()
        print(f"Time to parse sections: {datetime.now() - time_parse_start}")

        self.model_data_df = None
        self.model_data_df = self.create_model_data()

def abaqus_runtime_status(directory_name, sim_name):
    """
    Return the status of an Abaqus simulation that has been called
    via the command line, based on the .log and .lck files 
    that have been created. For this function to work the simulation must
    have been called with the 'background' option turned on, so that a .log
    file is created.

    It is possible for the status to be unsuccessfully finished while the
    simulation has not started yet, depending on the timing of files
    appearing or disappearing. In order to definitively decide that a
    simulation has terminated without success it is advised to check that its
    status is 'UNSUCCESSFULLY_FINISHED' for several consecutive calls of this function.
    """
    original_dir = os.getcwd()
    os.chdir(directory_name)
    
    namelck = f"{sim_name}.lck"
    namelog = f"{sim_name}.log"
    nameslurm = f"{sim_name}_slurm.log"

    # Check for exception files
    exception_pattern = "explicit_dp.*.exception"
    exception_files = fnmatch.filter(os.listdir(os.getcwd()), exception_pattern)
    
    # Check if log file exists
    if not os.path.exists(namelog):
        status = 'NOT_STARTED'
    if exception_files:
        status = 'UNSUCCESSFULLY_FINISHED'
    else:
        # Check if lock file exists
        if not os.path.exists(namelck):
            with open(namelog, 'r') as file:
                textlog = file.read()
            if 'COMPLETED' not in textlog:
                status = 'UNSUCCESSFULLY_FINISHED'
            else:
                status = 'SUCCESSFULLY_FINISHED'
        else:
            # Lock file exists; simulation is either running or waiting for licenses,
            # unless slurm has failed, in which case the slurm error log file will have content
            failed_from_slurm = False
            if os.path.exists(nameslurm):
                if os.path.getsize(nameslurm) > 0:
                    failed_from_slurm = True
            if not failed_from_slurm:
                with open(namelog, 'r') as file:
                    textlog = file.read()
                if 'Abaqus License Manager checked out the following licenses:' not in textlog:
                    status = 'QUEUED_FOR_LICENSE'
                else:
                    status = 'RUNNING'
            else:
                status = 'UNSUCCESSFULLY_FINISHED'
    
    os.chdir(original_dir)
    return status



def simulator(mat_prop_tensor, working_directory, name, run_simulation = False, num_cpus = 4):
    """
    input: MatProp Tensor, Name: e.g. H_5, 
    output: Tensor with values: "displacement" "Simulated Force" "Real Force"

    navigates to working directory

    locates Name.inp file, adjusts Material properties, creates Name_0xx.inp file

    runs simulation on Name_0xx.inp file

    extracts time force displacement tensor from Name_0xx.odb file

    creates experimental displacement interpolated tensor (matches simulated forces to experimental displacement values)

    returns df, dict
    df: X, displacement, force_sim, force_target
    dict: plastic_scalars_df / nodes
    plastic_scalars_df[element]: MISES, PEEQ, S11, S22, S33, Area_xy
    """

    start_time = datetime.now()
    if run_simulation:
        inp = AbaqusInputFile(working_directory, name)
        inp.change_plastic(mat_prop_tensor)
        inp.write_file()
        abq = AbaqusFunc(working_directory=working_directory, name=name, num_cpus=num_cpus)

        fail_counter = 0
        ABORT = False
        status = 'NOT_STARTED'
        while status != 'SUCCESSFULLY_FINISHED':
            print(f"Running simulation for {name}, Run_simulation = {run_simulation}")
            abq.run_simulation()
            job_log_path = file_picker(working_directory, name, '.log')
            job_name = os.path.basename(job_log_path).split('.')[0]
            status = abaqus_runtime_status(working_directory, job_name)
            if status != 'SUCCESSFULLY_FINISHED':
                print(f'simulation failed {fail_counter + 1} times, repeating simulation for {name}')
                fail_counter += 1

                #move the exception file to error_log folder if it exists
                exception_pattern = "explicit_dp.*.exception"
                exception_files = fnmatch.filter(os.listdir(working_directory), exception_pattern)
                if exception_files:
                    for file in exception_files:
                        shutil.move(os.path.join(working_directory, file), os.path.join(working_directory, 'error_log', file))
                #for 5 consecutive fails, abort the simulation process        
                if fail_counter > 4:
                    print(f"Simulation failed for {name} after 5 attempts, sending abort signal")
                    return pd.DataFrame(), {}
    
        print(f'time for simulation: {datetime.now() - start_time}')
        start_time_extract = datetime.now()
        abq.extract_output()
        print(f'time for output extraction: {datetime.now() - start_time_extract}')

    outp = AbaqusOutputFile(working_directory, name)
    df2 = outp.model_data_df

    return df2, outp.element_data


class TestrunSummaryWriter:
    def __init__(self, archive_directory, working_directory, validation=False, test_name= None, test_description = None, optimizer= None, train_loop_description= None, expname = None,
                 learning_rate= None, number_of_epochs= None, model=None, ):
        self.validation = validation
        self.archive_directory = archive_directory
        self.working_directory = working_directory
        #ensure the archive directory exists
        self.ensure_archive_exists()
        #generate subdirectory in the archive directory
        date = datetime.now().strftime('%Y%m%d_%H')
        count = 0
        while True:
            self.out_dir = os.path.join(archive_directory, f"{date}_{count}_{test_name}")
            if not os.path.exists(self.out_dir):
                break
            count += 1
        os.makedirs(self.out_dir)
        self.parent_out_directory = self.out_dir
        
        #initialize the loss_df_dict
        self.loss_df_dict = {}

        if not validation:
            # Inputs for the report
            self.test_name = test_name
            self.test_description = test_description
            self.optimizer = optimizer
            #get str representation of optimizer
            self.optimizer = self.optimizer.__class__.__name__
            self.train_loop_description = train_loop_description
            self.expname = expname
            self.learning_rate = learning_rate
            self.number_of_epochs = number_of_epochs
            self.model = model
            
            # Initialize empty lists for metrics
            self.loss_epoch_dict = {}
            self.mat_tensor_dict = {}
            self.backprop_element_df_dict = {}
            self.filtered_elements_dict = {}
            self.model_dict = {}
            self.save_epoch = False

            self.loss_epoch_list = []

            self.master_df = pd.DataFrame()

            #create epochs list to save in memory
            self.evenspaced_epochs = self.evenspaced_epochs()
            self.epoch_loss_dict = {}

            self.epochs_to_remove = set()

    def ensure_archive_exists(self):
        if not os.path.exists(self.archive_directory):
            os.makedirs(self.archive_directory)
    
    def validation_summary(self, title, loss_df):
        self.loss_df_dict[0] = loss_df
        self.force_plot(title=title, epochs_to_plot=0)

    def checkpoint(self,
        current_epoch: int,
        loss_epoch: float,
        removed_elements: set,
        mat_tensor: torch.Tensor,
        model: torch.nn.Module,
        loss_df: pd.DataFrame,
        backprop_element_df: pd.DataFrame,
        save_all_epochs: bool = False,
        ) -> None:

        """
        store all data from epochs with a lower loss:
        store model (with weights)
        store simulation data
        store debug data
        store mat_tensor
        saves the data to the archive directory for epochs where the loss was at a local minmum or inflection point
        """
        #always append the loss to the loss_epoch_list
        self.loss_epoch_list.append(loss_epoch)

        current_loss = loss_epoch
    
        save_epoch = False
        if self.update_lowest_loss(current_epoch, current_loss):
            save_epoch = True
        
        elif current_epoch in self.evenspaced_epochs:
            save_epoch = True
        

        if save_epoch:
            # Update metrics
            self.loss_epoch_dict[current_epoch] = loss_epoch
            self.loss_df_dict[current_epoch] = loss_df
            self.mat_tensor_dict[current_epoch] = mat_tensor
            self.backprop_element_df_dict[current_epoch] = backprop_element_df
            self.filtered_elements_dict[current_epoch] = removed_elements
            self.model_dict[current_epoch] = model
            self.data_saver(current_epoch)

        for epoch in self.epochs_to_remove:
            if epoch in self.evenspaced_epochs:
                continue
            del self.loss_epoch_dict[epoch]
            del self.loss_df_dict[epoch]
            del self.mat_tensor_dict[epoch]
            del self.backprop_element_df_dict[epoch]
            del self.filtered_elements_dict[epoch]
            del self.model_dict[epoch]
            if not save_all_epochs:
                self.delete_checkpoint(epoch)
            #reset the epochs to remove set
            self.epochs_to_remove = set()


    def update_lowest_loss(self, epoch, loss, num_epochs: int = 5) -> bool:
        """Keeps track of the epochs with the lowest losses."""
        if len(self.epoch_loss_dict) < num_epochs:
            self.epoch_loss_dict[epoch] = loss
            return True
        else:
            max_loss_epoch = max(self.epoch_loss_dict, key=self.epoch_loss_dict.get)
            if loss < self.epoch_loss_dict[max_loss_epoch]:
                print(f"Removing epoch {max_loss_epoch} with loss {self.epoch_loss_dict[max_loss_epoch]}")
                self.epochs_to_remove.add(max_loss_epoch)
                del self.epoch_loss_dict[max_loss_epoch]
                self.epoch_loss_dict[epoch] = loss
                return True
        return False

    def lr_scheduler(self, epoch):
        pass

    def delete_checkpoint(self, epoch: int):
        """
        Deletes the checkpoint for the specified epoch.

        Args:
            epoch (int): Epoch for which the checkpoint should be deleted.

        Returns:
            None
        """
        epoch_dir = os.path.join(self.parent_out_directory, f'epoch_{epoch}')
        if os.path.exists(epoch_dir):
            shutil.rmtree(epoch_dir)
        else:
            if (epoch == -1) or (epoch is None):
                #nothing to delete for the first epoch
                pass
            else:
                raise FileNotFoundError(f"Checkpoint for epoch {epoch} does not exist")
            
    def data_saver(self, epoch):
        #create the epoch directory and assign to output directory
        epoch_dir = os.path.join(self.parent_out_directory, f'epoch_{epoch}')
        os.makedirs(epoch_dir)
        self.out_dir = epoch_dir

        # Save the model
        model_path = os.path.join(self.out_dir, f'model.pt')
        torch.save(self.model.state_dict(), model_path)

        # Save the simulation data
        odb = file_picker(self.working_directory, self.expname, '.odb')
        shutil.copy(odb, self.out_dir)
        inp = file_picker(self.working_directory, self.expname, '.inp')
        shutil.copy(inp, self.out_dir)

        #save the filtered elements
        checkpoint_log_path = os.path.join(self.out_dir, 'checkpoint_log.txt')
        self.write_checkpoint_log(self.loss_epoch_list[epoch], epoch, checkpoint_log_path)
        filtered_elements = self.filtered_elements_dict[epoch]
        self.write_filtered_elements_to_file(filtered_elements, checkpoint_log_path)

        # Save the debug data
        self.save_debug_frames(epochs_to_save=epoch)
        self.kNN_plt(title=f'epoch {epoch + 1}', epochs_to_plot=epoch)
        self.force_plot(title=f'epoch {epoch + 1}', epochs_to_plot=epoch)

    def calculate_memory_usage(self):
        total_memory = 0

        # Calculate memory usage for loss_epoch_list (floats)
        size_loss_epoch = sum(sys.getsizeof(item) for item in self.loss_epoch_list)
        total_memory += size_loss_epoch

        # Calculate memory usage for loss_df_list (DataFrames)
        size_loss_df = sum(sys.getsizeof(df) + df.memory_usage(index=True).sum() for df in self.loss_df_list)
        total_memory += size_loss_df

        # Calculate memory usage for mat_tensor_list (tensors)
        size_mat_tensor = sum(sys.getsizeof(tensor) + tensor.element_size() * tensor.nelement() for tensor in self.mat_tensor_list)
        total_memory += size_mat_tensor

        # Calculate memory usage for backprop_element_df_list (DataFrames)
        size_backprop_element_df = sum(sys.getsizeof(df) + df.memory_usage(index=True).sum() for df in self.backprop_element_df_list)
        total_memory += size_backprop_element_df

        # Calculate memory usage for filtered_elements_list (lists of integers)
        size_filtered_elements = sum(sys.getsizeof(lst) + sum(sys.getsizeof(e) for e in lst) for lst in self.filtered_elements_list)
        total_memory += size_filtered_elements

        # Calculate memory usage for model_list (models)
        size_model = sum(sys.getsizeof(model) + sum(p.numel() * p.element_size() for p in model.parameters()) for model in self.model_list)
        total_memory += size_model

        print(f"Total memory usage (postprocess data): {total_memory / (1024 ** 2):.2f} MB")


    @staticmethod
    def write_checkpoint_log(loss: float, epoch: int, file_path: str) -> None:
        """
        Writes the loss and epoch number to a file.

        Args:
            loss (float): Loss value for the epoch.
            epoch (int): Epoch number.

        Returns:
            None
        """
        with open(file_path, 'a') as file:
            file.write(f"Epoch: {epoch}\n")
            file.write(f"Loss: {loss}\n")
    @staticmethod
    def write_filtered_elements_to_file(filtered_elements: set, file_path: str) -> None:
        """
        Writes the filtered elements to a file in the specified format.

        Args:
            filtered_elements (set): Set of integers representing filtered elements.
            file_path (str): Path to the file where the output should be written.

        Returns:
            None
            """
        # Convert the set to a sorted list
        sorted_elements = sorted(filtered_elements)

        # Create a list to hold the formatted elements
        formatted_elements = []

        # Iterate through the sorted elements and add a newline every 20 elements
        for i in range(0, len(sorted_elements), 20):
            chunk = sorted_elements[i:i+20]
            formatted_elements.append(', '.join(map(str, chunk)) + ',')

        # Join the chunks with newlines, ensuring a comma is added at the end of each line
        elements_str = '\n    '.join(formatted_elements)

        # Format the string, and add an additional comma at the end if required
        formatted_str = f"filtered elements: {{\n{elements_str}\n}}"

        # Write to file
        with open(file_path, 'a') as file:
            file.write(formatted_str)

    def color_picker(self, epoch):
        colormap = plt.cm.viridis  # You can choose any colormap you like
        total_epochs = self.number_of_epochs
        return colormap(epoch / total_epochs)


    def write_summary(self):
        #reinitialize output directory
        self.out_dir = os.path.join(self.parent_out_directory, 'summary')
        os.makedirs(self.out_dir)

        evenspaced_epochs = self.evenspaced_epochs
        self.save_debug_frames(epochs_to_save=evenspaced_epochs)
        self.epoch_loss_plt()
        self.epoch_logloss_plt()
        self.kNN_plt(title= 'epoch selection',epochs_to_plot=evenspaced_epochs)
        self.force_plot(title= 'epoch selection', epochs_to_plot=evenspaced_epochs)
        self.write_report()

        self.out_dir = os.path.join(self.parent_out_directory, 'lowest_losses')
        os.makedirs(self.out_dir)

        lowest_loss_epochs = self.epoch_loss_dict
        self.save_debug_frames(epochs_to_save=lowest_loss_epochs)
        self.kNN_plt(title= 'lowest losses',epochs_to_plot=lowest_loss_epochs)
        self.force_plot(title= 'lowest losses', epochs_to_plot=lowest_loss_epochs)

    def old_write_summary(self):
        #reinitialize output directory
        self.out_dir = os.path.join(self.parent_out_directory, 'summary')
        os.makedirs(self.out_dir)

        evenspaced_epochs = self.evenspaced_epochs
        self.save_debug_frames(epochs_to_save=evenspaced_epochs)
        self.epoch_loss_plt()
        self.epoch_logloss_plt()
        self.kNN_plt(title= 'epoch selection',epochs_to_plot=evenspaced_epochs)
        self.force_plot(title= 'epoch selection', epochs_to_plot=evenspaced_epochs)
        self.write_report()

        self.out_dir = os.path.join(self.parent_out_directory, 'lowest_losses')
        os.makedirs(self.out_dir)

        lowest_loss_epochs = self.epoch_loss_dict
        self.save_debug_frames(epochs_to_save=lowest_loss_epochs)
        self.kNN_plt(title= 'lowest losses',epochs_to_plot=lowest_loss_epochs)
        self.force_plot(title= 'lowest losses', epochs_to_plot=lowest_loss_epochs)


    def evenspaced_epochs(self):
        epochs_list = []
        if self.number_of_epochs <= 10:
            return list(range(self.number_of_epochs))
        else:
            plot_increment = self.number_of_epochs // 10
            for i in range(plot_increment - 1, self.number_of_epochs, plot_increment):
                epochs_list.append(i)
            if epochs_list[-1] != self.number_of_epochs - 1:  # Ensure last epoch is included
                epochs_list[-1] = self.number_of_epochs - 1
            return epochs_list

    
    def old_force_plot(self, title: str = '', epochs_to_plot: Union[int, dict, None] = None) -> None:


        if isinstance(epochs_to_plot, int):
            epochs_to_plot = [epochs_to_plot]
        elif isinstance(epochs_to_plot, dict):
            epochs_to_plot = list(epochs_to_plot.keys())

        # Plot the experimental and simulated forces
        plt.figure(figsize=(10, 6))

        first_epoch_idx = epochs_to_plot[0]
        displacement = self.loss_df_dict[first_epoch_idx]['displacement']
        force_target = self.loss_df_dict[first_epoch_idx]['force_target']
        plt.plot(displacement, force_target, label='Target Force', color='red')

        for epoch in epochs_to_plot:
            force_sim = self.loss_df_dict[epoch]['force_sim']
            plt.plot(displacement, force_sim,
                    label=f'Simulated force, epoch {epoch + 1}', color=self.color_picker(epoch))

        plt.xlabel('Displacement [mm]')
        plt.ylabel('Force [N]')
        plt.title(f'Force Comparison {title}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(self.out_dir, 'Force_Comparison.png'), bbox_inches='tight')
        plt.close()

        # Write data to csv
        columns = [displacement, force_target] + [self.loss_df_dict[epoch]['force_sim'] for epoch in epochs_to_plot]

        # Concatenate the columns into a DataFrame
        plot_df = pd.concat(columns, axis=1)

        # Optionally, set the column names if needed
        plot_df.columns = ['displacement', 'force_target'] + [f'force_sim_{epoch}' for epoch in epochs_to_plot]

        plot_df.to_csv(os.path.join(self.out_dir, 'force_comparison.csv'), index=False)


    def force_plot(self, title: str = '', epochs_to_plot: Union[int, List[int], dict, None] = None) -> None:
        """
        Plots the experimental and simulated forces for specified epochs and saves the plot and data to files.

        Args:
            title (str): The title of the plot.
            epochs_to_plot (Union[int, List[int], None]): Epoch(s) to plot. Can be a single epoch, a list of epochs, or None.
                                                        If None, plots the epochs specified in self.epochs_to_plot.

        Returns:
            None
        """

        if isinstance(epochs_to_plot, int):
            epochs_to_plot = [epochs_to_plot]
        elif isinstance(epochs_to_plot, dict):
            epochs_to_plot = list(epochs_to_plot.keys())
        elif epochs_to_plot is None:
            epochs_to_plot = self.epochs_to_plot


        # Plot the experimental and simulated forces
        plt.figure(figsize=(7, 7))

        first_epoch_idx = epochs_to_plot[0]
        displacement = self.loss_df_dict[first_epoch_idx]['displacement']
        force_target = self.loss_df_dict[first_epoch_idx]['force_target']
        
        # Plot the experimental force with x markers
        plt.plot(displacement, force_target, label='Experimental Force', color='red', linewidth=2)

        for epoch in epochs_to_plot:
            force_sim = self.loss_df_dict[epoch]['force_sim']
            if not self.validation:
                plt.plot(displacement, force_sim, label=f'Simulated Force, Epoch {epoch + 1}', 
                        color=self.color_picker(epoch), linestyle='--', linewidth=2)
            if self.validation:
                plt.plot(displacement, force_sim, label=f'Simulated Force, validation', 
                    color='blue', linestyle='--', linewidth=2)

        plt.xlabel('Displacement [mm]', fontsize=16)
        plt.ylabel('Force [N]', fontsize=16)
        plt.title(f'Target and Simulated Forces Over Displacement {title}', fontsize=16)
        if self.validation:
            plt.title(f'Validation: Target and S Force {title}', fontsize=16)
        plt.legend(loc='upper left', fontsize=16)
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.gca().yaxis.get_major_ticks()[0].label1.set_visible(False)  # Remove the 0 label on y-axis
        plt.tight_layout()
        savepath = os.path.join(self.out_dir, 'Force_Comparison.png')
        if self.validation:
            savepath = os.path.join(self.out_dir, f'Validation_{title}.png')
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()

        # Write data to CSV
        columns = [displacement, force_target] + [self.loss_df_dict[epoch]['force_sim'] for epoch in epochs_to_plot]
        plot_df = pd.concat(columns, axis=1)

        # Set the column names
        plot_df.columns = ['displacement', 'force_target'] + [f'force_sim_{epoch}' for epoch in epochs_to_plot]

        plot_df.to_csv(os.path.join(self.out_dir, 'force_comparison.csv'), index=False)


    def kNN_plt(self, title: str = '', epochs_to_plot: Union[int, List[int], dict, None] = None) -> None:
        """
        Plots the kNN model predictions for specified epochs and saves the plot and data to files.

        Args:
            title (str): The title of the plot.
            epochs_to_plot (Union[int, List[int], None]): Epoch(s) to plot. Can be a single epoch, a list of epochs, or None.
                                                        If None, plots the epochs specified in self.epochs_to_plot.

        Returns:
            None
        """

        if isinstance(epochs_to_plot, int):
            epochs_to_plot = [epochs_to_plot]
        elif isinstance(epochs_to_plot, dict):
            epochs_to_plot = list(epochs_to_plot.keys())
        elif epochs_to_plot is None:
            epochs_to_plot = self.epochs_to_plot

        # Initialize master_df with strain values as the index if strain is constant across epochs
        key = list(self.mat_tensor_dict.keys())[0]
        strain = list(self.mat_tensor_dict[key][:, 1].detach().numpy())  # Assuming the first tensor's strain values are representative
        master_df = pd.DataFrame(strain, columns=['strain'])

        # Generate plot of the kNN model prediction and save to df
        plt.figure(figsize=(7, 7))
        
        for epoch, mat_tensor in self.mat_tensor_dict.items():
            stress = list(mat_tensor[:, 0].detach().numpy())

            # Plot specified epochs
            if epoch in epochs_to_plot:
                plt.plot(strain, stress, label=f'kNN Epoch {epoch + 1}', color=self.color_picker(epoch))
                # Adding columns to the master_df
                new_column = pd.DataFrame(stress, columns=[f'kNN(strain)_{epoch + 1}'])
                master_df = pd.concat([master_df, new_column], axis=1)

        # Write data to CSV        
        master_df.to_csv(os.path.join(self.out_dir, 'kNN_master.csv'), index=False)

        # Plot styling
        plt.xlabel('Plastic Strain [-]', fontsize=16)
        plt.ylabel('Yield Stress [MPa]', fontsize=16)
        plt.title(f'kNN Model Prediction{title}', fontsize=16)
        plt.legend(loc='lower right', fontsize=16)
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.gca().yaxis.get_major_ticks()[0].label1.set_visible(False)  # Remove the 0 label on y-axis
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, f'kNN.png'), bbox_inches='tight')
        plt.close()



    def save_debug_frames(self, epochs_to_save: Union[int, dict, List[int], None] = None) -> None:
        """
        Saves the debug gradient data frames for specified epochs to CSV files.

        Args:
            epochs_to_plot (Union[int, List[int], None]): Epoch(s) to save. Can be a single epoch, a list of epochs
                                                       

        Returns:
            None
        """
        if isinstance(epochs_to_save, int):
            epochs_to_save = [epochs_to_save]
        elif isinstance(epochs_to_save, dict):
            epochs_to_save = list(epochs_to_save.keys())

        for epoch, df in self.backprop_element_df_dict.items():
            if epoch in epochs_to_save:
                # Save the debug data to a CSV file for the epoch
                df.to_csv(os.path.join(self.out_dir, f'debug_gradient_{epoch}.csv'), index=False)

    def epoch_loss_plt(self):
        # Generate plot of the loss along epoch
        plt.scatter(range(len(self.loss_epoch_list)), self.loss_epoch_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.savefig(os.path.join(self.out_dir, f'epoch_loss.png'))
        plt.close()

        #write data to csv
        # Assuming self.loss_epoch is a list of loss values
        data = [{'epoch': i + 1, 'loss': loss} for i, loss in enumerate(self.loss_epoch_list)]
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.out_dir, 'epoch_loss.csv'), index=False)

    def epoch_logloss_plt(self):
        # Generate plot of the loss along epoch
        plt.scatter(range(len(self.loss_epoch_list)), np.log(self.loss_epoch_list))
        plt.xlabel('Epoch')
        plt.ylabel('log (Loss)')
        plt.title('log (Loss) per Epoch')
        plt.savefig(os.path.join(self.out_dir, f'epoch_logloss.png'))
        plt.close()

        #write data to csv
        # Assuming self.loss_epoch is a list of loss values
        data = [{'epoch': i + 1, 'log(loss)': np.log(loss)} for i, loss in enumerate(self.loss_epoch_list)]
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.out_dir, 'epoch_log_loss.csv'), index=False)

    def write_report(self):
        with open(f'{self.out_dir}/{self.test_name}_report.txt', 'w') as report:
            # Write the report structure
            report.write(f'Test Run: {self.test_name}, Date: {datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}\n')
            report.write(f'Model:\n{self.model}\n\n')
            report.write(f'Run description: {self.test_description}\n\n')  # Assuming 'description' contains the model description and code

            report.write(f'Expname / Loss Data: {self.expname}\n\n')

            report.write('Training:\n')
            report.write(f'Optimizer: {self.optimizer}\n')
            report.write(f'Learn Rate: {self.learning_rate}\n')  
            report.write(f'Number of Epochs: {self.number_of_epochs}\n') 
            report.write(f'Train Loop Description: {self.train_loop_description}\n\n')


def cleaner(working_directory):
    expname_list = ['H_10', 'H_50', 'C_20']
    keep = ['exp_data', 'exp_data_original', 'exp_data_simulated', 'error_log'] + [f'{expname}_000.inp' for expname in expname_list]
    files = os.listdir(working_directory)

    # Delete unwanted files and directories
    for file in files:
        file_path = os.path.join(working_directory, file)
        if file not in keep:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    print('Files and directories cleaned up')

def two_stage_hardening_model(plastic_strain):
    k0 = 0.1 #MPa
    STRAIN_CRIT = 0.18

    def delta_k_func(plastic_strain):
        A = 1436 #MPa
        n = 1.606
        B= 6.30 #MPa
        Q = 1.0 #MPa
        BETA = (A * n * STRAIN_CRIT ** (n - 1) - B)/Q
        delta_k = None

        if plastic_strain <= STRAIN_CRIT:
            delta_k = A * plastic_strain ** n
        else:
            delta_k = A * STRAIN_CRIT ** n + B * (plastic_strain - STRAIN_CRIT) + Q * (1 - np.exp(-BETA * (plastic_strain - STRAIN_CRIT)))
        
        return delta_k
    
    # Define the hardening parameters
    k = k0 + delta_k_func(plastic_strain )

    return k

def clip_grad(grad: list, std_dev_multiplier: float = 2, delta_std= 0.5,  trim_percent: float = 0.1):
    
    if not grad:
        return grad, 0
    
    clipped_grad = []
    grad_list = grad.copy()
    
    # Sort the gradients
    sorted_grad = sorted(grad_list)
    
    # Calculate the number of values to trim from each end
    trim_count = int(len(sorted_grad) * trim_percent)
    
    # Ensure we don't trim more than we have
    if trim_count * 2 >= len(sorted_grad):
        trim_count = len(sorted_grad) // 2
    
    # Trim the gradients
    trimmed_grad = sorted_grad[trim_count:-trim_count]
    
    # Handle the case where trimmed_grad might be empty
    if not trimmed_grad:
        trimmed_grad = sorted_grad
    # Calculate mean and standard deviation of the trimmed gradients
    mean_trimmed_grad = np.mean(trimmed_grad)
    std_trimmed_grad = np.std(trimmed_grad)
    lower_bound = mean_trimmed_grad +  std_dev_multiplier * std_trimmed_grad
    upper_bound = mean_trimmed_grad + (std_dev_multiplier + delta_std) * std_trimmed_grad
    

    clip_count = 0
    clipped_grad = []
    min_grad = 100000
    max_grad = 0
    for gr_item in grad_list:
        if gr_item > lower_bound or gr_item < -lower_bound:
            clip_count += 1
            clipped_grad.append(gr_item)
            max_grad = max(max_grad, abs(gr_item))
            min_grad = min(min_grad, abs(gr_item))
        else:
            clipped_grad.append(0)
    def scale_grad(grad, min_grad, max_grad):
        epsilon = 1e-8
        sign = np.sign(grad)
        grad = (abs(grad) - min_grad) / ((max_grad - min_grad)+epsilon) * (upper_bound - lower_bound) + lower_bound
        grad = grad * sign
        return grad
    for i, grad in enumerate(clipped_grad):
        if grad != 0:
            grad = scale_grad(grad, min_grad, max_grad)
            grad_list[i] = grad        

    return grad_list, clip_count

def interval_inspection(grad_list, strain_list, grad_list_unclipped = None, interval=5):
    if not grad_list or not strain_list:
        return None
    
    # Inspect gradients and strains in intervals
    interval_size = len(grad_list) // interval
    grad_data = []
    grad_unclipped_data = []
    strain_data = []

    for i in range(interval):
        start_idx = i * interval_size
        end_idx = (i + 1) * interval_size if i < interval - 1 else len(grad_list)
        
        # Clipped Gradients
        interval_grads = grad_list[start_idx:end_idx]
        if interval_grads == []:
            continue
        grad_data.append({
            'Interval': f'{i + 1} ({start_idx}-{end_idx})',
            'Mean Grad': np.mean(interval_grads),
            'Median Grad': np.median(interval_grads),
            'Max Grad': np.max(interval_grads),
            'Min Grad': np.min(interval_grads),
            'Std Grad': np.std(interval_grads)
        })
        
        # Strains
        interval_strains = strain_list[start_idx:end_idx]
        if interval_strains == []:
            continue
        strain_data.append({
            'Interval': f'{i + 1} ({start_idx}-{end_idx})',
            'Mean Strain': np.mean(interval_strains),
            'Median Strain': np.median(interval_strains),
            'Max Strain': np.max(interval_strains),
            'Min Strain': np.min(interval_strains),
            'Std Strain': np.std(interval_strains)
        })

        # Unclipped Gradients
        interval_grads_unclipped = grad_list_unclipped[start_idx:end_idx]
        if interval_grads_unclipped == []:
            continue
        grad_unclipped_data.append({
            'Interval': f'{i + 1} ({start_idx}-{end_idx})',
            'Mean Grad Unclipped': np.mean(interval_grads_unclipped),
            'Median Grad Unclipped': np.median(interval_grads_unclipped),
            'Max Grad Unclipped': np.max(interval_grads_unclipped),
            'Min Grad Unclipped': np.min(interval_grads_unclipped),
            'Std Grad Unclipped': np.std(interval_grads_unclipped)
        })
        
    
    grad_df = pd.DataFrame(grad_data)
    grad_unclipped_df = pd.DataFrame(grad_unclipped_data)
    strain_df = pd.DataFrame(strain_data)
    
    print("Gradients:")
    print(grad_df)
    print("\nUnclipped Gradients:")
    print(grad_unclipped_df)
    print("\nStrains:")
    print(strain_df)

def histogram_plotter(list, num_bins, title, xlabel, ylabel, save_path):

    plt.hist(list, bins=num_bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()

def more_epoch_histograms(archive, df_column, num_bins, title, xlabel, ylabel, save_path):
    #archive = (r"D:\Bachelor_Thesis_Cedric_Grieder\Code\dont_change_working_code\template_for_cluster\archive_dev\20240801_17_0_SimpleModel_Adam_0.001_nepochs_100/debug")
    assert len(df_column.split(' ')) == 1, 'only one column allowed, no whitespaces'
    
    os.chdir(archive)
    for file in os.listdir(archive):
        print(f'file: {file}')
        if file.endswith('.csv'):
            print(f'reading {file}')
            df = pd.read_csv(file)
            strain_list = df[df_column].tolist()
            epoch = 1 + int((file.split('_')[-1]).split('.')[0])
            histogram_plotter(list =strain_list, num_bins= num_bins,
                            title=f'{df_column} histogram epoch: {epoch}', xlabel={df_column}, ylabel='frequency',
                            save_path=os.path.join(archive, f'{file}_{df_column}_hist.png'))
            print(f'histogram saved to {os.path.join(archive, f"{file}_strain_hist.png")}')

def evaluate_data_failed_job(file_path):
    """
    Reads data from result file in the specified format and creates a DataFrame
    containing all the information over all epochs.
    """
    with open(file_path, 'r') as file:
        data = file.read()

    # Regular expression to match the required sections
    pattern = re.compile(
        r"Epoch \[(\d+)/(\d+)\], Loss: ([\d.]+),\n"
        r"Mean Grad: ([\d.-]+), Median: ([\d.-]+)\n"
        r"clipped gradients: (\d+)\n"
        r"%of clipped gradients: ([\d.]+)\n"
        r"time: ([\d-]+ [\d:.]+)\n"
        r"time total epoch: ([\d:]+)\n"
        r"time for simulator: ([\d:]+)\n"
        r"time for backprop: ([\d:]+)\n"
        r"total training time: ([\d:]+)"
    )

    # List to store the extracted data
    data_list = []

    # Find all matches and store them in the list
    for match in pattern.finditer(data):
        epoch_data = {
            "Epoch": int(match.group(1)),
            "Total Epochs": int(match.group(2)),
            "Loss": float(match.group(3)),
            "Mean Grad": float(match.group(4)),
            "Median Grad": float(match.group(5)),
            "Clipped Gradients": int(match.group(6)),
            "% Clipped Gradients": float(match.group(7)),
            "Time": match.group(8),
            "Time Total Epoch": match.group(9),
            "Time for Simulator": match.group(10),
            "Time for Backprop": match.group(11),
            "Total Training Time": match.group(12)
        }
        data_list.append(epoch_data)

    # Create a DataFrame from the list
    df = pd.DataFrame(data_list)

    return df


import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

class PlotGenerator:
    def __init__(self, output_dir: str):
        """
        Initialize the PlotGenerator class with the output directory.

        Args:
            output_dir (str): Directory where the plots and processed data will be saved.
        """
        self.out_dir = output_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.fontsize = 20

    def force_plot(self, force_csv: str, filename:str = '' , title: str = '') -> None:
        """
        Plots the experimental and simulated forces and saves the plot and data to files.

        Args:
            force_csv (str): Path to the CSV file containing force comparison data.
            title (str): The title of the plot.

        Returns:
            None
        """
        # Load the data
        force_data = pd.read_csv(force_csv)

        # Extract data
        displacement = force_data['displacement']
        force_target = force_data['force_target']
        # Dynamically find the column that starts with 'force_sim'
        force_sim_column = [col for col in force_data.columns if col.startswith('force_sim')]
        if len(force_sim_column) == 0:
            raise KeyError("No column starting with 'force_sim' found.")
        force_sim = force_data[force_sim_column[0]]
        # Calculate the metrics
        mae = np.mean(np.abs(force_target - force_sim))
        epsilon = np.finfo(float).eps  # Small value to avoid division by 0
        mape = np.mean(np.abs((force_target - force_sim) / (epsilon + force_target)) * 100)
        mse = np.mean((force_target - force_sim) ** 2)

        # Plot the experimental and simulated forces
        plt.figure(figsize=(8, 8))
        plt.plot(displacement, force_target, label='Experimental Force', color='red', linewidth=2)
        plt.plot(displacement, force_sim, label='Simulated Force', linestyle='--', linewidth=2)

        fontsize = self.fontsize

        # Add the metrics to the plot in a text box
        metrics_text = f"MAE: {mae:.2f} N\nMAPE: {mape:.2f}%\nMSE: {mse:.2f} N^2"
        plt.text(0.95, 0.05, metrics_text, fontsize=fontsize, verticalalignment='bottom', horizontalalignment='right', 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        # Adjust the axis labels, title, and tick labels to have the same size
        plt.xlabel('Displacement [mm]', fontsize= fontsize)
        plt.ylabel('Force [N]', fontsize= fontsize)
        plt.title(f'{title}', fontsize= fontsize)
        plt.xticks(fontsize= fontsize)
        plt.yticks(fontsize= fontsize)

        plt.legend(loc='upper left', fontsize= fontsize)
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.gca().yaxis.get_major_ticks()[0].label1.set_visible(False)  # Remove the 0 label on y-axis
        plt.tight_layout()

        # Save the plot
        savepath = os.path.join(self.out_dir, f'{filename}Force_Comparison.png')
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()

    def kNN_plot(self, knn_csv: str, title: str = '') -> None:
        """
        Plots the kNN model predictions and saves the plot and data to files.

        Args:
            knn_csv (str): Path to the CSV file containing kNN prediction data.
            title (str): The title of the plot.

        Returns:
            None
        """
        # Load the kNN data
        knn_data = pd.read_csv(knn_csv)

        # Extract strain and stress data
        strain = knn_data['strain']
        knn_column = [col for col in knn_data.columns if col.startswith('kNN(strain)')]
        if len(knn_column) == 0:
            raise KeyError("No column starting with 'kNN(strain)' found.")
        stress = knn_data[knn_column[0]]
        # Plot the kNN model predictions
        plt.figure(figsize=(8, 8))
        plt.plot(strain, stress, label='kNN Prediction', linestyle='-', linewidth=2)

        fontsize = self.fontsize
        # Plot styling
        plt.xlabel('Plastic Strain [-]', fontsize=fontsize)
        plt.ylabel('Yield Stress [MPa]', fontsize=fontsize)
        plt.title(f'{title}', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc='lower right', fontsize=fontsize)
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.gca().yaxis.get_major_ticks()[0].label1.set_visible(False)  # Remove the 0 label on y-axis
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.out_dir, 'kNN.png'), bbox_inches='tight')
        plt.close()


