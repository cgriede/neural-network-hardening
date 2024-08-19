import os
import shutil
import torch.optim as optim
from utils import cleaner

# Define parameters
seeds = [37,]
models = [#'SimpleModel',
          'DeepModel',]
clipping_rates = [5]

# Define specific combinations
combinations = [
    ('tf.MSE()', 'tf.StandardFilter', 0.001, 600),
    ('tf.MSE()', 'tf.StandardFilter', 0.0005, 800),
        
    ('tf.MSE()', 'tf.HeavyRegularization', 0.001, 300),
    ('tf.MSE()', 'tf.HeavyRegularization', 0.0005, 400),

    ('optim.RMSprop', 'tf.MSE()', 'LightRegularization', 0.001, 300),

    ('tf.MSE()', 'tf.EnforceS33Direction', 0.001, 600),
    ('tf.MSE()', 'tf.EnforceS33Direction', 0.0005, 800),

]

video_combination = [
    ('optim.RMSprop', 'tf.MSE()', 'tf.HeavyRegularization', 0.001, 300),

]

expected_number_tests = len(seeds)*len(models) * len(combinations) * len(clipping_rates)
print(f"Expected number of tests: {expected_number_tests}")

source_dir = os.getcwd()

# Base directory for tests
base_dir = 'ML_Video'
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)

#include bash script to submit all jobs in base directory
shutil.copy2('submit_all_jobs.sh', base_dir)

#include extraction script
shutil.copy2('extract_results.sh', base_dir)

exclude_files = [base_dir, 'script_generator.py', 'scratch.py', '.gitignore', 'submit_all_jobs.sh',
                 '.git', 'ML_Tests', 'main.py', 'submission_file.sh', 'validation','validator.py', 'archive']

# Function to create new directories and scripts
def create_test_dir(n_epochs, seed, model, optimizer, optimizer_type, clipping_rate, learning_rate, loss_inst, feature_selector):
    cr_str = 'NoClip' if clipping_rate is None else f'CR{clipping_rate}'
    loss_name = loss_inst.split('(')[0].split('.')[1]
    feature_name = feature_selector.split('(')[0].split('.')[1]
    learn_rate_name = str(learning_rate).split('.')[1]
    #set the directory name
    dir_name = f"{model}_{optimizer_type}_n{n_epochs}_{cr_str}_LR{learn_rate_name}_{loss_name}_{feature_name}"
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)

    # Read the content of the original main.py
    with open(os.path.join(source_dir, 'main.py'), 'r') as file:
        main_content = file.readlines()

    # Read the content of the original submission_file.sh
    with open(os.path.join(source_dir, 'submission_file.sh'), 'r') as file:
        submission_file_content = file.readlines()

    updated_main_content = []
    for line in main_content:
        #use the generator flag to switch between the two training loops
        if 'generator = False' in line:
            line = line.replace('generator = False', 'generator = True')
        if 'EPOCH_PLACEHOLDER' in line:
            line = line.replace('EPOCH_PLACEHOLDER', str(n_epochs))
        if 'SEED_PLACEHOLDER' in line:
            line = line.replace('SEED_PLACEHOLDER', str(seed))
        if 'MODEL_PLACEHOLDER' in line:
            line = line.replace('MODEL_PLACEHOLDER', model)
        if 'LR_PLACEHOLDER' in line:
            line = line.replace('LR_PLACEHOLDER', str(learning_rate))
        if 'OPTIMIZER_PLACEHOLDER' in line:
            line = line.replace('OPTIMIZER_PLACEHOLDER', f'{optimizer}(model.parameters(), lr={learning_rate})')
        if 'CR_PLACEHOLDER' in line:
            line = line.replace('CR_PLACEHOLDER', str(clipping_rate))
        if 'LOSS_INST_PLACEHOLDER' in line:
            line = line.replace('LOSS_INST_PLACEHOLDER', loss_inst)
        if 'FEATURE_SELECTOR_PLACEHOLDER' in line:
            line = line.replace('FEATURE_SELECTOR_PLACEHOLDER', feature_selector)
        updated_main_content.append(line)

    # Write the new main.py script
    script_path = os.path.join(dir_path, 'main.py')
    with open(script_path, 'w') as script_file:
        script_file.writelines(updated_main_content)

    updated_submission_file_content = []
    # Replace job name placeholder in the submission_file.sh content
    for line in submission_file_content:
        if 'JOB_NAME_PLACEHOLDER' in line:
            print(f"Replacing JOB_NAME_PLACEHOLDER with {dir_name}")
            line = line.replace('JOB_NAME_PLACEHOLDER', dir_name)
        updated_submission_file_content.append(line)

    # Write the new submission_file.sh script
    submission_path = os.path.join(dir_path, 'submission_file.sh')
    with open(submission_path, 'w') as submission_file:
        submission_file.writelines(updated_submission_file_content)

    #clean the working directory before copying the files
    cleaner(working_directory='wd_dev')

    # Copy other necessary files
    for item in os.listdir(source_dir):
        src_path = os.path.join(source_dir, item)
        dst_path = os.path.join(dir_path, item)
        if item not in exclude_files:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

# Generate directories and scripts for each configuration
config = video_combination
for seed in seeds:
    for model in models:
        for optimizer, loss_inst, feature_selector, learning_rate, n_epochs in config:
            optimizer_type = optimizer.split('.')[1]
            for clipping_rate in clipping_rates:
                    create_test_dir(n_epochs, seed, model, optimizer, optimizer_type, clipping_rate, learning_rate, loss_inst, feature_selector)

print("All tests generated successfully!")