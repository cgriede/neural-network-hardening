import os
import shutil
import torch.optim as optim
# Define parameters
models = ['SimpleModel', 'DeepModel', 'ShallowModel']
optimizers = ['optim.Adam', 'optim.RMSprop']
optimizer_types = ['Adam', 'RMSprop',]
clipping_rates = [None, 2, 5,]
learning_rates = [0.005, 0.001, 0.0005,]

expected_number_of_tests = len(models) * len(optimizers) * len(clipping_rates) * len(learning_rates)
print(f"Expected number of tests: {expected_number_of_tests}")


source_dir = os.getcwd()
exclude_files = ['script_generator.py', 'scratch.py', '.gitignore', 'submit_all_jobs.sh',
                '.git', 'ML_Tests', 'model_dev.py', 'submission_file.sh']


# Base directory for tests
base_dir = 'ML_Tests'
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)


# Function to create new directories and scripts
def create_test_dir(model, optimizer, optimizer_type, clipping_rate, learning_rate):
    cr_str = 'NoClip' if clipping_rate is None else f'CR{clipping_rate}'
    dir_name = f"{model}_{optimizer_type}_CR{cr_str}_LR{learning_rate}"
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)

    # Read the content of the original model_dev.py
    with open(os.path.join(source_dir, 'model_dev.py'), 'r') as file:
        model_dev_content = file.readlines()

    # Read the content of the original submission_file.sh
    with open(os.path.join(source_dir, 'submission_file.sh'), 'r') as file:
        submission_file_content = file.readlines()

    updated_model_dev_content = []
    for line in model_dev_content:
        if 'MODEL_PLACEHOLDER' in line:
            line = line.replace('MODEL_PLACEHOLDER', model)
        if 'MODEL_TYPE_PLACEHOLDER' in line:
            line = line.replace('MODEL_TYPE_PLACEHOLDER', model)
        if 'LR_PLACEHOLDER' in line:
            line = line.replace('LR_PLACEHOLDER', str(learning_rate))
        if 'OPTIMIZER_PLACEHOLDER' in line:
            line = line.replace('OPTIMIZER_PLACEHOLDER', f'{optimizer}(model.parameters(), lr={learning_rate})')
        if 'OPTIM_TYPE_PLACEHOLDER' in line:
            line = line.replace('OPTIM_TYPE_PLACEHOLDER', optimizer_type)
        if 'CR_PLACEHOLDER' in line:
            line = line.replace('CR_PLACEHOLDER', str(clipping_rate))
        updated_model_dev_content.append(line)

    # Write the new model_dev.py script
    script_path = os.path.join(dir_path, 'model_dev.py')
    with open(script_path, 'w') as script_file:
        script_file.writelines(updated_model_dev_content)

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

    # Copy other necessary files
    for item in os.listdir(source_dir):
        src_path = os.path.join(source_dir, item)
        dst_path = os.path.join(dir_path, item)
        if item not in exclude_files:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

max_combinations = 100
combination_count = 0

# Generate directories and scripts for each configuration
for model in models:
    for index, optimizer in enumerate(optimizers):
        optimizer_type = optimizer_types[index]
        for clipping_rate in clipping_rates:
            for learning_rate in learning_rates:
                create_test_dir(model, optimizer, optimizer_type, clipping_rate, learning_rate)
                combination_count += 1
                if combination_count >= max_combinations:
                    print(f"Reached the maximum number of combinations: {max_combinations}")
                    break
            if combination_count >= max_combinations:
                break
        if combination_count >= max_combinations:
            break
    if combination_count >= max_combinations:
        break
