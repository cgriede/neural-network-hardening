import os
import shutil
import torch.optim as optim

# Define parameters
models = ['SimpleModel', 'DeepModel', 'BatchNormModel']
optimizers = ['optim.Adam', 'optim.RMSprop']
clipping_rates = [None, 5]
#learning_rates = [0.001, 0.0005]

# Define specific combinations
combinations = [
    #('tf.MSE()', 'tf.StandardFilter', 0.001, None),
    #('tf.MSE()', 'tf.StandardFilter', 0.0005, None),
    ('tf.MSE()', 'tf.DynamicFilter001', 0.001, None),
    ('tf.MSE()', 'tf.DynamicFilter0005', 0.0005, None),
    #('tf.WSE(5, 0.5)', 'tf.StandardFilter', 0.001, None),
    #('tf.WSE(5, 0.5)', 'tf.StandardFilter', 0.0005, None)
]

expected_number_tests = len(models) * len(optimizers) * len(combinations) * len(clipping_rates)
print(f"Expected number of tests: {expected_number_tests}")

source_dir = os.getcwd()

# Base directory for tests
base_dir = 'ML_Tests_DynamicFilter'
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(base_dir, exist_ok=True)

exclude_files = [base_dir, 'script_generator.py', 'scratch.py', '.gitignore', 'submit_all_jobs.sh',
                 '.git', 'ML_Tests', 'main.py', 'submission_file.sh', 'validation.py', 'archive_dev']

# Function to create new directories and scripts
def create_test_dir(model, optimizer, optimizer_type, clipping_rate, learning_rate, loss_inst, feature_selector):
    cr_str = 'NoClip' if clipping_rate is None else f'CR{clipping_rate}'
    loss_name = loss_inst.split('(')[0].split('.')[1]
    feature_name = feature_selector.split('(')[0].split('.')[1]
    learn_rate_name = str(learning_rate).split('.')[1]
    dir_name = f"{model}_{optimizer_type}_{cr_str}_LR{learn_rate_name}_{loss_name}_{feature_name}"
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
for model in models:
    for index, optimizer in enumerate(optimizers):
        optimizer_type = optimizer.split('.')[1]
        for loss_inst, feature_selector, learning_rate, _ in combinations:
            for clipping_rate in clipping_rates:
                create_test_dir(model, optimizer, optimizer_type, clipping_rate, learning_rate, loss_inst, feature_selector)

print("All tests generated successfully!")