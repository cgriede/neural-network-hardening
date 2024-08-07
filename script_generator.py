import os
import shutil

# Define parameters
models = ['SimpleModel', 'BatchNormModel', 'ReLUModel', 'DropoutModel', 'DeepModel', 'ShallowModel']
optimizers = ['optim.Adam', 'optim.RMSprop', 'optim.Rprop']
clipping_rates = [None, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
learning_rates = [0.001, 0.01, 0.1]
source_dir = os.getcwd()
exclude_files = ['script_generator.py', 'scratch.py', '.gitignore']

# Base directory for tests
base_dir = 'ML_Tests'
os.makedirs(base_dir, exist_ok=True)

# Read the content of the original model_dev.py
with open(os.path.join(source_dir, 'model_dev.py'), 'r') as file:
    model_dev_content = file.read()

# Function to create new directories and scripts
def create_test_dir(model, optimizer, clipping_rate, learning_rate):
    cr_str = 'NoClip' if clipping_rate is None else f'CR{clipping_rate}'
    dir_name = f"{model}_{optimizer.split('.')[-1]}_{cr_str}_LR{learning_rate}"
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)

    # Modify the model_dev.py script content
    script_content = model_dev_content
    script_content += f"""
# Optimizer and training setup
optimizer = {optimizer}(model.parameters(), lr={learning_rate})
clipping_rate = {clipping_rate}

# Dummy data for illustration
strain_scaled_shuffled = torch.randn(32, 1, dtype=torch.float64)
grad_shuffled = torch.randn(32, 1, dtype=torch.float64)

for epoch in range(2000):  # Number of epochs set to 2000
    optimizer.zero_grad()
    stress_scaled = model(strain_scaled_shuffled)
    stress_scaled.backward(grad_shuffled)

    # Apply gradient clipping
    if clipping_rate is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_rate)

    optimizer.step()
"""

    # Write the new model_dev.py script
    script_path = os.path.join(dir_path, 'model_dev.py')
    with open(script_path, 'w') as script_file:
        script_file.write(script_content)

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
    for optimizer in optimizers:
        for clipping_rate in clipping_rates:
            for learning_rate in learning_rates:
                create_test_dir(model, optimizer, clipping_rate, learning_rate)
