import torch
import torch.nn as nn




"""
here the neural net model layers and activation functions are defined


"""
#works ok
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
    
    @property
    def name(self):
        return 'SimpleModel'

#to be tested
class DropoutModel(nn.Module):
    def __init__(self):
        super(DropoutModel, self).__init__()
        self.fc1 = nn.Linear(1, 8, dtype=torch.float64)
        self.fc2 = nn.Linear(8, 16, dtype=torch.float64)
        self.fc3 = nn.Linear(16, 8, dtype=torch.float64)
        self.fc4 = nn.Linear(8, 1, dtype=torch.float64)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.5)

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.softplus(x)
        return x
    
    @property
    def name(self):
        return 'DropoutModel'
    
#to be tested
class ReLUModel(nn.Module):
    def __init__(self):
        super(ReLUModel, self).__init__()
        self.fc1 = nn.Linear(1, 8, dtype=torch.float64)
        self.fc2 = nn.Linear(8, 16, dtype=torch.float64)
        self.fc3 = nn.Linear(16, 8, dtype=torch.float64)
        self.fc4 = nn.Linear(8, 1, dtype=torch.float64)
        self.relu = nn.ReLU()

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        return x
    
    @property
    def name(self):
        return 'ReLUModel'
    
    
#not useful for this case
class BatchNormModel(nn.Module):
    def __init__(self):
        super(BatchNormModel, self).__init__()
        self.fc1 = nn.Linear(1, 8, dtype=torch.float64)
        self.bn1 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.fc2 = nn.Linear(8, 16, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.fc3 = nn.Linear(16, 8, dtype=torch.float64)
        self.bn3 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.fc4 = nn.Linear(8, 1, dtype=torch.float64)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.softplus(x)
        return x

    @property
    def name(self):
        return 'BatchNormModel'

#works good
class DeepModel(nn.Module):
    def __init__(self):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(1, 8, dtype=torch.float64)
        self.fc2 = nn.Linear(8, 16, dtype=torch.float64)
        self.fc3 = nn.Linear(16, 32, dtype=torch.float64)  # New layer
        self.fc4 = nn.Linear(32, 16, dtype=torch.float64)  # New layer
        self.fc5 = nn.Linear(16, 8, dtype=torch.float64)
        self.fc6 = nn.Linear(8, 1, dtype=torch.float64)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

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
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        x = self.softplus(x)
        return x
        
    @property
    def name(self):
        return 'DeepModel'

class DeepSquareModel(nn.Module):
    def __init__(self):
        super(DeepSquareModel, self).__init__()
        self.fc1 = nn.Linear(1 , 18, dtype=torch.float64)
        self.fc2 = nn.Linear(18, 18, dtype=torch.float64)
        self.fc3 = nn.Linear(18, 18, dtype=torch.float64)  # New layer
        self.fc4 = nn.Linear(18, 18, dtype=torch.float64)  # New layer
        self.fc5 = nn.Linear(18, 18, dtype=torch.float64)
        self.fc6 = nn.Linear(18, 1 , dtype=torch.float64)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

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
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        x = self.softplus(x)
        return x
        
    @property
    def name(self):
        return 'DeepSquareModel'

#works bad
class ShallowModel(nn.Module):
    def __init__(self):
        super(ShallowModel, self).__init__()
        self.fc1 = nn.Linear(1, 8, dtype=torch.float64)
        self.fc2 = nn.Linear(8, 1, dtype=torch.float64)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.softplus(x)
        return x
    
    @property
    def name(self):
        return 'ShallowModel'