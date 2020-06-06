from torch import nn
import func

model = nn.Sequential()
model.add_module('fc1', nn.Linear(1025, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 1025))

print(model)

print(func.wavesp("."))
