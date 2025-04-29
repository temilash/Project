#!/bin/bash

# List of instruments
instruments=(Bassoon Cello Clarinet Flute Oboe Sax Viola Violin)

# Base directory where your .pth files are
BASE_DIR="/mnt/scratch/sc22ol/Project/recipes/cad2/task2/ConvTasNet/crm20"

# Loop through each instrument and convert the model
for inst in "${instruments[@]}"; do
  echo "?? Converting model for $inst..."

  python -c "
import torch
from safetensors.torch import save_file

pth_path = '${BASE_DIR}/${inst}/best_model.pth'
safetensor_path = pth_path.replace('.pth', '.safetensors')

state_dict = torch.load(pth_path)
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']

# Remove 'model.' prefix if present
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

save_file(state_dict, safetensor_path)
print(f'? Saved: {safetensor_path}')
"
done

echo "?? All models converted to .safetensors!"
