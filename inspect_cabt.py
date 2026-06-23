import sys
import os
import importlib
import kaggle_environments

print(kaggle_environments.__file__)
cabt_dir = os.path.join(os.path.dirname(kaggle_environments.__file__), 'envs', 'cabt')
if os.path.exists(cabt_dir):
    print("Found cabt directory:", cabt_dir)
    print(os.listdir(cabt_dir))
