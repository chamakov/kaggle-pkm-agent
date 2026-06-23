import subprocess
import json
import base64

# Try to extract the kaggle_environments cabt data
subprocess.run("cp -r /usr/local/lib/python3.10/site-packages/kaggle_environments/envs/cabt /app/cabt_extracted", shell=True)
