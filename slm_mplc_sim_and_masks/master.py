# run_batch.py
import numpy as np
import subprocess
D = np.linspace(6E-2, 7E-2, 200)

for d in D:  # or some other param logic
    subprocess.run(['python', 'init.py', str(param)])
