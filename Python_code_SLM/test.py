from init import *

# Load trained MPLC system from file
with open(file_name, "rb") as f:
    loaded_sys = pickle.load(f)
print('uploaded MPLC succesfully')

# Visualize mode sorting (HG01 input), save animation as video
snapshots = loaded_sys.sort(HG01, True, 50)
animate(snapshots, supermode.X, supermode.Y, save_as=f'{file_name}.mp4')

# Compute transfer matrix and extract insertion loss (IL) and mode-dependent loss (MDL)
loaded_sys.compute_transfer_matrix(inputs[0:4], targets[0:4])
il, mdl = loaded_sys.compute_IL_MDL_from_T()
print(f"IL = {il} dB; MDL = {mdl} dB")

# Show crosstalk matrix for visualization
loaded_sys.visualize_crosstalk_matrix()


