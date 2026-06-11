import numpy as np

data = np.load("1.5_tensors.npz")

h0 = data['hc']
h1 = data['h1e']
h2 = data['h2e']

print("Tensors at 1.5")
print("- - - H0 - - - ")
print(h0)
print(h0.shape)
print("- - - H1 - - - ")
print(h1)
print(h1.shape)
print("- - - H2 - - - ")
print(h2)
print(h2.shape)

print(data.keys())
print(data.get("hc"))

# Read H0 values
import glob
import os

# Define directory
data_directory = '/Users/admin/PycharmProjects/pyQCTools/tests/N2_tensors_ccpvdz'

# 1. Get the list of files
file_list = glob.glob(os.path.join(data_directory, '*.npz'))

# 2. Sort the file list numerically
# This splits the filename and converts the first part to a float for sorting
file_list.sort(key=lambda x: float(os.path.basename(x).split('_')[0]))

loaded_H0_data = []

# 3. Process the sorted list
for file_path in file_list:
    try:
        data = np.load(file_path)
        # Accessing the specific key 'hc'
        loaded_H0_data.append(data['hc'])
        print(f"Successfully loaded: {os.path.basename(file_path)}")

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")

print("\n* * * H0 values (Sorted) * * *")
for hc in loaded_H0_data:
    print(hc)

