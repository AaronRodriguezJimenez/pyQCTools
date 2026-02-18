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