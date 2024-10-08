import numpy as np
import pathlib
from natsort import natsorted

p = 48

root = pathlib.Path("data/ionization/")

#Points
points = np.load(root/"points.npy").reshape(p, -1, 3)

for i in range(p):
    np.save(root/f"points_{i}.npy", points[i,...])

#Feature directories
for i in range(p):
    (root/f"features_{i}/").mkdir(exist_ok=True)

#Features
for file in natsorted((root/"features").glob("*.txt.*")):
    # data = np.loadtxt(file)[:,[1]]
    data = np.loadtxt(file)
    data = data.reshape(p,-1,1)

    for i in range(p):
        np.savetxt(root/f"features_{i}"/file.name, data[i,...])