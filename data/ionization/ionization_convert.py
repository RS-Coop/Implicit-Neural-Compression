import numpy as np
import pathlib
from natsort import natsorted

path = pathlib.Path("data/ionization/features")
files = natsorted(path.glob("*.txt.*"))

part_paths = [pathlib.Path(f"data/ionization/features_{i}") for i in range(4)]

for file in files:
    data = np.loadtxt(file)[:,[1]]
    data = data.reshape(4,-1,1)

    for i in range(4):
        np.savetxt(part_paths[i]/file.name, data[i,...])
