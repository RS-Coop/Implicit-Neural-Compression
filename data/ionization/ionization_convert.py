import numpy as np
import pathlib
from natsort import natsorted


path = pathlib.Path("data/ionization/")

for i, file in enumerate(natsorted(path.glob("*.txt"))):
    #convert to numpy
    arr = np.loadtxt(file)

    #extract channels
    arr = arr[:,[1]] #just temperature

    #save
    np.save(path/f"features_{i}.npy", arr)