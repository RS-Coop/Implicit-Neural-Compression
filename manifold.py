import numpy as np
import skdim

device = torch.device("gpu")

def estimate_dim(samples, scale=1e-5):

    lpca = skdim.id.lPCA().fit_pw(samples, n_neighbors=100, n_jobs=16)

    return np.mean(lpca.dimension_pw_)

data_path = "/gscratch/amath/cooper/data/manifold/"

samples = np.load(data_path+"neuron_transport_partial.npy")

print(estimate_dim(samples))