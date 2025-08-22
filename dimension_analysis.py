import numpy as np
import torch
from core.model import Model
import skdim
import yaml
import copy

device = torch.device("gpu")

def estimate_dim(path, coords, d, c, N=1000, scale=1e-5):

    log_dir = "lightning_logs/" + path
    config_path = log_dir + "config.yaml"
    ckpt_path = log_dir + "/checkpoints/last.ckpt"
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    #Load model
    model = Model.load_from_checkpoint(ckpt_path, input_shape=((1,1),(1,1,1,d)), output_shape=(1,1,c), **config["model"], map_location=device)
    state = copy.deepcopy(model.state_dict())

    N = 1000
    samples = np.zeros((N, coords[1].shape[2]*c))

    scale = 1e-5

    for i in range(N):
        #Perturb
        for param in model.parameters():
            if param.requires_grad:
                with torch.no_grad():
                    param.data.add_(scale*torch.randn(param.shape, device=device))
    
        #Evaluate
        with torch.no_grad():
            output = model(coords)
    
        #Append sample
        samples[i,:] = output.detach().cpu().numpy().flatten()
    
        #Reset state
        model.load_state_dict(state)

    lpca = skdim.id.lPCA().fit_pw(samples[:,:], n_neighbors=100, n_jobs=16)

    return np.mean(lpca.dimension_pw_)

#NOTE: Setting the seed for reproducibility
torch.manual_seed(7175775717)

data_path = "/gscratch/amath/cooper/data/"

#Ignition
points = torch.from_numpy(np.load(data_path+"ignition/points.npy").astype(np.float32)).expand(1, 1, -1, -1)
t = torch.tensor([[0.]])

coords = (t, points)

print("Ignition post first snapshot: ", estimate_dim("ignition/hnet_online_manifold/version_0/", coords, 2, 4))
print("Ignition post training: ", estimate_dim("ignition/hnet_online_fjlt/version_3/", coords, 2, 4))

#Channel flow
points = torch.from_numpy(np.load(data_path+"channel_flow/points.npy").astype(np.float32)).expand(1, 1, -1, -1)
t = torch.tensor([[0.]])

coords = (t, points)

print("Channel flow post first snapshot: ", estimate_dim("channel_flow/hnet_online_manifold/version_0/", coords, 3, 3))
print("Channel flow post training: ", estimate_dim("channel_flow/hnet_online_fjlt/version_4/", coords, 3, 3))

#Neuron transport
points = torch.from_numpy(np.load(data_path+"neuron_transport/points.npy").astype(np.float32)).expand(1, 1, -1, -1)
t = torch.tensor([[0.]])

coords = (t, points)

print("Neuron transport post first snapshot: ", estimate_dim("neuron_transport/hnet_online_manifold/version_0/", coords, 3, 1))
print("Neuron transport post training: ", estimate_dim("neuron_transport/hnet_online_fjlt/version_1/", coords, 3, 1))

