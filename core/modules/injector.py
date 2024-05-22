import torch
import torch.nn as nn

'''
Based on Continual Backprop from the following sources:
    - https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/algos/cbp.py
    - https://arxiv.org/abs/2108.06325v3
'''

class Injector(nn.Module):
    def __init__(
            self,
            model,
            replacement_rate=0.1,
            decay_rate=0.9
        ):
        super().__init__()
        
        #hyperparameters
        self.model = model

        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate

        self.num_hidden_layers = self.model.num_hidden_layers
        self.layers = [layer for layer in self.model.get_layers()]

        #utilities
        self.weight_util = []
        self.bias_util = []
        self.mean_activations = []

        for layer in self.layers:
            self.weight_util.append(torch.zeros(layer.out_features, requires_grad=False))
            self.bias_util.append(torch.zeros(layer.out_features, requires_grad=False))
            self.mean_activations.append(torch.zeros(layer.out_features, requires_grad=False))

        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]

        return
    
    def update_utility(self, layer_idx, activations):
        self.weight_util[layer_idx] *= self.decay_rate

        #bias correction
        self.mean_activations[layer_idx] *= self.decay_rate
        self.mean_activations[layer_idx] -= - (1 - self.decay_rate) * activations.mean(dim=0)

        current_layer = self.layers[layer_idx]
        next_layer = self.layers[layer_idx + 1]
        output_wight_mag = next_layer.weight.data.abs().mean(dim=0)
        input_wight_mag = current_layer.weight.data.abs().mean(dim=1)

        new_util = output_wight_mag * activations.abs().mean(dim=0)
        
        self.weight_util[layer_idx] += (1 - self.decay_rate) * new_util

        #bias correction
        self.bias_util[layer_idx] = self.weight_util[layer_idx]

        return

    def detect(self, activations):
        features_to_replace = [torch.empty(0, dtype=torch.long) for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        
        for i in range(self.num_hidden_layers):
            #update feature utility
            self.update_utility(i, activations[i])

            #calculate number of features to replace
            num_new_features_to_replace = self.replacement_rate*self.layers[i].out_features
            self.accumulated_num_features_to_replace[i] += num_new_features_to_replace

            #case when the number of features to be replaced is between 0 and 1.
            if num_new_features_to_replace < 1:
                if torch.rand(1) <= num_new_features_to_replace:
                    num_new_features_to_replace = 1
            num_new_features_to_replace = int(num_new_features_to_replace)
    
            if num_new_features_to_replace == 0:
                continue

            #find features to replace in the current layer
            new_features_to_replace = torch.topk(-self.bias_util[i], num_new_features_to_replace)[1]

            #initialize utility for new features
            self.weight_util[i][new_features_to_replace] = 0
            self.mean_activations[i][new_features_to_replace] = 0.

            features_to_replace[i] = new_features_to_replace
            num_features_to_replace[i] = num_new_features_to_replace

        return features_to_replace, num_features_to_replace

    def update_params(self, params, num):
        for i in range(self.num_hidden_layers):
            if num[i] == 0:
                continue

            current_layer = self.layers[i]
            next_layer = self.layers[i+1]

            current_layer.init_weights(current_layer.weight.data[params[i], :])
            current_layer.bias.data[params[i]] *= 0

            # Update bias to correct for the removed features and set the outgoing weights and ages to zero
            next_layer.bias.data += (next_layer.weight.data[:,params[i]]*self.mean_activations[i][params[i]]).sum(dim=1)
            next_layer.weight.data[:, params[i]] = 0

        return

    @torch.no_grad()
    def forward(self, input):
        # do a forward pass and get the hidden activations
        activations = self.model.activations(input)

        #detect parameters to update
        params, num = self.detect(activations)
        
        #update parameters
        self.update_params(params, num)

        return