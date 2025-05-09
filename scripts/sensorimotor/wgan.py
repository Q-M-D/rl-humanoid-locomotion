"""
Wasserstein discriminator for imitation learning
"""

from torch import nn
from torch.nn import functional as F
import torch
import json
import numpy as np


class DiscriminatorNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim1,
        hidden_dim2,
        hidden_dim3
    ):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim1)
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.hidden3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.output = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        x = F.elu(self.hidden1(x))
        x = F.elu(self.hidden2(x))
        x = F.elu(self.hidden3(x))
        x = self.output(x)
        return x

class ExpertDataLoader:
    def __init__(self, expert_data_info, his_len):
        """
        Initialize the expert data loader with data from files.
        
        Parameters
        ----------
        expert_data_info : list of dict
            List of dictionaries containing data paths and command velocities.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.his_len = his_len
        self._load_data(expert_data_info)
    
    def _load_data(self, expert_data_info):
        """
        Load data from files and convert to tensors.
        
        Parameters
        ----------
        expert_data_info : list of dict
            List of dictionaries containing data paths and command velocities.
        """
        data_arrays = {
            'v_command': [], 'qpos': [], 'qvel': [], 
            'fb_orientation': [], 'fb_angular_velocity': [], 
            'control_input': [], 'contact_mask': []
        }
        
        # Collect all data
        for info in expert_data_info:
            with open(info['path'], 'r') as f:
                data = json.load(f)
            
            # Add command velocity as a constant array
            data_arrays['v_command'] += [[info['v_command']] for _ in range(len(data['qpos']))]
            
            # Add all other data
            for key in ['qpos', 'qvel', 'fb_orientation', 'fb_angular_velocity', 
                       'control_input', 'contact_mask']:
                data_arrays[key] += data[key] if key in data else data[key.replace('_', '_')]
        
        # Convert to tensors and store as attributes
        self.q = torch.tensor(data_arrays['qpos'], dtype=torch.float32, device=self.device)
        self.qvel = torch.tensor(data_arrays['qvel'], dtype=torch.float32, device=self.device)
        self.fb_quat = torch.tensor(data_arrays['fb_orientation'], dtype=torch.float32, device=self.device)
        self.fb_angle_vel = torch.tensor(data_arrays['fb_angular_velocity'], dtype=torch.float32, device=self.device)
        self.action = torch.tensor(data_arrays['control_input'], dtype=torch.float32, device=self.device)
        self.contact_mask = self.get_contact_mask(data_arrays)
        self.v_command = torch.tensor(data_arrays['v_command'], dtype=torch.float32, device=self.device)
        
    def get_contact_mask(self, data_arrays):
        contact_mask = torch.tensor(data_arrays['contact_mask'], dtype=torch.float32, device=self.device)
        for i in range(contact_mask.shape[0]):
            for j in range(contact_mask.shape[1]):
                if contact_mask[i, j] > 0:
                    contact_mask[i, j] = 1.0
                else:
                    contact_mask[i, j] = 0.0
        return contact_mask
    
    def get_expert_data(self, num_samples):
        """
        Get num_samples of expert data, where each sample consists of 5 consecutive states.

        Parameters
        ----------
        num_samples : int
            Number of samples to get.

        Returns
        ----------
        states : torch.tensor((num_samples, state_dim * 5), dtype=float)
            States visited by expert policy, concatenated from 5 consecutive states.
        actions : torch.tensor((num_samples, action_dim), dtype=float)
            Corresponding actions to be taken by expert.
        """
        # Ensure we have enough samples for 5 consecutive states
        max_start_idx = len(self.q) - self.his_len
        indices = np.random.choice(max_start_idx, size=num_samples)
        
        # Initialize lists to store concatenated states
        concatenated_states_actions = []
        
        for idx in indices:
            # Get 5 consecutive states
            state_action_sequence = torch.cat((
                self.v_command[idx:idx+self.his_len],    # 5 x 1
                self.fb_quat[idx:idx+self.his_len],      # 5 x 4
                self.fb_angle_vel[idx:idx+self.his_len], # 5 x 3
                self.q[idx:idx+self.his_len],            # 5 x 12
                self.qvel[idx:idx+self.his_len],         # 5 x 12
                # self.contact_mask[idx:idx+self.his_len], # 5 x 24
                # self.action[idx:idx+self.his_len]        # 5 x 12
            ), dim=-1)
            
            # Flatten the sequence into a single state vector
            concatenated_states_actions.append(state_action_sequence.flatten())
        
        # Convert lists to tensors
        states_actions = torch.stack(concatenated_states_actions)
        return states_actions


def get_discriminator_loss(
    discriminator_net,
    state_action_exp,
    state_action_pi,
    lambda_=10,
    eta=0.5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Collect samples with policy pi.
    To speed up training, this function runs as batch.

    Parameters
    ----------
    discriminator_net : torch.nn.Module
        Discriminator network to be updated (critic in WAIL)
    state_action_exp: torch.tensor((num_samples), dtype=int)
        States visited by expert datasets
    state_action_pi : torch.tensor((num_samples), dtype=int)
        States visited by policy pi.
    lambda_ : float
        Hyperparameter for the gradient penalty.
    eta : float
        Hyperparameter for the discriminator output.
    device : str
        Device to run the computation on. Default is "cuda" if available, otherwise "cpu".

    Returns
    ----------
    Mean of discriminator loss using Wasserstein distance
    """
    d_pi = discriminator_net(state_action_pi)
    d_exp = discriminator_net(state_action_exp)

    # Ref: http://arxiv.org/abs/2309.14225
    tanh_d_pi = torch.tanh(eta * d_pi)
    tanh_d_exp = torch.tanh(eta * d_exp)

    # Wasserstein loss： maximize E[D(expert)] - E[D(policy)]
    wasserstein_loss = torch.mean(tanh_d_pi) - torch.mean(tanh_d_exp)

    # compute gradient penalty
    alpha = torch.rand(state_action_exp.size(0), 1, device=device)
    # interpolate between expert and policy samples
    interpolates = alpha * state_action_exp + (1 - alpha) * state_action_pi
    interpolates.requires_grad_(True)

    # 计算插值点的判别器输出
    disc_interpolates = discriminator_net(interpolates)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Calculate gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    # lambda_ is a hyperparameter for the gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_

    # Combine the Wasserstein loss and gradient penalty
    total_loss = wasserstein_loss + gradient_penalty

    return total_loss
