import numpy as np
import os


class Record:
    def __init__(self, name, dir):
        self.name = name
        self.dir = dir
        self.obs = []
        self.critic_obs = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.actions = []
        self.check_dir()
    
    def check_dir(self):
        if not os.path.exists(self.dir):
            print(self.dir)
            os.makedirs(self.dir)
            print(f"Directory {self.dir} created.")
        else:
            print(f"Directory {self.dir} already exists.")

    def log(self, obs, critic_obs, rews, dones, infos, actions):
        self.obs.append(obs)
        self.critic_obs.append(critic_obs)
        self.rewards.append(rews)
        self.dones.append(dones)
        self.infos.append(infos)
        self.actions.append(actions)
        # Save to file or process as needed

    def save(self):
        dones_array = np.empty(len(self.dones), dtype=object)
        for i, d in enumerate(self.dones):
            dones_array[i] = d.cpu()
        
        np.savez(self.dir + '/' + self.name,
                 obs=np.array([o.cpu().numpy() for o in self.obs]),
                 critic_obs=np.array([co.cpu().numpy() for co in self.critic_obs]),
                 rewards=np.array([r.cpu().numpy() for r in self.rewards]),
                 dones=dones_array,
                 infos=np.array(self.infos),
                 actions=np.array([a.cpu().detach().numpy() for a in self.actions]))
        print(f"Data saved to {self.dir}/{self.name}.npz")

    def clear(self):
        self.obs = []
        self.critic_obs = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.actions = []
