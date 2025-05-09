from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from wgan import DiscriminatorNet, get_discriminator_loss, ExpertDataLoader
import torch as th
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from typing import TypeVar

from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

class SensorimotorPPO(PPO):
    """
    Custom PPO class for sensorimotor tasks.
    """
    def __init__(self, *args, **kwargs):
        self.device = 'cuda:0' if th.cuda.is_available() else 'cpu'
        # Wasserstein GAN model init
        
        if kwargs.get("wgan_descriminator_net") is None:
            print("Play model without WGAN")
        else:
            self.wgan_descriminator_net = kwargs.pop("wgan_descriminator_net")
            self.wgan_descriminator_lr = kwargs.pop("wgan_descriminator_lr")
            self.wgan_descriminator_betas = kwargs.pop("wgan_descriminator_betas")
            self.wgan_descriminator_lambda = kwargs.pop("wgan_descriminator_lambda")
            self.wgan_descriminator_eta = kwargs.pop("wgan_descriminator_eta")
            self.wgan_coef = kwargs.pop("wgan_coef")
            self.discriminator_net = DiscriminatorNet(
                self.wgan_descriminator_net[0], 
                self.wgan_descriminator_net[1], 
                self.wgan_descriminator_net[2],
                self.wgan_descriminator_net[3]).to(self.device)
            self.his_len = kwargs.pop("observation_history_len")
            self.single_step_obs = kwargs.pop("single_step_observation")
            self.expert_data_loader = ExpertDataLoader(kwargs.pop("expert_data"), self.his_len)
            self.opt_imizer = th.optim.Adam(
                self.discriminator_net.parameters(), 
                lr=self.wgan_descriminator_lr, 
                betas=(self.wgan_descriminator_betas[0], self.wgan_descriminator_betas[1])
            )
        
        super().__init__(*args, **kwargs)
        
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        style_losses = []
        wgan_train_loss = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                # exp(log_pi_new - log_pi_old) = exp(log(pi_new / pi_old)) = pi_new / pi_old = ratio
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                
                observation_size = rollout_data.observations.shape[0]

                # wgan_x_part1 = rollout_data.observations.view(observation_size, -1, self.single_step_obs)[:, :, :self.single_step_obs-36]
                # wgan_x_part2 = rollout_data.observations.view(observation_size, -1, self.single_step_obs)[:, :, self.single_step_obs-24:]
                # wgan_x = th.cat([wgan_x_part1, wgan_x_part2], dim=2).reshape(observation_size, -1)
                wgan_x = th.cat([
                    rollout_data.observations.view(observation_size, -1, self.single_step_obs)[:, :, :self.single_step_obs-12]
                ], dim=-1).reshape(observation_size, -1)
                
                style_loss = -self.discriminator_net(wgan_x).mean()
                style_losses.append(float(style_loss.item()))

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.wgan_coef * style_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
                exp_states_actions = self.expert_data_loader.get_expert_data(self.batch_size)
                wgan_loss = get_discriminator_loss(
                    self.discriminator_net,
                    exp_states_actions,
                    wgan_x,
                    lambda_=self.wgan_descriminator_lambda,
                    eta=self.wgan_descriminator_eta,
                    device=self.device,
                )
                self.opt_imizer.zero_grad()
                wgan_loss.backward()
                self.opt_imizer.step()
                wgan_train_loss.append(float(wgan_loss))

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("ppo/entropy_loss", np.mean(entropy_losses))
        self.logger.record("ppo/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("ppo/value_loss", np.mean(value_losses))
        self.logger.record("ppo/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("ppo/clip_fraction", np.mean(clip_fractions))
        self.logger.record("ppo/loss", loss.item())
        self.logger.record("ppo/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        # Add more training metrics
        # self.logger.record("train/learning_rate", self.learning_rate)
        self.logger.record("ppo_advantages/advantages_mean", th.mean(advantages).item())
        self.logger.record("ppo_advantages/advantages_std", th.std(advantages).item())
        self.logger.record("ppo_returns/returns_mean", th.mean(rollout_data.returns).item())
        self.logger.record("ppo_returns/returns_std", th.std(rollout_data.returns).item())
        self.logger.record("ppo_values/values_mean", th.mean(values).item())
        self.logger.record("ppo_values/values_std", th.std(values).item())
        self.logger.record("ppo_ratio/ratio_mean", th.mean(ratio).item())
        self.logger.record("ppo_ratio/ratio_std", th.std(ratio).item())

        self.logger.record("wgan/discriminator_loss", np.mean(wgan_train_loss))
        self.logger.record("wgan/style_loss", np.mean(style_losses))

        self.logger.record("ppo/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("ppo/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("ppo/clip_range_vf", clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            
            #! Change to the original code
            env.env.env.update_last_actions(clipped_actions)
            

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True