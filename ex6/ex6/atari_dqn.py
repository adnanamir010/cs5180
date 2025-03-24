import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import copy
import tqdm
import time
import os
import gc  # Garbage collection
import ale_py
from collections import namedtuple
import matplotlib.pyplot as plt

# Simple namedtuple to hold batches of transitions
Batch = namedtuple('Batch', ['states', 'actions', 'rewards', 'next_states', 'dones'])

class ExponentialSchedule:
    def __init__(self, start_value, end_value, decay_steps):
        self.start_value = start_value
        self.end_value = end_value
        self.decay_steps = decay_steps
        
    def value(self, step):
        decay_rate = -np.log(self.start_value / self.end_value) / self.decay_steps
        current_value = self.start_value * np.exp(-decay_rate * min(step, self.decay_steps))
        return max(current_value, self.end_value)

class AtariDQN(nn.Module):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        self.input_shape = input_shape
        self.action_dim = action_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_output_size = self._get_conv_output_size(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def _get_conv_output_size(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        conv_out = self.conv_layers(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        q_values = self.fc_layers(flattened)
        return q_values

def make_atari_env(env_name, skip=4, stack=4):
    env = gym.make(env_name, render_mode=None)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=skip,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, stack)
    return env

class AtariReplayMemory:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.state_shape = state_shape
        
        # Use float16 to reduce memory usage by half
        self.states = np.zeros((capacity, *state_shape), dtype=np.float16)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float16)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float16)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.position = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        if hasattr(state, 'shape'):
            state = np.array(state, dtype=np.float16)  # Convert to float16
        if hasattr(next_state, 'shape'):
            next_state = np.array(next_state, dtype=np.float16)  # Convert to float16
            
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = Batch(
            states=self.states[indices].astype(np.float32),  # Convert back to float32 for training
            actions=self.actions[indices],
            rewards=self.rewards[indices].astype(np.float32),
            next_states=self.next_states[indices].astype(np.float32),
            dones=self.dones[indices]
        )
        
        return batch
        
    def __len__(self):
        return self.size

def train_atari_dqn(
    env_name,
    num_frames,
    *,
    num_saves=5,
    replay_size=500000,  # Reduced from 1M to 500K
    replay_prepopulate_steps=10000,  # Reduced from 50K to 10K
    batch_size=32,
    gamma=0.99,
    learning_rate=0.0001,
    target_update_frequency=10000,
    train_frequency=4,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_frames=1000000,
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_frequency=100,
    checkpoint_frequency=500000,  # Save checkpoints every 500K frames
    checkpoint_dir="checkpoints"
):
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Create the environment
    env = make_atari_env(env_name)
    
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"Observation shape: {input_shape}")
    print(f"Action space: {action_dim}")
    print(f"Device: {device}")
    print(f"Replay memory size: {replay_size} transitions")
    print(f"Replay memory estimated size: {replay_size * np.prod(input_shape) * 2 * 2 / (1024**2):.1f} MB")
    
    # Initialize models
    dqn_model = AtariDQN(input_shape, action_dim).to(device)
    dqn_target = AtariDQN(input_shape, action_dim).to(device)
    dqn_target.load_state_dict(dqn_model.state_dict())
    dqn_target.eval()
    
    optimizer = torch.optim.Adam(dqn_model.parameters(), lr=learning_rate)
    
    # Initialize replay memory with reduced size
    memory = AtariReplayMemory(replay_size, input_shape)
    
    exploration = ExponentialSchedule(epsilon_start, epsilon_end, epsilon_decay_frames)
    
    # Initialize tracking variables - store less data
    returns = []
    episode_lengths = []
    recent_losses = []  # Only store recent losses to save memory
    max_stored_losses = 10000
    epsilon_values = []
    episode_reward = 0
    episode_steps = 0
    episode_count = 0
    total_loss = 0
    loss_count = 0
    
    t_saves = np.linspace(0, num_frames, num_saves - 1, endpoint=False)
    saved_models = {}
    
    print("Prepopulating replay memory with random actions...")
    
    state, _ = env.reset()
    prepop_pbar = tqdm.tqdm(range(replay_prepopulate_steps), desc="Prepopulating")
    for _ in prepop_pbar:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        memory.add(state, action, reward, next_state, done)
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    print(f"Starting training for {num_frames} frames...")
    
    # Main training loop
    state, _ = env.reset()
    pbar = tqdm.tqdm(range(num_frames), desc="Training")
    
    for t in pbar:
        # Periodically run garbage collection
        if t % 10000 == 0:
            gc.collect()
            
        # Save model at specified intervals
        if t in t_saves:
            model_name = f'{100 * t / num_frames:04.1f}'.replace('.', '_')
            saved_models[model_name] = copy.deepcopy(dqn_model)
        
        # Checkpoint saving
        if t % checkpoint_frequency == 0 and t > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{env_name.replace('/', '_')}_frame_{t}.pt")
            torch.save({
                'frame': t,
                'model_state_dict': dqn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'returns': returns,
                'episode_lengths': episode_lengths,
                'epsilon_values': epsilon_values,
                'episode_count': episode_count
            }, checkpoint_path)
            print(f"\nCheckpoint saved to {checkpoint_path}")
        
        epsilon = exploration.value(t)
        
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = dqn_model(state_tensor)
                action = q_values.max(1)[1].item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        memory.add(state, action, reward, next_state, done)
        
        episode_reward += reward
        episode_steps += 1
        
        # Train DQN model
        if t % train_frequency == 0:
            batch = memory.sample(batch_size)
            
            states = torch.tensor(batch.states, dtype=torch.float32).to(device)
            actions = torch.tensor(batch.actions).long().view(-1, 1).to(device)
            rewards = torch.tensor(batch.rewards).float().view(-1, 1).to(device)
            next_states = torch.tensor(batch.next_states, dtype=torch.float32).to(device)
            dones = torch.tensor(batch.dones).float().view(-1, 1).to(device)
            
            q_values = dqn_model(states)
            q_values = q_values.gather(1, actions)
            
            with torch.no_grad():
                next_q_values = dqn_target(next_states).max(1, keepdim=True)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)
            
            loss = F.smooth_l1_loss(q_values, target_q_values)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn_model.parameters(), 10)
            optimizer.step()
            
            # Store loss data efficiently
            loss_value = loss.item()
            total_loss += loss_value
            loss_count += 1
            
            if len(recent_losses) < max_stored_losses:
                recent_losses.append(loss_value)
            else:
                # Replace a random old loss value with the new one
                idx = np.random.randint(0, max_stored_losses)
                recent_losses[idx] = loss_value
        
        # Update target network
        if t % target_update_frequency == 0:
            dqn_target.load_state_dict(dqn_model.state_dict())
        
        # Handle episode termination
        if done:
            returns.append(episode_reward)
            episode_lengths.append(episode_steps)
            epsilon_values.append(epsilon)
            episode_count += 1
            
            episode_reward = 0
            episode_steps = 0
            
            state, _ = env.reset()
            
            avg_return = np.mean(returns[-100:]) if returns else 0
            
            pbar.set_description(
                f'Episode: {episode_count} | Avg Return: {avg_return:.2f} | Epsilon: {epsilon:.3f}'
            )
            
            if episode_count % log_frequency == 0:
                avg_loss = total_loss / max(1, loss_count) if loss_count > 0 else 0
                total_loss = 0  # Reset for next logging period
                loss_count = 0
                
                elapsed_time = time.time() - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                print(f"\nEpisode {episode_count}:")
                print(f"  Frames: {t}/{num_frames} ({t/num_frames*100:.1f}%)")
                print(f"  Epsilon: {epsilon:.4f}")
                print(f"  Avg Return (last 100): {avg_return:.2f}")
                print(f"  Avg Loss: {avg_loss:.6f}")
                print(f"  Memory usage: {len(memory)}/{memory.capacity} transitions")
                print(f"  Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
                print(f"  Episodes per hour: {episode_count / (elapsed_time / 3600):.1f}")
                
                # Force garbage collection
                gc.collect()
        else:
            state = next_state
    
    # Save final model
    saved_models['100_0'] = copy.deepcopy(dqn_model)
    
    # Close environment
    env.close()
    
    # Calculate total training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining complete!")
    print(f"Total time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print(f"Total episodes: {episode_count}")
    print(f"Final average return (last 100): {np.mean(returns[-100:]):.2f}")
    
    return (
        saved_models,
        returns,
        episode_lengths,
        recent_losses,
        epsilon_values,
        training_time
    )

def plot_training_results(results, env_name, save_dir=None):
    """Simplified plotting that requires less memory"""
    saved_models, returns, episode_lengths, losses, epsilon_values, training_time = results
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot only returns (simpler)
    plt.figure(figsize=(10, 6))
    plt.plot(returns, alpha=0.4, label='Returns')
    
    # Moving average
    window_size = min(100, len(returns) // 10) if len(returns) > 10 else 1
    if len(returns) >= window_size:
        smoothed_returns = [np.mean(returns[max(0, i-window_size):i+1]) 
                         for i in range(len(returns))]
        plt.plot(smoothed_returns, linewidth=2, label=f'Moving Avg (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'DQN Learning Curve on {env_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{env_name.replace('/', '_')}_returns_{timestamp}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=200)
        print(f"Plot saved to {os.path.join(save_dir, filename)}")
    
    plt.show()
    
    # Clear memory
    plt.close()

# Save just the essential data to CSV
def save_essential_data(results, env_name, save_dir=None):
    if not save_dir:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{env_name.replace('/', '_')}_data_{timestamp}.csv"
    
    # Unpack results
    _, returns, episode_lengths, _, epsilon_values, _ = results
    
    # Save episode returns and lengths
    episode_data = np.column_stack((np.arange(len(returns)), returns, episode_lengths, epsilon_values))
    np.savetxt(
        os.path.join(save_dir, filename),
        episode_data,
        delimiter=',',
        header='episode,return,length,epsilon',
        comments=''
    )
    
    print(f"Data saved to {os.path.join(save_dir, filename)}")

# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Output directories
    save_dir = "dqn_results"
    checkpoint_dir = "dqn_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training parameters - reduce memory usage
    env_name = 'PongNoFrameskip-v4'
    num_frames = 5_000_000  # Reduced from 10M to 5M
    
    # Train the DQN with memory optimizations
    results = train_atari_dqn(
        env_name,
        num_frames=num_frames,
        replay_size=200_000,  # Further reduced from 500K to 200K
        replay_prepopulate_steps=5_000,  # Further reduced from 10K to 5K
        batch_size=32,
        gamma=0.99,
        learning_rate=0.0001,
        log_frequency=50,
        checkpoint_dir=checkpoint_dir
    )
    
    # Plot and save results (minimal version)
    plot_training_results(results, env_name, save_dir)
    save_essential_data(results, env_name, save_dir)
    
    # Save only the final model
    final_model = results[0]['100_0']
    model_path = os.path.join(save_dir, f"{env_name.replace('/', '_')}_final.pt")
    torch.save(final_model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")