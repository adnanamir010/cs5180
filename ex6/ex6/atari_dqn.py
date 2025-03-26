import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import copy
import tqdm
from collections import namedtuple
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ale_py

# Simple namedtuple to hold batches of transitions
Batch = namedtuple('Batch', ['states', 'actions', 'rewards', 'next_states', 'dones'])

def plot_training_progress(returns, losses, window_size=100):
    """
    Plot the training progress including returns and loss
    
    Args:
        returns: List of episode returns
        losses: List of training losses
        window_size: Size of the window for smoothing the returns
    """
    # Use non-blocking mode for plots
    plt.ion()  # Turn on interactive mode
    
    # Clear previous figure if it exists
    plt.figure(1, figsize=(20, 10))
    plt.clf()
    
    # Plot returns
    plt.subplot(2, 1, 1)
    plt.title('Episode Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(returns, label='Returns', alpha=0.5)
    
    # Plot smoothed returns
    if len(returns) >= window_size:
        smoothed_returns = [np.mean(returns[max(0, i-window_size):i+1]) 
                           for i in range(len(returns))]
        plt.plot(smoothed_returns, label=f'Smoothed Returns (window={window_size})', color='orange')
    
    plt.legend()
    
    # Plot losses
    plt.subplot(2, 1, 2)
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    
    # If there are too many loss points, subsample for plotting
    max_points = 1000
    if len(losses) > max_points:
        indices = np.linspace(0, len(losses)-1, max_points, dtype=int)
        losses_to_plot = [losses[i] for i in indices]
    else:
        losses_to_plot = losses
    
    plt.plot(losses_to_plot)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # Short pause to update the figure

class ExponentialSchedule:
    def __init__(self, start_value, end_value, decay_steps):
        self.start_value = start_value
        self.end_value = end_value
        self.decay_steps = decay_steps
        
    def value(self, step):
        # Calculate linear decay factor between 0 and 1
        fraction = min(float(step) / self.decay_steps, 1.0)
        
        # Use exponential interpolation between start and end values
        return self.start_value * (self.end_value / self.start_value) ** fraction

class AtariDQN(nn.Module):
    def __init__(self, input_shape, action_dim):
        """
        DQN with convolutional layers for Atari games
        
        Args:
            input_shape: Tuple (frames, height, width) - typically (4, 84, 84) for frame-stacked grayscale images
            action_dim: Number of possible actions in the environment
        """
        super().__init__()
        
        # Store dimensions
        self.input_shape = input_shape
        self.action_dim = action_dim
        
        # Convolutional layers (following the architecture from the DQN paper)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the output from the last convolutional layer
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def _get_conv_output_size(self, shape):
        """Calculate the output size of the convolutional layers"""
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, frames, height, width)
        
        Returns:
            Q-values for each action
        """
        # Ensure the input has the correct shape
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension if needed
            
        # Forward through convolutional layers  
        conv_out = self.conv_layers(x)
        
        # Flatten the output for the fully connected layers
        flattened = conv_out.view(conv_out.size(0), -1)
        
        # Forward through fully connected layers to get Q-values
        q_values = self.fc_layers(flattened)
        
        return q_values
    
def make_atari_env(env_name, skip=4, stack=4):
    """
    Create an Atari environment with standard preprocessing
    Args:
        env_name: Name of the Atari environment
        skip: Number of frames to skip between actions
        stack: Number of frames to stack together
    Returns:
        Wrapped Atari environment
    """
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
        """
        Replay memory for Atari environments.
        
        Args:
            capacity: Maximum number of transitions to store
            state_shape: Shape of the state observations (typically (4, 84, 84) for frame stacks)
        """
        self.capacity = capacity
        self.state_shape = state_shape
        
        # Preallocate arrays for storage
        self.states = np.zeros((capacity, *state_shape), dtype=np.float16)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float16)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float16)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.position = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        """Add a new transition to the buffer"""
        # Convert LazyFrames to numpy array if necessary
        if hasattr(state, 'shape'):
            state = np.array(state)
        if hasattr(next_state, 'shape'):
            next_state = np.array(next_state)
            
        # Store transition
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """Sample a batch of transitions randomly"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = Batch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices]
        )
        
        return batch
    
    def __len__(self):
        return self.size

def analyze_training_results(saved_models, returns, episode_lengths, losses, q_tracking=None):
    """
    Create comprehensive analysis plots for DQN training results
    
    Args:
        saved_models: Dictionary of saved models at different training stages
        returns: List of episode returns
        episode_lengths: List of episode lengths
        losses: List of training losses
        q_tracking: Dictionary with Q-value tracking information
    """
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 20))
    
    # 1. Plot returns
    plt.subplot(3, 2, 1)
    plt.title('Episode Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(returns, alpha=0.5, label='Returns')
    
    # Plot smoothed returns with window size 100
    if len(returns) >= 100:
        smoothed_returns = [np.mean(returns[max(0, i-100):i+1]) 
                           for i in range(len(returns))]
        plt.plot(smoothed_returns, label='Smoothed Returns (window=100)', color='orange')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Plot episode lengths
    plt.subplot(3, 2, 2)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(episode_lengths, alpha=0.7)
    # Plot smoothed episode lengths
    if len(episode_lengths) >= 100:
        smoothed_lengths = [np.mean(episode_lengths[max(0, i-100):i+1]) 
                           for i in range(len(episode_lengths))]
        plt.plot(smoothed_lengths, color='green')
    plt.grid(True, alpha=0.3)
    
    # 3. Plot loss values
    plt.subplot(3, 2, 3)
    plt.title('Training Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    # If there are too many loss points, subsample for plotting
    max_points = 10000
    if len(losses) > max_points:
        indices = np.linspace(0, len(losses)-1, max_points, dtype=int)
        losses_to_plot = [losses[i] for i in indices]
        plt.plot(np.linspace(0, len(losses), max_points), losses_to_plot)
    else:
        plt.plot(losses)
    plt.grid(True, alpha=0.3)
    
    # 4. Plot Q-values if available
    if q_tracking and q_tracking['steps']:
        plt.subplot(3, 2, 4)
        plt.title('Q-Value Statistics')
        plt.xlabel('Training Step')
        plt.ylabel('Q-Value')
        plt.plot(q_tracking['steps'], q_tracking['avg_q'], label='Average Q-Value', color='blue')
        plt.plot(q_tracking['steps'], q_tracking['max_q'], label='Max Q-Value', color='red')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. Plot return distribution
    plt.subplot(3, 2, 5)
    plt.title('Return Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.hist(returns, bins=50, alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    # 6. Plot return vs episode length
    plt.subplot(3, 2, 6)
    plt.title('Return vs Episode Length')
    plt.xlabel('Episode Length')
    plt.ylabel('Return')
    plt.scatter(episode_lengths, returns, alpha=0.5, s=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_training_analysis.png', dpi=300)
    
    # Create an additional plot for comparing performance at different stages
    if len(saved_models) > 1:
        plt.figure(figsize=(15, 7))
        plt.title('Training Progress by Checkpoint')
        
        # Get stages as percentages
        stages = sorted([float(key.replace('_', '.')) for key in saved_models.keys() if key != '100_0'])
        
        # Create x-axis for bars
        x = np.arange(len(stages))
        
        # Get last 100 returns for each stage
        last_100_mean = []
        for stage in stages:
            stage_idx = int(stage * len(returns) / 100)
            last_100 = returns[max(0, stage_idx-100):stage_idx]
            last_100_mean.append(np.mean(last_100) if last_100 else 0)
        
        # Plot bar chart
        plt.bar(x, last_100_mean)
        plt.xlabel('Training Percentage')
        plt.ylabel('Avg Return (Last 100 Episodes)')
        plt.xticks(x, [f'{s}%' for s in stages])
        plt.grid(True, alpha=0.3)
        
        plt.savefig('dqn_checkpoint_comparison.png', dpi=300)
    
    plt.close('all')

def train_atari_dqn(
    env_name,
    num_frames,
    *,
    num_saves=5,
    replay_size=1000000,
    replay_prepopulate_steps=10000,
    batch_size=32,
    gamma=0.99,
    learning_rate=0.0001,
    target_update_frequency=10000,
    train_frequency=4,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_frames=1000000,
    initial_random_frames=50000,  # Added parameter for initial random actions
    device="cuda" if torch.cuda.is_available() else "cpu",
    plot_frequency=100  # Plot every 100 episodes
):
    """
    DQN algorithm for Atari environments with progress visualization.
    
    Args:
        - env_name: The name of the Atari environment (e.g., 'PongNoFrameskip-v4')
        - num_frames: Total number of frames to be used for training
        - num_saves: How many models to save to analyze the training progress
        - replay_size: Maximum size of the ReplayMemory
        - replay_prepopulate_steps: Number of steps with which to prepopulate the memory
        - batch_size: Number of experiences in a batch
        - gamma: The discount factor
        - learning_rate: Learning rate for the optimizer
        - target_update_frequency: Frequency of target network updates (in frames)
        - train_frequency: Frequency of training (in frames)
        - epsilon_start: Initial value for epsilon in epsilon-greedy
        - epsilon_end: Final value for epsilon in epsilon-greedy
        - epsilon_decay_frames: Number of frames over which to decay epsilon
        - initial_random_frames: Number of frames to take random actions at the start
        - device: Device to run the training on (cuda or cpu)
        - plot_frequency: How often to update the training progress plot (in episodes)
        
    Returns: (saved_models, returns, episode_lengths, losses)
        - saved_models: Dictionary whose values are trained DQN models
        - returns: List containing the return of each training episode
        - episode_lengths: List containing the length of each episode
        - losses: List containing the loss of each training batch
    """
    # Create the environment
    env = make_atari_env(env_name)
    
    # Get input shape and action space size
    input_shape = env.observation_space.shape
    action_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"Observation shape: {input_shape}")
    print(f"Action space: {action_dim}")
    print(f"Device: {device}")
    
    # Initialize the DQN and DQN-target models
    dqn_model = AtariDQN(input_shape, action_dim).to(device)
    dqn_target = AtariDQN(input_shape, action_dim).to(device)
    dqn_target.load_state_dict(dqn_model.state_dict())
    dqn_target.eval()  # Set target network to evaluation mode
    
    # Initialize the optimizer
    optimizer = torch.optim.Adam(dqn_model.parameters(), lr=learning_rate)
    
    # Initialize the replay memory
    memory = AtariReplayMemory(replay_size, input_shape)
    
    # Create epsilon schedule
    exploration = ExponentialSchedule(epsilon_start, epsilon_end, epsilon_decay_frames)
    
    # Initialize tracking variables
    returns = []
    episode_lengths = []
    losses = []
    episode_reward = 0
    episode_steps = 0
    episode_count = 0
    
    # Initialize structures to store the models at different stages of training
    t_saves = np.linspace(0, num_frames, num_saves - 1, endpoint=False)
    saved_models = {}
    
    print("Prepopulating replay memory with random actions...")
    
    # Prepopulate replay memory with random actions
    state, _ = env.reset()
    prepop_pbar = tqdm.tqdm(range(replay_prepopulate_steps), desc="Prepopulating")
    for _ in prepop_pbar:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Clip rewards to [-1, 1]
        reward = np.clip(reward, -1, 1)
        done = terminated or truncated
        
        # Store in replay memory
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
        # Save model at specified intervals
        if t in t_saves:
            model_name = f'{100 * t / num_frames:04.1f}'.replace('.', '_')
            saved_models[model_name] = copy.deepcopy(dqn_model)
        
        # Get current epsilon value
        epsilon = exploration.value(t)
        
        # Select action using epsilon-greedy policy with initial random phase
        if t < initial_random_frames:  # Always take random actions for the first 50K frames
            action = env.action_space.sample()
        elif np.random.random() < epsilon:
            # Explore: select random action
            action = env.action_space.sample()
        else:
            # Exploit: select greedy action
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = dqn_model(state_tensor)
                action = q_values.max(1)[1].item()
        
        # Execute action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Clip rewards to [-1, 1]
        reward = np.clip(reward, -1, 1)
        done = terminated or truncated
        
        # Store transition in replay memory
        memory.add(state, action, reward, next_state, done)
        
        # Update episode tracking
        episode_reward += reward
        episode_steps += 1
        
        # Train DQN model
        if t % train_frequency == 0:
            batch = memory.sample(batch_size)
            
            # Convert numpy arrays to tensors and move to device
            states = torch.tensor(np.array(batch.states), dtype=torch.float32).to(device)
            actions = torch.tensor(batch.actions).long().view(-1, 1).to(device)
            rewards = torch.tensor(batch.rewards).float().view(-1, 1).to(device)
            next_states = torch.tensor(np.array(batch.next_states), dtype=torch.float32).to(device)
            dones = torch.tensor(batch.dones).float().view(-1, 1).to(device)
            
            # Compute current Q values
            q_values = dqn_model(states)
            q_values = q_values.gather(1, actions)
            
            # Compute target Q values with special terminal state handling
            with torch.no_grad():
                next_q_values = dqn_target(next_states).max(1, keepdim=True)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)
                # Special terminal state handling - set target to -1 for terminal states
                target_q_values = target_q_values * (1 - dones) - dones
            
            # Compute loss and update model
            loss = F.smooth_l1_loss(q_values, target_q_values)  # Huber loss (smooth L1)
            
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to stabilize training
            torch.nn.utils.clip_grad_norm_(dqn_model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
        
        # Update target network
        if t % target_update_frequency == 0:
            dqn_target.load_state_dict(dqn_model.state_dict())
            
            # Compute Q-value stats
            if t % 10000 == 0:
                with torch.no_grad():
                    # Sample some states from memory to check Q-values
                    sample_states = torch.tensor(memory.states[:min(1000, memory.size)], dtype=torch.float32).to(device)
                    sample_q_values = dqn_model(sample_states)
                    avg_q = sample_q_values.mean().item()
                    max_q = sample_q_values.max().item()
                    # Save Q-values for later analysis
                    if not hasattr(train_atari_dqn, 'q_tracking'):
                        train_atari_dqn.q_tracking = {'steps': [], 'avg_q': [], 'max_q': []}
                    train_atari_dqn.q_tracking['steps'].append(t)
                    train_atari_dqn.q_tracking['avg_q'].append(avg_q)
                    train_atari_dqn.q_tracking['max_q'].append(max_q)
        
        # Handle episode termination
        if done:
            # Record episode statistics
            returns.append(episode_reward)
            episode_lengths.append(episode_steps)
            episode_count += 1
            
            # Reset episode tracking
            episode_reward = 0
            episode_steps = 0
            
            # Reset environment
            state, _ = env.reset()
            
            # Update progress bar with Q values if available
            avg_return = np.mean(returns[-100:]) if returns else 0
            
            if hasattr(train_atari_dqn, 'q_tracking') and train_atari_dqn.q_tracking['steps'] and train_atari_dqn.q_tracking['steps'][-1] // 10000 * 10000 == t // 10000 * 10000:
                # If we just calculated Q values this iteration, include them in progress bar
                avg_q = train_atari_dqn.q_tracking['avg_q'][-1]
                max_q = train_atari_dqn.q_tracking['max_q'][-1]
                pbar.set_description(
                    f'Episode: {episode_count} | Avg Return: {avg_return:.2f} | Epsilon: {epsilon:.3f} | Avg Q: {avg_q:.4f} | Max Q: {max_q:.4f}'
                )
            else:
                pbar.set_description(
                    f'Episode: {episode_count} | Avg Return: {avg_return:.2f} | Epsilon: {epsilon:.3f}'
                )
            
            # Visualize training progress periodically
            if episode_count % plot_frequency == 0:
                plot_training_progress(returns, losses)
        else:
            state = next_state
    
    # Save final model
    saved_models['100_0'] = copy.deepcopy(dqn_model)
    
    # Final visualization (non-blocking)
    plot_training_progress(returns, losses)
    
    # Close environment
    env.close()
    
    return (
        saved_models,
        returns,
        episode_lengths,
        losses
    )

if __name__ == "__main__":
    # Turn on interactive mode at the beginning
    plt.ion()
    
    results = train_atari_dqn(
        'PongNoFrameskip-v4',
        num_frames=10_000_000,  # 10M frames
        replay_size=500_000,    # Increase replay buffer size
        batch_size=32,
        learning_rate=0.00025,  # Original DQN paper value
        target_update_frequency=5000,  # More frequent updates
        epsilon_decay_frames=2_000_000,  # Slower epsilon decay
        initial_random_frames=50000  # Added parameter for initial random actions
    )

    # Extract results
    saved_models, returns, episode_lengths, losses = results
    
    # Turn off interactive mode for final plotting
    plt.ioff()
    
    # Create comprehensive analysis plots
    analyze_training_results(
        saved_models, 
        returns, 
        episode_lengths, 
        losses, 
        q_tracking=train_atari_dqn.q_tracking if hasattr(train_atari_dqn, 'q_tracking') else None
    )
    
    # Plot final learning curve summary (simpler version)
    plt.figure(figsize=(10, 6))
    plt.plot(returns, alpha=0.5)
    if len(returns) >= 100:
        smoothed_returns = [np.mean(returns[max(0, i-100):i+1]) for i in range(len(returns))]
        plt.plot(smoothed_returns, color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('DQN Learning Curve on Pong')
    plt.grid(True, alpha=0.3)
    plt.savefig('pong_learning_curve.png')
    
    print("\n========== Training Complete ==========")
    print(f"Total episodes: {len(returns)}")
    print(f"Final average return (last 100 episodes): {np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns):.2f}")
    print(f"Best episode return: {max(returns):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} steps")
    print(f"Total frames processed: {sum(episode_lengths)}")
    print("Saved comprehensive analysis plots to:")
    print("  - dqn_training_analysis.png")
    print("  - dqn_checkpoint_comparison.png")
    print("  - pong_learning_curve.png")
    print("=========================================")