import time
import numpy as np
from crazy_rl.multi_agent.numpy.circle.circle import Circle
from crazy_rl.multi_agent.numpy.surround.surround import Surround
from crazy_rl.multi_agent.numpy.escort.escort import Escort
from crazy_rl.multi_agent.numpy.catch.catch import Catch
import orbax.checkpoint
import jax.numpy as jnp
import jax

def load_model(model_path, model_constructor):
	checkpointer = orbax.checkpoint.PyTreeCheckpointer()
	params = checkpointer.restore(model_path)
	model = model_constructor()  # Instantiate the Actor model with action_dim
	return model, params

def run_environment(env_class, env_name, model_path=None):
    print(f"Launching simulation: {env_name}")

    # Common config
    drone_ids = np.array([0, 1, 2])
    init_flying_pos = np.array([[0, 0, 1], [1.5, 0, 1], [0.75, 1.3, 1]])

    # Handle different environment init parameters
    if env_name == "Circle":
        env = env_class(
            drone_ids=drone_ids,
            render_mode="human",
            init_flying_pos=init_flying_pos
        )
    elif env_name == "Surround":
        env = env_class(
            drone_ids=drone_ids,
            render_mode="human",
            init_flying_pos=init_flying_pos,
            target_location=np.array([0.75, 0.75, 1.0])  # Target point to surround
        )
    elif env_name == "Escort":
        env = env_class(
            drone_ids=drone_ids,
            render_mode="human",
            init_flying_pos=init_flying_pos,
            init_target_location=np.array([0.75, -1.0, 1.0]),  # Start point of target
            final_target_location=np.array([0.75, 2.0, 1.0])   # End point of target
        )
    elif env_name == "Catch":
        env = env_class(
            drone_ids=drone_ids,
            render_mode="human",
            init_flying_pos=init_flying_pos,
            init_target_location=np.array([0.75, 0.75, 1.0]),  # Target start position
            target_speed=np.array([0.2, 0.2, 0.0])              # Target velocity
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # Load the pretrained model if provided
    model = None
    params = None
    if model_path:
        print(f"Loading pretrained model from {model_path}")
        from learning.fulljax.mappo_fulljax import Actor
        action_dim = env.action_space(0).shape[0]  # Get action dimension from env
        model, params = load_model(model_path, lambda: Actor(action_dim))  # Pass constructor

    # Reset environment
    obs, info = env.reset()
    done = False
    step = 0

    while not done and step < 300:  # Limit to 300 steps (~30 sec)
        actions = {}

        for agent_id in env.possible_agents:
            if model and params:
                agent_index = int(agent_id.split("_")[1])  # e.g. 'agent_0' -> 0
                agent_id_one_hot = np.zeros(3)
                agent_id_one_hot[agent_index] = 1.0

                obs_input = np.concatenate([obs[agent_id], agent_id_one_hot])  # Now shape (9,)
                obs_array = jnp.array(obs_input)
                dist = model.apply(params["params"], obs_array)  # A distrax distribution
                action = dist.sample(seed=jax.random.PRNGKey(step))
            else:
                # Gentle small random actions
                action = 0.2 * (2 * np.random.rand(*env.action_space(agent_id).shape) - 1)
            actions[agent_id] = action

        obs, rewards, terminated, truncated, info = env.step(actions)

        if all(terminated.values()) or all(truncated.values()):
            done = True

        if step % 20 == 0:
            print(f"[{step}] Rewards: {rewards}")

        step += 1
        time.sleep(0.05)

    print("Simulation finished.")

if __name__ == "__main__":
    # Choose one environment to test:
    env_class = Circle
    #env_class = Surround
    #env_class = Escort
    #env_class = Catch

    env_name = env_class.__name__
    model_path = "c:/Users/kalin/Projects/CrazyRL/trained_model/actor_circle_3"  # Updated path
    run_environment(env_class, env_name, model_path)
