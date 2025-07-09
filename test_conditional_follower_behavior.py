# test_fsm_follower_behavior.py
#
# Leader: take-off → short move → land  (same as before)
# Follower: phase-classifier ⟶ FollowerFSM ⟶ velocity set-points
#
# Requires:  cflib, jax, flax, optax, orbax, etils
# ---------------------------------------------------------------------------

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"        # run JAX on CPU
import time, threading, numpy as np, jax, jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax, orbax.checkpoint as oc
from etils import epath

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.commander import Commander
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.log import LogConfig

# ---------------- Actor (phase classifier) ---------------------------------
class Actor(nn.Module):
	act_dim: int = 3
	@nn.compact
	def __call__(self, x):
		x = nn.tanh(nn.Dense(256)(x))
		x = nn.tanh(nn.Dense(256)(x))
		mean    = nn.Dense(self.act_dim,
		                   kernel_init=nn.initializers.orthogonal(0.01))(x)
		log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
		std     = jnp.exp(log_std)
		logits  = nn.Dense(3)(x)    # 3 behaviour classes
		return mean, std, logits

# ---------------- Finite-state follower controller -------------------------
class FollowerFSM:
	R_HOVER     = 1.0
	FOLLOW_DIST = 1.0
	MIN_DIST    = 0.5
	KP_DEFAULT  = 2.0
	KP_LAND     = 1.2
	ASCEND_H    = 0.7
	Z_TOL       = 0.03
	MAX_ACT_VEL = 1.0             # model space  ↦ real space scaled later

	def __init__(self):
		self.theta = 0.0
		self.prev_phase = -1
		self.landing_stage = 0
		self.target_apex = 0.0

	def _on_phase_change(self, phase, follower_pos, leader_pos):
		if phase == 0:	# HOVER
			rel = follower_pos[:2] - leader_pos[:2]
			if np.linalg.norm(rel) < 1e-3:
				rel = np.array([self.R_HOVER, 0.0])
			self.theta = np.arctan2(rel[1], rel[0])
		elif phase == 2:	# LAND
			self.landing_stage = 0
			self.target_apex = follower_pos[2] + self.ASCEND_H

	def step_from_phase(self, phase, *, follower_pos, leader_pos, leader_vel, dt):
		if phase != self.prev_phase:
			self._on_phase_change(phase, follower_pos, leader_pos)
			self.prev_phase = phase

		if phase == 0:		# HOVER: circle around leader
			self.theta = (self.theta + 2*np.pi*dt/4) % (2*np.pi)
			target = leader_pos + np.array(
				[self.R_HOVER*np.cos(self.theta),
				 self.R_HOVER*np.sin(self.theta), 0.0])
			kp = self.KP_DEFAULT

		elif phase == 1:	# MOVE: follow behind velocity vector
			offset = -leader_vel.copy(); offset[2] = 0.0
			if np.linalg.norm(offset) < 1e-3:
				offset = np.array([1.0, 0.0, 0.0])
			offset = self.FOLLOW_DIST * offset / np.linalg.norm(offset)
			target = leader_pos + offset
			kp = self.KP_DEFAULT

		else:				# LAND: ascend → descend
			kp = self.KP_LAND
			if self.landing_stage == 0:
				target = follower_pos.copy(); target[2] = self.target_apex
				if follower_pos[2] >= self.target_apex - self.Z_TOL:
					self.landing_stage = 1
			else:
				target = follower_pos.copy(); target[2] = 0.0

		cmd = kp * (target - follower_pos)
		cmd = np.clip(cmd, -self.MAX_ACT_VEL, self.MAX_ACT_VEL)

		# bumper so we never collide with leader (except in land phase)
		next_pos = follower_pos + cmd*dt
		flat = next_pos - leader_pos; flat[2] = 0.0
		dist = np.linalg.norm(flat)
		if dist < self.MIN_DIST and phase != 2:
			outward = flat / (dist + 1e-6)
			cmd[:2] = outward[:2] * self.MAX_ACT_VEL
		return cmd

# ---------------- Shared state --------------------------------------------
leader_state   = {"pos": np.zeros(3), "vel": np.zeros(3)}
follower_state = {"pos": np.zeros(3), "vel": np.zeros(3)}
done_event     = threading.Event()

def leader_cb(ts, data, _):
	leader_state["pos"] = np.array([data[f"stateEstimate.{k}"] for k in "xyz"])
	leader_state["vel"] = np.array([data[f"stateEstimate.v{k}"] for k in "xyz"])

def follower_cb(ts, data, _):
	follower_state["pos"] = np.array([data[f"stateEstimate.{k}"] for k in "xyz"])
	follower_state["vel"] = np.array([data[f"stateEstimate.v{k}"] for k in "xyz"])

# ---------------- Leader flight thread ------------------------------------
def leader_thread(uri: str):
	with SyncCrazyflie(uri, Crazyflie(rw_cache="./cache")) as scf:
		scf.cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
		scf.cf.param.set_value("kalman.resetEstimation", "0"); time.sleep(1.0)

		log = LogConfig("Leader", 100)
		for v in ("x","y","z","vx","vy","vz"):
			log.add_variable(f"stateEstimate.{v}", "float")
		scf.cf.log.add_config(log); log.data_received_cb.add_callback(leader_cb); log.start()

		hlc: HighLevelCommander = scf.cf.high_level_commander
		hlc.takeoff(0.7, 2.0); time.sleep(8.0)                      # hover
		hlc.go_to(0.7, 0.0, 0.7, 0.0, 5.0); time.sleep(5.0)         # move forward
		hlc.land(0.0, 2.0); time.sleep(3.0)
		done_event.set(); print("[Leader] Flight complete")

# ---------------- Follower flight thread ----------------------------------
def follower_thread(uri: str, actor: Actor, params, dt=0.1, v_max=0.3, noise_std=0.01):
	fsm = FollowerFSM()

	with SyncCrazyflie(uri, Crazyflie(rw_cache="./cache")) as scf:
		scf.cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
		scf.cf.param.set_value("kalman.resetEstimation", "0"); time.sleep(1.0)

		log = LogConfig("Follower", 100)
		for v in ("x","y","z","vx","vy","vz"):
			log.add_variable(f"stateEstimate.{v}", "float")
		scf.cf.log.add_config(log); log.data_received_cb.add_callback(follower_cb); log.start()

		hlc: HighLevelCommander = scf.cf.high_level_commander
		cmd:  Commander         = scf.cf.commander
		hlc.takeoff(0.7, 2.0); time.sleep(3.0)
		print("[Follower] FSM active")

		try:
			while not done_event.is_set():
				# ---------------- observation vector ------------------
				obs = np.concatenate((follower_state["pos"], follower_state["vel"],
				                      leader_state["pos"],   leader_state["vel"]))
				obs += np.random.normal(scale=noise_std, size=obs.shape)
				_, _, logits = actor.apply({"params": params},
				                           jnp.array(obs, dtype=jnp.float32))
				phase = int(jnp.argmax(logits))
				label = ("hover", "move", "land")[phase]

				# ---------------- FSM control -------------------------
				cmd_vel = fsm.step_from_phase(
					phase,
					follower_pos=follower_state["pos"],
					leader_pos=leader_state["pos"],
					leader_vel=leader_state["vel"],
					dt=dt)

				# model clip (±1 m/s) → real clip (±v_max m/s)
				cmd_vel = np.clip(cmd_vel, -1.0, 1.0) * v_max
				scf.cf.commander.send_velocity_world_setpoint(
					float(cmd_vel[0]), float(cmd_vel[1]), float(cmd_vel[2]), 0.0
				)
				print(f"[Follower] {label:<5} vel={cmd_vel}")
				time.sleep(dt)

		finally:
			print("[Follower] Landing")
			hlc.land(0.0, 2.0); time.sleep(3.0)
			cmd.send_stop_setpoint(); print("[Follower] Landed")

# ---------------- Main -----------------------------------------------------
if __name__ == "__main__":
	CKPT_DIR = epath.Path("trained_model/escort_follower_behavior")
	URI_LEADER   = "radio://0/60/2M/E7E7E7E7E1"
	URI_FOLLOWER = "radio://0/100/2M/E7E7E7E7E2"

	actor  = Actor()
	dummy  = TrainState.create(
		apply_fn=actor.apply,
		params=actor.init(jax.random.PRNGKey(0), jnp.zeros((12,)))["params"],
		tx=optax.adam(1e-3))
	dummy = oc.PyTreeCheckpointer().restore(CKPT_DIR, item=dummy)
	params = dummy.params

	cflib.crtp.init_drivers()
	t_lead = threading.Thread(target=leader_thread,   args=(URI_LEADER,))
	t_foll = threading.Thread(target=follower_thread, args=(URI_FOLLOWER, actor, params))
	t_lead.start(); time.sleep(0.5); t_foll.start()
	t_lead.join();  t_foll.join()
	print("FSM test complete.")
