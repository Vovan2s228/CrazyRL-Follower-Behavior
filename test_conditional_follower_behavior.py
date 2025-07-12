# test_conditional_follower_behavior.py
# Real-world test for conditional follower behavior using Crazyflie drones
# with classification jitter filter, orbit with altitude trim, and robust landing

import os
import time
import threading
import collections
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import orbax.checkpoint as oc
from etils import epath
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.commander import Commander
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.log import LogConfig

# -------------------- Parameters ----------------------------
SMOOTH_WINDOW      = 5      # ⬅ keep it short – only 0.3 s at 10 Hz
GROUND_Z_TOL       = 0.05   # m: altitude threshold to consider landed
LEADER_GROUND_WAIT = 10     # s: wait on ground before finishing

# Hysteresis thresholds (loosened)
VEL_ON  = 0.06  # m/s – enter move
VEL_OFF = 0.04  # m/s – exit move
# Exponential smoothing factor for velocity
ALPHA_VEL = 0.20   # was 0.30

# Orbit timing
ORBIT_PERIOD = 6.0   # s for one lap

# -------------------- Follower FSM --------------------------
class FollowerFSM:
    R_HOVER     = 0.5    # 0.5 m around the leader
    FOLLOW_DIST = 0.25    # 0.5 m behind when moving
    MIN_DIST    = 0.3    # bumper distance (must be < R_HOVER)

    KP_DEFAULT = 2.0
    KP_LAND    = 1.2
    ASCEND_H   = 0.7
    Z_TOL      = 0.03

    V_MAX_H    = 0.3   # m/s horizontal
    V_MAX_Z    = 0.2   # m/s vertical

    def __init__(self):
        self.theta = 0.0
        self.prev_phase = -1
        self.landing_stage = 0
        self.target_apex = 0.0

    def _on_phase_change(self, phase, follower_pos, leader_pos):
        if phase == 0:
            rel = follower_pos[:2] - leader_pos[:2]
            if np.linalg.norm(rel) > 0.3:        # reset only if we are ≥ 3 cm off
                self.theta = np.arctan2(rel[1], rel[0])
        elif phase == 2:  # land
            self.landing_stage = 0
            self.target_apex = follower_pos[2] + self.ASCEND_H

    def step_from_phase(self, phase, *, follower_pos, leader_pos, leader_vel, dt):
        if phase != self.prev_phase:
            self._on_phase_change(phase, follower_pos, leader_pos)
            self.prev_phase = phase

        if phase == 0:  # hover orbit
            self.theta = (self.theta + 2*np.pi*dt/ORBIT_PERIOD) % (2*np.pi)
            horiz_target = leader_pos[:2] + self.R_HOVER * np.array([
                np.cos(self.theta), np.sin(self.theta)
            ])
            z_diff = leader_pos[2] - follower_pos[2]
            vz_corr = np.clip(0.5 * z_diff, -0.05, 0.05)
            target = np.array([horiz_target[0], horiz_target[1], follower_pos[2] + vz_corr])
            kp = self.KP_DEFAULT

        elif phase == 1:  # move
            offset = -leader_vel.copy(); offset[2] = 0
            if np.linalg.norm(offset) < 1e-3:
                offset = np.array([1, 0, 0])
            offset = self.FOLLOW_DIST * offset / np.linalg.norm(offset)
            target = leader_pos + offset
            kp = self.KP_DEFAULT

        else:  # land
            kp = self.KP_LAND
            if self.landing_stage == 0:
                target = follower_pos.copy()
                target[2] = self.target_apex
                if follower_pos[2] >= self.target_apex - self.Z_TOL:
                    self.landing_stage = 1
            else:
                target = follower_pos.copy()
                target[2] = 0.0

        cmd = kp * (target - follower_pos)
        cmd = np.clip(cmd, -1.0, 1.0)

        nextp = follower_pos + cmd * dt
        horiz_err = nextp - leader_pos; horiz_err[2] = 0.0
        if np.linalg.norm(horiz_err) < self.MIN_DIST and phase != 2:
            outward = horiz_err / (np.linalg.norm(horiz_err) + 1e-6)
            cmd[:2] = outward[:2]

        if phase == 1:
            cmd[2] = 0.0

        if phase == 0:            # orbit → leave velocities in m/s
            vx, vy, vz = cmd
        else:                     # move & land → scale down
            vx = cmd[0] * self.V_MAX_H
            vy = cmd[1] * self.V_MAX_H
            vz = cmd[2] * self.V_MAX_Z
        return np.array([vx, vy, vz])

        return np.array([vx, vy, vz])

# -------------------- Actor & Classifier --------------------
class Actor(nn.Module):
    act_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(256)(x))
        x = nn.tanh(nn.Dense(256)(x))
        mean = nn.Dense(self.act_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
        std = jnp.exp(log_std)
        logits = nn.Dense(3)(x)
        return mean, std, logits

# -------------------- Shared State --------------------------
leader_state   = {"pos": np.zeros(3), "vel": np.zeros(3)}
follower_state = {"pos": np.zeros(3), "vel": np.zeros(3)}
done_event     = threading.Event()

# -------------------- Callbacks -----------------------------
def leader_cb(ts, data, _):
    leader_state["pos"] = np.array([data[f"stateEstimate.{k}"] for k in "xyz"])
    leader_state["vel"] = np.array([data[f"stateEstimate.v{k}"] for k in "xyz"])

def follower_cb(ts, data, _):
    follower_state["pos"] = np.array([data[f"stateEstimate.{k}"] for k in "xyz"])
    follower_state["vel"] = np.array([data[f"stateEstimate.v{k}"] for k in "xyz"])

# -------------------- Threads -------------------------------
def leader_thread(uri: str):
    with SyncCrazyflie(uri, Crazyflie(rw_cache="./cache")) as scf:
        # reset estimator
        scf.cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
        scf.cf.param.set_value("kalman.resetEstimation", "0"); time.sleep(1.0)

        # start logging
        log = LogConfig("Leader", 50)
        for v in ("x","y","z","vx","vy","vz"):
            log.add_variable(f"stateEstimate.{v}", "float")
        scf.cf.log.add_config(log)
        log.data_received_cb.add_callback(leader_cb)
        log.start()

        hlc = scf.cf.high_level_commander

        # TAKEOFF
        hlc.takeoff(0.7, 2.0)
        time.sleep(4.0)

        # 1) HOVER for 15s
        print("[Leader] Hover phase (15s)")
        time.sleep(15.0)

        # 2) MOVE forward 1 m (x +1)
        print("[Leader] Move forward 1 m")
        hlc.go_to(1.7, 0.0, 0.7, 0.0, 5.0)
        time.sleep(5.0)

        # 3) MOVE left 1 m (y +1)
        print("[Leader] Move left 1 m")
        hlc.go_to(0.7, 1.0, 0.7, 0.0, 5.0)
        time.sleep(5.0)

        # 4) LAND
        print("[Leader] Land phase")
        hlc.land(0.0, 2.0)
        time.sleep(5.0)

        # 5) WAIT on ground
        print("[Leader] On ground")
        time.sleep(LEADER_GROUND_WAIT)

        done_event.set()
        print("[Leader] Done")


def follower_thread(uri: str, actor: Actor, params):
    fsm = FollowerFSM()
    classify = jax.jit(lambda p, o: jnp.argmax(actor.apply({"params": p}, o)[2], axis=-1)[0])
    dt = 0.1

    # set up a tiny speed‐only vote buffer
    vote_buf = collections.deque(maxlen=SMOOTH_WINDOW)

    smooth_leader_vel = np.zeros(3)
    last_phase = 0
    latched_land = False
    landing_initiated = False

    with SyncCrazyflie(uri, Crazyflie(rw_cache="./cache")) as scf:
        scf.cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
        scf.cf.param.set_value("kalman.resetEstimation", "0"); time.sleep(1.0)

        log = LogConfig("Follower", 50)
        for v in ("x","y","z","vx","vy","vz"):
            log.add_variable(f"stateEstimate.{v}", "float")
        scf.cf.log.add_config(log)
        log.data_received_cb.add_callback(follower_cb)
        log.start()

        cmd = scf.cf.commander
        hlc = scf.cf.high_level_commander
        hlc.takeoff(0.7, 2.0); time.sleep(3.0)
        print("[Follower] Loop start")

        try:
            while True:
                if done_event.is_set():
                    latched_land = True

                L_Z  = leader_state["pos"][2]
                L_VZ = leader_state["vel"][2]
                smooth_leader_vel = (1 - ALPHA_VEL)*smooth_leader_vel + ALPHA_VEL*leader_state["vel"]
                speed = np.linalg.norm(smooth_leader_vel[:2])

                obs = np.concatenate((follower_state["pos"], follower_state["vel"],
                                      leader_state["pos"],   leader_state["vel"]))
                o   = jnp.array(obs, dtype=jnp.float32).reshape(1, -1)
                raw = int(classify(params, o))

                # latch land on descent + altitude
                if raw == 2 and (L_Z < 0.15) and (L_VZ < -0.02):
                    latched_land = True

                # -------------- PHASE LOGIC --------------
                moving_now = (speed > VEL_ON) if last_phase != 1 else (speed > VEL_OFF)
                vote_buf.append(moving_now)

                if latched_land:            # altitude/descent gate or leader done
                    phase = 2
                else:                       # need 3/3 agreeing “move” votes to flip
                    phase = 1 if all(vote_buf) else 0

                last_phase = phase

                # STEP FSM & SEND
                cmd_vel = fsm.step_from_phase(
                    phase,
                    follower_pos=follower_state["pos"],
                    leader_pos=leader_state["pos"],
                    leader_vel=leader_state["vel"],
                    dt=dt
                )
                cmd.send_velocity_world_setpoint(*cmd_vel, 0.0)
                print(f"[Follower] Phase={['hover','move','land'][phase]} vel={cmd_vel}")

                if phase == 2:
                    landing_initiated = True
                if landing_initiated and follower_state["pos"][2] < GROUND_Z_TOL:
                    print("[Follower] Ground reached, exiting follow loop")
                    break

                time.sleep(dt)

        finally:
            print("[Follower] Landing safe")
            hlc.land(0.0, 2.0); time.sleep(3.0)
            cmd.send_stop_setpoint()
            print("[Follower] Landed")

def main():
    ckpt   = oc.PyTreeCheckpointer()
    actor  = Actor(3)
    dummy  = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(jax.random.PRNGKey(0), jnp.zeros((1,12)))['params'],
        tx=optax.adam(1e-3),
    )
    dummy  = ckpt.restore(epath.Path("trained_model/escort_follower_behavior"), item=dummy)
    params = dummy.params

    cflib.crtp.init_drivers()
    URI_L = "radio://0/60/2M/E7E7E7E7E1"
    URI_F = "radio://0/100/2M/E7E7E7E7E2"

    tL = threading.Thread(target=leader_thread, args=(URI_L,))
    tF = threading.Thread(target=follower_thread, args=(URI_F, actor, params))
    tL.start(); time.sleep(0.5); tF.start()
    tL.join(); tF.join()
    print("Conditional follower test complete.")

if __name__ == "__main__":
    main()
