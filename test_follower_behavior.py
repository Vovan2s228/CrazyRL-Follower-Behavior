# test_follower_behavior.py

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
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

# ---------------- Actor -----------------------------------------------------------------
class Actor(nn.Module):
    act_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(256)(x))
        x = nn.tanh(nn.Dense(256)(x))
        mean    = nn.Dense(self.act_dim,
                           kernel_init=nn.initializers.orthogonal(0.01))(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
        std     = jnp.exp(log_std)
        logits  = nn.Dense(3)(x)
        return mean, std, logits  # continuous mean, std, class-logits

# ---------------- Shared State ----------------------------------------------------------
leader_state   = {"pos": np.zeros(3), "vel": np.zeros(3)}
follower_state = {"pos": np.zeros(3), "vel": np.zeros(3)}
done_event     = threading.Event()

def leader_cb(ts, d, _):
    leader_state["pos"] = np.array([d[f"stateEstimate.{k}"] for k in "xyz"])
    leader_state["vel"] = np.array([d[f"stateEstimate.v{k}"] for k in "xyz"])

def follower_cb(ts, d, _):
    follower_state["pos"] = np.array([d[f"stateEstimate.{k}"] for k in "xyz"])
    follower_state["vel"] = np.array([d[f"stateEstimate.v{k}"] for k in "xyz"])

# ---------------- Leader Thread ---------------------------------------------------------
def leader_thread(uri: str):
    with SyncCrazyflie(uri, Crazyflie(rw_cache="./cache")) as scf:
        scf.cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
        scf.cf.param.set_value("kalman.resetEstimation", "0"); time.sleep(1.0)

        log = LogConfig("Leader", 100)
        for v in ("x","y","z","vx","vy","vz"): log.add_variable(f"stateEstimate.{v}", "float")
        scf.cf.log.add_config(log); log.data_received_cb.add_callback(leader_cb); log.start()

        hlc: HighLevelCommander = scf.cf.high_level_commander
        hlc.takeoff(0.7, 2.0); time.sleep(8.0)
        hlc.go_to(0.7, 0.0, 0.7, 0.0, 5.0); time.sleep(5.0)   # 0.3 m forward
        hlc.land(0.0, 2.0); time.sleep(3.0)
        done_event.set(); print("[Leader] Flight complete")

# ---------------- Follower Thread -------------------------------------------------------
def follower_thread(uri: str, actor: Actor, params, noise_std=0.01, v_max=0.3):
    with SyncCrazyflie(uri, Crazyflie(rw_cache="./cache")) as scf:
        scf.cf.param.set_value("kalman.resetEstimation", "1"); time.sleep(0.1)
        scf.cf.param.set_value("kalman.resetEstimation", "0"); time.sleep(1.0)

        log = LogConfig("Follower", 100)
        for v in ("x","y","z","vx","vy","vz"): log.add_variable(f"stateEstimate.{v}", "float")
        scf.cf.log.add_config(log); log.data_received_cb.add_callback(follower_cb); log.start()

        hlc: HighLevelCommander = scf.cf.high_level_commander
        cmd: Commander        = scf.cf.commander
        hlc.takeoff(0.7, 2.0); time.sleep(3.0)
        print("[Follower] Hover & follow")

        try:
            while not done_event.is_set():
                # Observation
                obs = np.concatenate((follower_state["pos"], follower_state["vel"],
                                      leader_state["pos"],   leader_state["vel"]))
                obs += np.random.normal(scale=noise_std, size=obs.shape)
                mean, _, logits = actor.apply({"params": params},
                                              jnp.array(obs, dtype=jnp.float32))
                act = np.array(mean)             # continuous action
                act = np.clip(act, -1.0, 1.0)    # model output range
                vel_cmd = act * v_max            # scale to m/s
                scf.cf.commander.send_velocity_world_setpoint(
                    float(vel_cmd[0]), float(vel_cmd[1]), float(vel_cmd[2]), 0.0
                )
                pred = int(jnp.argmax(logits))
                label = ["hover", "move", "land"][pred]
                print(f"[Follower] {label:<5} vel={vel_cmd}")
                time.sleep(0.1)
        finally:
            print("[Follower] Landing")
            hlc.land(0.0, 2.0); time.sleep(3.0)
            cmd.send_stop_setpoint(); print("[Follower] Landed")

# ---------------- Main ------------------------------------------------------------------
if __name__ == "__main__":
    ckpt_dir = epath.Path("trained_model/escort_follower_behavior")
    ACT_DIM  = 3
    actor    = Actor(ACT_DIM)
    dummy    = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(jax.random.PRNGKey(0), jnp.zeros((12,)))["params"],
        tx=optax.adam(1e-3),
    )
    dummy = oc.PyTreeCheckpointer().restore(ckpt_dir, item=dummy)
    params = dummy.params

    URI_LEADER   = "radio://0/60/2M/E7E7E7E7E1"
    URI_FOLLOWER = "radio://0/100/2M/E7E7E7E7E2"

    cflib.crtp.init_drivers()
    t_lead = threading.Thread(target=leader_thread,   args=(URI_LEADER,))
    t_foll = threading.Thread(target=follower_thread, args=(URI_FOLLOWER, actor, params))
    t_lead.start(); time.sleep(0.5); t_foll.start()
    t_lead.join();  t_foll.join()
    print("Test complete.")
