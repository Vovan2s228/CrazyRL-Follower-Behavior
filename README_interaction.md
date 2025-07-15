# Interactive Signaling & Response System – Thesis Implementation 🇳🇱

This repository contains **only the code and assets developed for my Bachelor thesis**  
“Interactive Signaling and Response System” (Leiden University, 2025).  
It shows how a follower Crazyflie nano‑quadrotor can **recognise** the motion of a leader
drone and **respond** in real time, using a lightweight phase‑classifier and a deterministic
Finite‑State Machine (FSM).  
All generic CrazyRL simulation infrastructure lives upstream in the
[`CrazyRL`](https://github.com/your‑lab/crazyrl) project and is **not duplicated here**.

---

## ✈️  Project Overview

| Component | Purpose |
|-----------|---------|
| **Phase Classifier (JAX / Flax)** | 2‑layer MLP (256 units, `tanh`) that predicts the leader’s behaviour class—`hover`, `move`, or `land`—from a 12‑D relative state. |
| **Follower FSM (Python)** | Maps the predicted class to safe, interpretable manoeuvres: 0.5 m **orbit** (hover), 0.25 m **trail** (move), or a two‑stage **landing** routine. |
| **Real‑Time Controller** | Runs at ≈ 10 Hz on a ground‑station laptop, sends velocity set‑points over Crazyradio to both drones via **cflib High‑Level Commander**. |
| **Evaluation & Logging** | Scripts to capture trajectories, compute RMSE, classification accuracy, and export CSV / plots for the thesis figures. |

Key experimental results  
(simulation → real‑world, Flow Deck v2 + Mocap):

| Metric | Simulation | Real |
|--------|------------|------|
| Classification Accuracy | **97.8 %** | **91.9 %** |
| Trajectory RMSE | **0.319 m** | **1.617 m** |
| Control Smoothness (action var) | 2.43 m² s⁻² | 0.059 m² s⁻² |

---

## 🏗️  Repository Layout (Only part done by me)
```
.
├── evaluate_conditional_follower_behavior.py       
├── connection_moving_test.py                
├── test_follower_behavior.py                 
├── test_conditional_follower_behavior.py
├── train_follower_behavior.py
├── evaluate_follower_behavior.py                     
├── crazy_rl                                  
│   ├── multi_agent
│   │   ├── jax
│   │   │   ├── behavior_recognition   
│   │   │       └── behavior_recognition.py                            
│   │   │   └── escort
│   │   │       └── escort_follower_behavior.py
│   │   └── numpy                             ←
│   │       └── behavior_recognition
│   │           └── behavior_recognition.py
├── trained_model                             
│   ├── escort_follower_behavior
│   └── escort_follower_behavior_conditional
└── requirements.txt                          


May be more, I am not sure
```
---

## 🔧 Quick Start

> **Python 3.9.21 is required** (tested with Anaconda py39_21‑0).

```bash
# 1. Create & activate env
conda create -n signaling python=3.9.21
conda activate signaling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Connect two Crazyflie 2.1 drones + Crazyradio
#    (set channel, address, flow deck)
#    URI_L = "radio://0/60/2M/E7E7E7E7E1"
#    URI_F = "radio://0/100/2M/E7E7E7E7E2"

# 4. Make a connection test
python connection_moving_test.py

# 4. Fly the demos
python test_conditional_follower_behavior.py
python test_follower_behavior.py
```

The script will:

1. Open two Crazyradio links.  
2. Reset each drone’s Kalman estimator (`param set stabilizer.resetEstimation 1`).  
3. Execute the leader’s scripted **hover → move → land** routine.  
4. Run the follower’s classifier + FSM loop, streaming set‑points at ~ 10 Hz.  
5. Log positions, actions, and predicted phases to **CSV** (`logs/…`).  

Safety guards: emergency‐stop on exception, velocity clamps, 30 cm stand‑off.

---

## 🧑‍🔬  Reproducing the Training

All training uses the **JAX backend** of CrazyRL.
As a fallback, please use the CPU.

```bash
python scripts/train_classifier_jax.py \
       --env EscortFollowerBehavior-jax \
       --steps 200000 \
       --save_dir checkpoints/follower_classifier
```

Important flags:

* `--dense_distance_reward` — keep distance term continuous.  
* `--aux_phase_loss` — enables cross‑entropy head.  
* `--key_chain` — fresh PRNG keys each rollout (avoids divergence).  

See thesis §5 (“Methodology”) for hyper‑parameters and ablation studies.

---

## 📝  Thesis Highlights (What This Repo Adds)

* **Two‑Headed Policy**: joint control + classification network with PPO + CE loss.  
* **Key‑Chain PRNG Helper**: fixes hidden nondeterminism in vectorised roll‑outs.  
* **FrameAdapter Utility**: enforces consistent ENU ↔ CF coordinate frames.  
* **FSM Controller**: interpretable fallback that solved pure‑RL instability.  
* **Real‑World Validation**: first Crazyflie demo of implicit motion signaling at 2‑drone scale.

---

## 👤  Author

**Volodymyr Kalinin** — volodymyr.kalinin [at] student.leidenuniv.nl  
Supervisor: **Dr. Mike Preuss** (LIACS)  
