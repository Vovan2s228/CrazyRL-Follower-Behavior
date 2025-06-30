import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # registers the 3-D projection

# ------------------------------------------------------------------
# 1.  Load trajectories
# ------------------------------------------------------------------
follower = np.load("follower_traj.npy")   # shape (T, 3)
leader   = np.load("leader_traj.npy")     # shape (T, 3)

# ------------------------------------------------------------------
# 2.  3-D plot
# ------------------------------------------------------------------
fig = plt.figure(figsize=(7, 6))
ax  = fig.add_subplot(111, projection="3d")

ax.plot(leader[:, 0], leader[:, 1], leader[:, 2],
        lw=2, label="Leader", color="tab:blue")
ax.plot(follower[:, 0], follower[:, 1], follower[:, 2],
        lw=2, label="Follower", color="tab:orange")

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("3-D Trajectories of Leader and Follower")
ax.legend()
ax.set_box_aspect([1, 1, 0.6])            # nicer aspect ratio
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3.  Optional: time-series of altitude (helpful for the LAND phase)
# ------------------------------------------------------------------
plt.figure(figsize=(6, 3))
plt.plot(leader[:, 2],  label="Leader Z")
plt.plot(follower[:, 2], label="Follower Z")
plt.xlabel("Timestep")
plt.ylabel("Altitude [m]")
plt.title("Altitude Profiles")
plt.legend()
plt.tight_layout()
plt.show()
