import time
import threading
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander

# Drone URIs
URI_1 = 'radio://0/60/2M/E7E7E7E7E1' #change this to your drone's URI
URI_2 = 'radio://0/100/2M/E7E7E7E7E2' #change this to your drone's URI

# Optional: Reset Kalman filter
def reset_estimator(scf):
	print(f"Resetting Kalman filter for {scf._link_uri}...")
	scf.cf.param.set_value('kalman.resetEstimation', '1')
	time.sleep(0.1)
	scf.cf.param.set_value('kalman.resetEstimation', '0')
	time.sleep(1.0)

# Optional: Send emergency stop
def emergency_stop(cf):
	cf.commander.send_stop_setpoint()
	print(f"Emergency stop sent to {cf.link_uri}")

# Drone's flight pattern
def flight_sequence(scf):
	try:
		reset_estimator(scf)
		hlc: HighLevelCommander = scf.cf.high_level_commander

		print(f"{scf._link_uri} Taking off...")
		hlc.takeoff(0.7, 2.0)
		time.sleep(3.0)

		print(f"{scf._link_uri} Moving forward 0.3m...")
		hlc.go_to(0.3, 0.0, 0.7, 0.0, 1.5)
		time.sleep(2.0)

		print(f"{scf._link_uri} Landing...")
		hlc.land(0.0, 2.0)
		time.sleep(2.5)

		hlc.stop()
		print(f"{scf._link_uri} Flight complete.")

	except KeyboardInterrupt:
		emergency_stop(scf.cf)
		raise

# One thread per drone
def drone_thread(uri):
	with SyncCrazyflie(uri, Crazyflie(rw_cache='./cache')) as scf:
		# Make sure the estimator is stable before flight
		reset_estimator(scf)
		flight_sequence(scf)

# Run both drones simultaneously
def run():
	cflib.crtp.init_drivers()
	uris = [URI_1, URI_2]
	threads = []

	try:
		for uri in uris:
			t = threading.Thread(target=drone_thread, args=(uri,))
			t.start()
			threads.append(t)

		for t in threads:
			t.join()

	except KeyboardInterrupt:
		print("\nInterrupted! Attempting emergency stop.")
		for uri in uris:
			try:
				cf = Crazyflie()
				cf.open_link(uri)
				emergency_stop(cf)
				time.sleep(0.2)
				cf.close_link()
			except Exception as e:
				print(f"Could not emergency stop {uri}: {e}")

	print("Flight session complete.")

if __name__ == '__main__':
	run()
