all: 

drone_uwb_px4:
	@echo "Running Drone UWB PX4"
	bot-procman-sheriff -l procman/drone_uwb_px4 general
start_drone_uwb_sjtu:
	@echo "Running Start SJTU Drone with UWB"
	bot-procman-sheriff -l procman/drone_uwb_sjtu general