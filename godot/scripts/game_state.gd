extends Node

# Current speed in km/h, driven by hardware input.
# Set to a non-zero value for testing without hardware.
var current_speed: float = 25.0

# Pedal cadence in RPM (if available from Pico)
var cadence: int = 0

# Heart rate in BPM from Coros HRM
var heart_rate: int = 0

# World position of the camera along the track (meters)
var world_position: float = 0.0
