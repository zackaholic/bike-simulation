extends Node

const UDP_PORT = 4433

var _udp: PacketPeerUDP

func _ready() -> void:
	print("HardwareInput ready")
	_udp = PacketPeerUDP.new()
	var err = _udp.bind(UDP_PORT, "127.0.0.1")
	if err != OK:
		push_error("HardwareInput: failed to bind UDP port %d (error %d)" % [UDP_PORT, err])

func _process(_delta: float) -> void:
	while _udp.get_available_packet_count() > 0:
		var raw = _udp.get_packet()
		var text = raw.get_string_from_utf8()
		_parse(text)

func _parse(text: String) -> void:
	var json = JSON.new()
	var err = json.parse(text)
	if err != OK:
		push_warning("HardwareInput: malformed packet: %s" % text)
		return
	var data = json.get_data()
	if data.has("speed"):
		GameState.current_speed = float(data["speed"])
	if data.has("cadence"):
		GameState.cadence = int(data["cadence"])
	if data.has("hr"):
		GameState.heart_rate = int(data["hr"])
