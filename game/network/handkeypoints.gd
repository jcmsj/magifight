extends Node2D

var ip = "127.0.0.1"
var port = 4242
@onready
var server
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	server = UDPServer.new()
	server.listen(port, ip)

signal DetectHand(v:Array)
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	server.poll()
	if not server.is_connection_available():
		return
	var peer: PacketPeerUDP = server.take_connection()
	var packet = peer.get_packet()
	var json = JSON.new()
	var error = json.parse(packet.get_string_from_ascii())
	if error == OK:
		if typeof(json.data) == TYPE_ARRAY:
			DetectHand.emit(
				json.data.map(func(d): return Vector3(d[0], d[1], d[2]))
			)
		else:
			print("Unexpected data", json.data)
	else:
		print(error)
	
