extends Node2D

var ip = "127.0.0.1"
var port = 4243
@onready
var server
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	server = UDPServer.new()
	server.listen(port, ip)

signal CastSpell(spellname:String)
var text = ['Wingardium Leviosa', 'Protego', 'Stupefy', 'Engorgio','Reducio', 'Unknown']
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	server.poll()
	if not server.is_connection_available():
		return
	var peer: PacketPeerUDP = server.take_connection()
	var packet = peer.get_packet()
	var json = JSON.parse_string(packet.get_string_from_utf8())
	if json != null:
		if typeof(json) == TYPE_FLOAT:
			CastSpell.emit(text[int(json)])
		else:
			print("Unexpected data", json)
	else:
		print('Err')
	
