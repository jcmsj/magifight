extends Skeleton3D
var server: UDPServer
@onready
var OUT_W:int = 1920
@onready
var OUT_H:int = 1080
#var keypoints = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# need only the pointer finger for now
var keypoints = [1,2,3,4]
const pixels_per_meter = 100
var scale_x = 1
var scale_y = 1

#var keypoints = [9,10,11,12]
var ip = "172.26.0.1"
var port = 4242
func _ready() -> void:
	server = UDPServer.new()
	#server.listen(port, ip)
	keypoints = keypoints.map(func(i): 
		var bone = "Bone." + ("{0}".format([i]).lpad(3,"0"))
		var idx = find_bone(bone)
		return idx
	)
	var screen_size = get_viewport().size
	print("Viewport", screen_size)
	OUT_W = screen_size.x
	OUT_H = screen_size.y
#	axis * screen_length / pixels per meter
	scale_x = OUT_W / pixels_per_meter * -1
	scale_y = OUT_H / pixels_per_meter * -1

# Called every frame. 'delta'  is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	server.poll()
	if not server.is_connection_available():
		return
	var peer: PacketPeerUDP = server.take_connection()
	var packet = peer.get_packet()
	var json = JSON.new()
	var error = json.parse(packet.get_string_from_ascii())
	if error == OK:
		if typeof(json.data) == TYPE_ARRAY:
			# update my xyz
			var x = json.data[0][0] - 0.5
			var y = json.data[0][1] - 0.5
			#var z = json.data[0][1] # why is it working if im using the y value again
#			# IMPORTANT NOTE: assumed camera z pos is -5m
#			scaled_axis = axis * screen_length / pixels per meter
			var scaled_x = x * scale_x
			var scaled_y = y * scale_y
			var scaled_z = 0
			#var scaled_z = (-0.5+z) * OUT_W / pixels_per_meter * -1 # For now, dont adjust z
			position = Vector3(scaled_x,scaled_y,scaled_z)
			# TODO: scale and position landmarks
			#for i in keypoints:
				##var relative_i = i+1
				#x = json.data[i][0] - 0.5
				#y = json.data[i][1] - 0.5
				#scaled_x = x * scale_x * -1
				#scaled_y = y * scale_y * -1
				##scaled_x = x
				##scaled_y = y
				#var local_pos = Vector3(scaled_x,scaled_z,scaled_y)
				#set_bone_pose_position(i-1,local_pos)
	else:
		# reset hand
		localize_rests()
		print("Unexpected data", json.data)
 		#packet.get_string_from_utf8().d
  #print("Received: '%s' %s:%s" % [packet.get_string_from_utf8(), peer.get_packet_ip(), peer.get_packet_port()])
  # JSON decode

  # TODO: send greeting

signal DidMove(v:Vector3)
