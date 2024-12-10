extends Camera3D

@export var move_speed: float = 5.0
@export var rotation_speed: float = 2.0

func _process(delta: float) -> void:
	var input_dir := Vector3.ZERO
	
	# Forward/Backward
	if Input.is_action_pressed("ui_up"):
		input_dir.z -= 1
	if Input.is_action_pressed("ui_down"):
		input_dir.z += 1
		
	# Left/Right
	if Input.is_action_pressed("ui_left"):
		input_dir.x -= 1
	if Input.is_action_pressed("ui_right"):
		input_dir.x += 1
		
	# Rotation
	if Input.is_key_pressed(KEY_Q):
		rotate_y(rotation_speed * delta)
	if Input.is_key_pressed(KEY_E):
		rotate_y(-rotation_speed * delta)
	
	# Apply movement relative to camera orientation
	position += transform.basis * (input_dir.normalized() * move_speed * delta)
