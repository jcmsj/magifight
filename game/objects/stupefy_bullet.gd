class_name StupefyBullet
extends CharacterBody3D

const SPEED = 8.0
var lifetime = 0.0
var target: CharacterBody3D
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	lifetime += delta
	if lifetime >= 2.0:
		queue_free()
func _physics_process(delta: float) -> void:
	## make the bullet move towards the direction it is facing
	if target == null:
		return

	var direction = target.global_position - global_position
	direction = direction.normalized()
	velocity.x = direction.x * SPEED
	velocity.z = direction.z * SPEED

	move_and_slide()
	for i in get_slide_collision_count():
		var collision: KinematicCollision3D = get_slide_collision(i)
		if collision.get_collider() == target:
			target.set_physics_process(false)
			# attach a timer that resumes the physics process after 3 seconds
			get_tree().create_timer(3.0).connect("timeout", target.set_physics_process.bind(true))
			queue_free()
			# Allow casting immediately
			get_node("../../Arcane").emit_signal('EndEffect')
