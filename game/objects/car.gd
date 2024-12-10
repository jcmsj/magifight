extends CharacterBody3D

const SPEED = 5.0
const ROTATION_SPEED = 0.8

func _physics_process(delta: float) -> void:
	# Add the gravity.
	if not is_on_floor():
		velocity += get_gravity() * delta

	rotation.y += ROTATION_SPEED * delta
	var direction = +transform.basis.z
	velocity.x = direction.x * SPEED
	velocity.z = direction.z * SPEED

	move_and_slide()
