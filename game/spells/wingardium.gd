extends Node3D

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

# a dictionary
#  {
#    position: Vector2 # point in world space for collision
#    normal: Vector2 # normal in world space for collision
#    collider: Object # Object collided or null (if unassociated)
#    collider_id: ObjectID # Object it collided against
#    rid: RID # RID it collided against
#    shape: int # shape index of collider
#    metadata: Variant() # metadata of collider
# }
#  Contains info like: { "position": (-3.480669, 0.005562, -2.536512), "normal": (0, 1, 0), "face_index": -1, "collider_id": 32682018115, "collider": Floor:<StaticBody3D#32682018115>, "shape": 0, "rid": RID(4668629450754) }
var last_target = {}

var tip = null
var speed = 1.0  # Adjust the speed as needed

func _physics_process(delta: float) -> void:
	# move the target's x or y to the tip's location using raycast
	if tip == null || last_target.is_empty() || last_target.collider is not CharacterBody3D:
		return
	
	var cam = get_viewport().get_camera_3d()
	var ray_direction = cam.project_ray_normal(tip)
	
	# Check distance to camera before moving
	var distance_to_camera = cam.global_position.distance_to(last_target.collider.global_position)
	var next_pos = last_target.collider.global_position + ray_direction * speed * delta
	var would_be_distance = cam.global_position.distance_to(next_pos)
	
	# Only move if we're not exceeding n meter distance
	#if would_be_distance <= 5.0:
	last_target.collider.global_translate(ray_direction * speed * delta)

	# TODO 1: dont let the object go out of view of the camera
	# TODO 2: or be too far from the camera
	# var pos = c.position
	# print(pos, "->", to)

func _on_spell_target_do_spell_effect(d: Dictionary, spellname:String, from:Vector3) -> void:
	if spellname != 'Wingardium Leviosa':
		return
	last_target = d
	last_target.collider.set_physics_process(false)

func _on_arcane_targeting(crosshair: Vector2) -> void:
	if last_target.is_empty():
		return
	tip = crosshair

func _on_arcane_end_effect() -> void:
	# basically clears
	if last_target.is_empty():
		return
	last_target.collider.set_physics_process(true)
	last_target = {};
	tip = null;
