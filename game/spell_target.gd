extends Node3D

var pt = null
	
const RAY_LENGTH = 1_000
	
signal DoSpellEffect(d:Dictionary, spellname:String, from:Vector3)
var castTime = 1.0
var castTimer:float = -1.0
var last_result:Dictionary = {}
func _physics_process(delta: float) -> void:
	if pt == null:
		return
	# if current time is greater than cast timer's time, then check for the target
	#if Time.get_ticks_msec() < castTimer:
		#return
	var space_state = get_world_3d().direct_space_state
	var cam = get_viewport().get_camera_3d()
	var from = cam.project_ray_origin(pt)
	var to = from + cam.project_ray_normal(pt) * RAY_LENGTH
	var query = PhysicsRayQueryParameters3D.create(from, to)
	var result = space_state.intersect_ray(query)
	if result.is_empty():
		return
	if result.collider is not CharacterBody3D:
		return
		
	print("Targeted: ", result)
	set_physics_process(false)
	DoSpellEffect.emit(result, get_node('../Arcane').spell, from)
	castTimer = -1

func _on_arcane_targeting(crosshair: Vector2) -> void:
	pt = crosshair

func _on_arcane_cast_spell(spellname: String) -> void:
	set_physics_process(spellname != "Unknown")
