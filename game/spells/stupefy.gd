extends Node3D

var last_target: CharacterBody3D
var bullet_scene = preload("res://objects/stupefy_bullet.tscn")
@onready
var arcane = get_node("../../Arcane")
@onready
var camera = get_viewport().get_camera_3d()

func _on_spell_target_do_spell_effect(d: Dictionary, spellname: String, _from:Vector3) -> void:
	if spellname != 'Stupefy':
		return
	# Convert 2d finger position to 3d world space
	var finger_ray_origin = camera.project_ray_origin(arcane.point)
	# Make bullet
	var bullet:StupefyBullet = bullet_scene.instantiate()
	bullet.scale = Vector3(0.02, 0.02, 0.02)
	bullet.target = d.collider
	get_parent().add_child(bullet)
	# scale the bullet to be smaller
	# save the target for later (move to physics process)
	last_target = d.collider
	# adjust finger ray origin to be a tiny bit in front of the camera
	finger_ray_origin += camera.global_transform.basis.z * -0.1
	bullet.global_transform.origin = finger_ray_origin

	bullet.look_at(last_target.position)
