class_name Protego
extends Node3D

@onready
var camera = get_viewport().get_camera_3d()
var barrier = preload("res://spells/BarrierEffect.tscn").instantiate()

func _on_arcane_cast_spell(spellname: String) -> void:
	if spellname != 'Protego':
		return

	camera.add_child(barrier)
	# positions it in front of the camera
	barrier.position = Vector3(0, -0.5, -1)
	barrier.rotation_degrees.y = 90

func _on_arcane_end_effect() -> void:
	camera.remove_child(barrier)
