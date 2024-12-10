class_name Reducio
extends Node3D

const growth_rate: float = 0.3  # 3% per second
var current_target = null

func _ready():
	set_physics_process(false)

func _physics_process(delta):
	if current_target == null:
		return
	# ex: 1.0 - 0.03 = 0.97
	var scale_factor = 1.0 - (growth_rate * delta)
	# adjust the basis's scale
	current_target.basis = current_target.basis.scaled(Vector3(scale_factor, scale_factor, scale_factor))
	
func _on_spell_target_do_spell_effect(d: Dictionary, spellname: String, from:Vector3) -> void:
	if spellname == 'Reducio':
		current_target = d.collider
		set_physics_process(true)

func _on_arcane_end_effect() -> void:
	current_target = null
	set_physics_process(false)
