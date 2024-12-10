extends Node2D
var handlandmark = preload("res://handlandmark.gd")

var active = true
var active_col = Color(0, 1, 0) # Green
# var inactive_col = Color(1, 0, 0)  # Red
const RADIUS_CIRCLE = 5
func _draw() -> void:
	# draw a 5 radius circle at every pt
	for pt in get_parent().gesture:
		draw_circle(pt, RADIUS_CIRCLE, active_col)
		
func _on_arcane_gesturing(gesture_path: Array) -> void:
	queue_redraw()

func _on_arcane_end_effect() -> void:
	active = false;
	queue_redraw()

func _on_arcane_cast_spell(spellname: String) -> void:
	if active:
		queue_redraw()
		active = false
