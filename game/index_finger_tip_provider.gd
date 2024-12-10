extends Node2D
var handlandmark = preload("res://handlandmark.gd")
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

signal IndexFingerTipPosition(v:Vector2)
func _on_hand_keypoint_provider_detect_hand(pts: Array) -> void:
	if pts.is_empty():
		return
	var tip = pts[8]
	if pts[16].y > pts[7].y and pts[20].y > pts[7].y:
		if tip.y < pts[7].y:
			var s = get_viewport().size
			# if pt 8 is below pt 7, then the user ain't pointing
			var tipLocation = Vector2(tip.x*s.x, tip.y * s.y)
			IndexFingerTipPosition.emit(tipLocation)
