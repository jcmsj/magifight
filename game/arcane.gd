extends Node2D
## A finite arcaneState machine based magic casting system
var handlandmark = preload("res://handlandmark.gd")
var ArcaneState = preload("res://ArcaneState.gd")

## the spell currently being casted or null
var spell = ""
## the index finger's current location or null
var point = null # Vector2
# Array<Vector2>
var hand: Array = []
## the index finger points collected since ~the last spell was activated~ entering the current arcaneState. It is a list of Vector2
## Array<Vector2>
var gesture: Array = []

signal CastSpell(spellname: String)
signal HandDrawing(handkeypoints: Array)
signal Gesturing(gesture_path:Array)
signal Targeting(crosshair:Vector2)
signal EndEffect()
var arcaneState = ArcaneState.CASTING

func _on_hand_keypoint_provider_detect_hand(pts: Array) -> void:
	hand = pts
	HandDrawing.emit(hand)

	if pts.is_empty():	
		return
	var tip = pts[8]
	if pts[16].y > pts[7].y and pts[20].y > pts[7].y:
		if tip.y < pts[7].y:
			var s = get_viewport().size
			# if pt 8 is below pt 7, then the user ain't pointing
			point = Vector2(tip.x*s.x, tip.y * s.y)
			gesture.append(point)
			if arcaneState == ArcaneState.CASTING:
				Gesturing.emit(gesture)
			elif arcaneState == ArcaneState.TARGETING:
				Targeting.emit(point)

func _on_spell_provider_cast_spell(spellname: String) -> void:
	if arcaneState == ArcaneState.CASTING:
		if spellname == 'Unknown':
			gesture.clear()
			EndEffect.emit()
			return
		spell = spellname
		arcaneState = ArcaneState.TARGETING
		gesture.clear()
		CastSpell.emit(spell)
	elif arcaneState == ArcaneState.TARGETING:
		spell = ""
		arcaneState = ArcaneState.CASTING
		gesture.clear()
		EndEffect.emit()
