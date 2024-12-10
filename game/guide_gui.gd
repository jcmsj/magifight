extends RichTextLabel

var ArcaneState = preload("res://ArcaneState.gd")

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	update_me()

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

func update_me() -> void:
	var ast = get_parent().arcaneState
	var state_str = "State: "
	if ast == ArcaneState.CASTING:
		state_str += "Casting"
	elif ast == ArcaneState.TARGETING:
		state_str += "Targeting"
	self.clear()
	self.append_text(state_str)

func _on_arcane_cast_spell(spellname: String) -> void:
	update_me()

func _on_arcane_end_effect() -> void:
	update_me()
	
