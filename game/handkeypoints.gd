extends Node3D


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
signal DidMove(v:Vector3)


func _on_skeleton_3d_did_move(v: Vector3) -> void:
	DidMove.emit(v)
