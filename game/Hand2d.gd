extends Node2D
var handlandmark = preload("res://handlandmark.gd")
const HAND_PALM: Array = [
	  Vector2i(0, 1),
	  Vector2i(1, 5),
	  Vector2i(9, 13),
	  Vector2i(13, 17),
	  Vector2i(5, 9),
	  Vector2i(0, 17),
]

const HAND_THUMB: Array = [
	  Vector2i(1, 2),
	  Vector2i(2, 3),
	  Vector2i(3, 4),
]

const HAND_INDEX_FINGER: Array = [
	  Vector2i(5, 6),
	  Vector2i(6, 7),
	  Vector2i(7, 8),
  ]

const HAND_MIDDLE_FINGER: Array = [
	  Vector2i(9, 10),
	  Vector2i(10, 11),
	  Vector2i(11, 12),
]

const  HAND_RING_FINGER: Array= [
	  Vector2i(13, 14),
	  Vector2i(14, 15),
	  Vector2i(15, 16),
]

const HAND_PINKY_FINGER: Array = [
	  Vector2i(17, 18),
	  Vector2i(18, 19),
	  Vector2i(19, 20),
]

const HAND: Array = (
	  HAND_PALM +
	  HAND_THUMB +
	  HAND_INDEX_FINGER +
	  HAND_MIDDLE_FINGER +
	  HAND_RING_FINGER +
	  HAND_PINKY_FINGER
  )
@onready
var OUT_W:int = 1920
@onready
var OUT_H:int = 1080
const keypoints = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# need only the pointer finger for now
#var keypoints = [1,2,3,4]
const pixels_per_meter = 100
var scale_x = 1
var scale_y = 1
var handData: Array = []

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	var screen_size = get_viewport().size
	print("Viewport", screen_size)
	OUT_W = screen_size.x
	OUT_H = screen_size.y
#	axis * screen_length / pixels per meter
	scale_x = OUT_W / pixels_per_meter * -1
	scale_y = OUT_H / pixels_per_meter * -1
	print("Scale X:", scale_x)
	print("Scale Y:", scale_y)

const draw_color = Color(1, 0, 0) # Red color for drawing
const draw_size = 5 # Size of the points

func _draw() -> void:
	if handData.is_empty():
		return
	# Draw points for each landmark
	for point in handData:
		draw_circle(point, draw_size, draw_color)

	# Draw lines for each connection
#	# TODO: use draw polygon instead??
	for connection in HAND:
		var start_point = handData[connection.x]
		var end_point = handData[connection.y]
		draw_line(start_point, end_point, draw_color)
func _process(delta: float) -> void:
	pass

func _on_arcane_hand_drawing(v: Array) -> void:
	handData = v.map(func(d):
		var x = d[0]# - 0.5
		var y = d[1]# - 0.5
		#var z = d[2] - 0.5
		x *= OUT_W
		y *= OUT_H
		# var z = 0
		return Vector2(x,y)
	)
	queue_redraw()
