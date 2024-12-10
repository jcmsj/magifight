extends RichTextLabel

func _on_arcane_cast_spell(spellname: String) -> void:
	# Draw the spell name in the rich text label
	if get_parent().arcaneState != 1:
		return
		
	print("Cast: " + spellname)

	# update width/height based on when spellname is written to normal font 
	# update text
	self.clear()
	self.append_text(spellname)
	
func _on_arcane_end_effect() -> void:
	self.clear()
	
