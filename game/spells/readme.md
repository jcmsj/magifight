## This contains code for casting spells. Including determining the target, and determining when it has been casted.

### Wingardium Leviosa
1. Cast gesture
2. Target an object
3. Move its 3d position
### Engorgio / Reducio
1. Cast gesture
2. Target an object
3. Enlarge/Minimize it slowly at a rate of 3% per 1s
4. When the index finger tip unfocuses the object, end the spell

### Protego
1. Cast gesture
2. Put up a force field
3. Ends when user closes fist

### Stupefy
1. Cast gesture
2. Target an object
3. Play effect
4. Damage, destroy, do something to it

Props: a moving car

### Unknown
1. Show a warning


## Game Environment
The following objects should exist
2. A moving toy car in circles
1. A table
3. 

## Demo Idea
1. Float table/car w/ wingardium leviosa
2. Resize car
3. Temporarily stop car
4. cast protego to not get run over


Spell Casting
1. cast spell
2. lock spell gesture detection (sgd)
3. start target detection
4. if wingardium,
	await target
	move target
	await cancel gesture
	unlock sgd
5. if engorgio/reducio
	await target
	slowly change size
	await cancel gesture
	unlock  sgd
6. if barrrier
	await cancel
	unlock sgd
7. if stupefy
	await target
	halt movment for n secs
	unlock sgd
	
Targetable
	should be movable
		by itself
		by others
	resizable
	haltable

Spell:
	emits:
		cast
		target
		casted
	
Wingardium:Spell
	onCast:
		await the target
Arcane
:desc - A finite state machine based magic casting system
:implementation - must be attached to a node
:data
	spell - the spell currently being casted or null
	point - the index finger's current location or null
	hand - the current location of the hand keypoints
	gesture - the gesture points collected since the last spell was activated
:state
	cast - Spell Gesture Detection(SGD) is running
	target - Target Detection is running, after finding a target, execute the effect`
	effect - Executes the spell's effects, then goes back to cast state once it is done
	
Observers

TODO: Include spell name in DoEffect
