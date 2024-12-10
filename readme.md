# MagiFight

## Description
MagiFight is a magical combat game where players can cast spells and battle against each other in a fantasy world.

## Directory Structure
```bash
/game - contains the godot project files
/models - constains the Machine Learning models
/datasets - constains the datasets used to train the models
    /spells - The spell classification dataset
/dist - Executables
```
## Prerequisites
- [Python 3.11 or later](https://www.python.org/downloads/)
- [Godot 4.3](https://godotengine.org/download/archive/4.3-stable)
## Installation
1. Clone the repository 
```bash
git clone https://github.com/jcmsj/magifight.git
cd magifight
```
2. Create a virtual environment
```bash
python -m venv venv
```
3. Activate the virtual environment
```bash
## Windows
.\.venv\Scripts\activate
## Linux
source venv/bin/activate
```
4. Install python dependencies:
```bash
python -m pip install -r requirements.txt
```
    
## Running the demo
1. Run the server
```bash
python server.py --headless
```
2. Open [the godot project](./game/project.godot) in Godot
3. Run the game
## Training:
1. Extract [./datasets/spells.7z](./datasets/spells.7z) to [./datasets/spells](./datasets/spells)
2. See help
```
python spell_classification.py --help
```

## Dataset Development
1. Annotate new data using the [Tracer](./tracer.py)
```bash
python tracer.py
```
2. You can count the number of annotated data using [Saver](./saver.py)
```bash
python saver.py
```

## Model Evaluation
1. See help
```bash
python spell_classification.py --help
```
2. Example: 
```bash
python spell_classification.py --arch harrynet --model ./models/harry/harrynet.ckpt --val-only --val-data ./datasets/spells
```

## Roadmap
### Phase 1: Spell Classification Using Spell Gestures (SCURGE)
- [x] Dataset development
- [x] Model training
- [x] Demo game world w/ Godot
- [X] Integration of SCURGE into the game
### Phase 2: World Development
### Phase 3: Multiplayer
