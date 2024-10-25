class Spells:
    wingardiumLeviosa = {
        'name': 'Wingardium Leviosa',
        'description': 'Levitates objects',
        'equation': []
    }
    stupefy = {
        'name': 'Stupefy',
        'description': 'Stuns a target',
    }   
    protego = {
        'name': 'Protego',
        'description': 'Shield Charm',
    }

    # class method
    @classmethod
    def cast(cls, spell:str):
        '''Cast a spell'''
        if spell == cls.wingardiumLeviosa['name']:
            print(f"{cls.wingardiumLeviosa['name']}! {cls.wingardiumLeviosa['description']}")
        elif spell == cls.stupefy['name']:
            print(f"{cls.stupefy['name']}! {cls.stupefy['description']}")
        elif spell == cls.protego['name']:
            print(f"{cls.protego['name']}! {cls.protego['description']}")
        else:
            print(f"Unknown spell: {spell}")

    @classmethod
    def from_gesture(cls, pts:list[list[float]]):
        '''Cast a spell from a gesture'''
