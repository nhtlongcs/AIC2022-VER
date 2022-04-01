class Vehicle(object):
    def __init__(self, vehicle: str, colors: list):
        self.vehicle = vehicle 
        self.colors = []
        self.combines = []    
        for col_pair in colors:
            color, adv = col_pair['color'], col_pair['adv']
            self.colors.append(color)
            if adv:
                self.combines.append(f'{adv}_{color}')
            else:
                self.combines.append(color)
        
        # self.combines = list(set(self.combines))

    def __str__(self):
        return f'veh={self.vehicle}, col={self.combines}'
        
        

