from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np

class CellAlgorithm():
    import sys
    FLT_MAX = sys.float_info.max
    def __init__(self, g=None, h=None, f=None, x=-1, y=-1, parent_x=None, parent_y=None, n=0):
        # Como se tienen varios destinos se ocupa una lista para cada atributo
        if g is None: self.g = [self.FLT_MAX for i in range(n)]
        else: self.g = g
        
        if h is None: self.h = [self.FLT_MAX for i in range(n)]
        else: self.h = h
        
        if f is None: self.f = [self.FLT_MAX for i in range(n)]
        else: self.f = f
        
        if parent_x is None: self.parent_x = [-1 for i in range(n)]
        else: self.parent_x = parent_x
        
        if parent_y is None: self.parent_y = [-1 for i in range(n)]
        else: self.parent_y = parent_y
        self.x = x
        self.y = y

    def __lt__(self, other):
        return self.f < other.f

class Cell(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class Charger(Agent):
    def __init(self,unique_id,model):
        super().__init(unique_id, model)


class Picker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.iteration = 0
        self.is_active = False
        self.capacity = 5
        self.max_capacity = 5
    
    def step(self):
        self.iteration += 1
        if self.iteration % 20 == 0:
            self.is_active = True
        if self.is_active:
            # for item in self.model.grid.__getitem__(self.pos):
            #     if isinstance(item, Box): self.capacity += 1
            if self.capacity == 0:
                self.is_active = False
                self.capacity = self.max_capacity

    def advance(self):
        pass


class Rack(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.box = None
    
    def step(self):
        # Rack de llegada de paquetes, checar si hay paquetes esperando
        if self.pos == (0, 20):
            for item in self.model.grid.__getitem__(self.pos):
                if isinstance(item, Box): self.model.box_waiting = True
                else: self.model.box_waiting = False
        
    def advance(self):
        pass


class Box(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sig_pos = None
        self.robot = None

    def step(self):
        on_belt = False
        delivery = False
        
        # Determinar si está en una cinta y si es de entrega o recolección
        for item in self.model.grid.__getitem__(self.pos):
            if isinstance(item, ConveyorBelt):
                on_belt = True
                delivery = item.delivery
                break
        
        # Si está en una cinta de entrega y no esta al final, avanzar hacia arriba
        # if on_belt and delivery and self.pos[1] != 19:
        if on_belt and delivery:
            self.sig_pos = (self.pos[0], self.pos[1] + 1)
        # Si está en una cinta de recolección, avanzar hacia abajo
        elif on_belt and not delivery and self.pos[1] != 0 or self.pos == (self.model.grid.width - 1, 20):
            self.sig_pos = (self.pos[0], self.pos[1] - 1)
        if self.robot != None:
            self.sig_pos = self.robot.sig_pos

    def advance(self):
        # Si la caja no se ocupa mover
        if self.sig_pos == None or self.sig_pos == self.pos: return
        
        can_move = True
        for item in self.model.grid.__getitem__(self.sig_pos):
            if isinstance(item, Box):
                can_move = False
                break
        
        # Checar si el robot que lo lleva cambia de dirección
        if self.robot != None and self.sig_pos != self.robot.sig_pos:
            self.sig_pos = self.robot.sig_pos
        if can_move:
            self.model.grid.move_agent(self, self.sig_pos)


class ConveyorBelt(Agent):
    def __init__(self, unique_id, model, speed_box_arrival, delivery=False):
        super().__init__(unique_id, model)
        self.delivery = delivery
        self.speed_box_arrival = speed_box_arrival
        self.iteration = 0
    
    def step(self):
        # Contador de iteraciones para la llegada de paquetes
        self.iteration += 1

        # Banda de llegada de paquetes - Crear paquete
        if self.delivery and self.pos[1] == 0 and self.iteration % self.speed_box_arrival == 0:
            box = Box(1500 + int(self.iteration / self.speed_box_arrival), self.model)
            self.model.grid.place_agent(box, self.pos)
            self.model.schedule.add(box)
            self.model.boxes.append(box)
        # elif not self.delivery and self.pos[1] == 19 and self.iteration % self.speed_box_arrival == 0:
        #     box = Box(2000 + int(self.iteration / self.speed_box_arrival), self.model)
        #     self.model.grid.place_agent(box, (self.pos[0], self.pos[1] + 1))
        #     self.model.schedule.add(box)
        #     self.model.boxes.append(box)
        # Banda de recolección de paquetes, eliminar paquete como si fuera de salida
        elif not self.delivery and self.pos[1] == 0:
            picker: Picker = None
            box: Box = None
            for item in self.model.grid.__getitem__(self.pos):
                if isinstance(item, Picker):
                    picker = item
                if isinstance(item, Box):
                    box = item
            if picker.is_active and box != None:
                picker.capacity -= 1
                self.model.grid.remove_agent(item)
                self.model.schedule.remove(item)
                self.model.boxes.remove(item)

    def advance(self):
        pass


class Robot(Agent):
    import sys
    FLT_MAX = sys.float_info.max
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.isCharging = False
        self.sig_pos = None
        self.movimientos = 0
        self.carga = 100
        self.has_box = False


    def seleccionar_nueva_pos(self, lista_de_vecinos):
        self.sig_pos = self.random.choice(lista_de_vecinos).pos

    def step(self):
        vecinos = self.model.grid.get_neighbors(
            self.pos, moore=False, include_center=False)

        vecinos_disponibles = list()
        rack_disponible_dejar: Rack = None
        rack_disponible_tomar: Rack = None
        for vecino in vecinos:
            if not isinstance(vecino, (Rack, Robot, ConveyorBelt, Box)): vecinos_disponibles.append(vecino)
            if isinstance(vecino, Rack) and vecino.box == None: rack_disponible_dejar = vecino
            elif isinstance(vecino, Rack) and vecino.box != None: rack_disponible_tomar = vecino


        self.sig_pos = self.random.choice(vecinos_disponibles).pos
        

    def advance(self):        
        if self.pos != self.sig_pos:
            self.movimientos += 1
        
        if self.carga > 0:
            self.carga -= 1
            self.model.grid.move_agent(self, self.sig_pos)


class Bodega(Model):
    def __init__(self, M: int = 50, N: int = 25,
                 num_robots: int = 5,
                 modo_pos_inicial: str = 'Aleatoria',
                 speed_box_arrival: int = 2
                 ):
        # Listas de agentes
        self.robots: list[Robot] = []
        self.chargers: list[Charger] = []
        self.racks: list[Rack] = []
        self.boxes: list[Box] = []
        self.conveyor_belts: list[ConveyorBelt] = []

        self.num_chargers = 4
        self.num_robots = num_robots

        self.grid = MultiGrid(M, N, False)
        self.schedule = SimultaneousActivation(self)

        self.box_waiting = False

        posiciones_disponibles = [pos for _, pos in self.grid.coord_iter()]

        # Posicionamiento de cintas
        belt_size = 20
        positions_belts = [(0, i) for i in range(belt_size)] + [(M - 1, i) for i in range(belt_size)]
        for id, pos in enumerate(positions_belts):
            if pos[0] == 0: belt = ConveyorBelt(15 + id, self, speed_box_arrival=speed_box_arrival, delivery=True)
            else: belt = ConveyorBelt(15 + id, self, speed_box_arrival=speed_box_arrival, delivery=False)
            self.grid.place_agent(belt, pos)
            posiciones_disponibles.remove(pos)
            self.schedule.add(belt)
            self.conveyor_belts.append(belt)

        # Posicionamiento de picker
        picker = Picker(60, self)
        self.grid.place_agent(picker, (M - 1, 0))
        self.schedule.add(picker)

        # Posicionamiento de racks
        positions_racks = list()
        for y in [3, 4, 8, 9, 13, 14, 18, 19]:
            positions_racks += [(x, y) for x in list(range(3, 11)) + list(range(14, 22)) + list(range(25, 33)) + list(range(37, 45))]
        positions_racks.append((0, belt_size))
        positions_racks.append((M - 1, belt_size))

        for id, pos in enumerate(positions_racks):
            mueble = Rack(70 + id, self)
            self.grid.place_agent(mueble, pos)
            posiciones_disponibles.remove(pos)
            if pos == (0, belt_size) or pos == (M - 1, belt_size):
                self.schedule.add(mueble)
            self.racks.append(mueble)

        # Posicionamiento de cargadores
        # Cargadores en mitad superior y inferior
        self.positions_chargers = [(int(M / 2) + i, j) for i in [-1, 1] for j in [0, N - 1]]
        for id, pos in enumerate(self.positions_chargers):
            charger = Charger(id + 10, self)
            self.grid.place_agent(charger, pos)
            posiciones_disponibles.remove(pos)
            self.chargers.append(charger)

        for id, pos in enumerate(posiciones_disponibles):
            celda = Cell(400 + id, self)
            self.grid.place_agent(celda, pos)

        # Posicionamiento de agentes robot
        if modo_pos_inicial == 'Aleatoria':
            pos_inicial_robots = self.random.sample(posiciones_disponibles, k=num_robots)
        else:  # 'Fija'
            pos_inicial_robots = [(1, 1)] * num_robots

        for id in range(num_robots):
            robot = Robot(id, self)
            self.grid.place_agent(robot, pos_inicial_robots[id])
            self.schedule.add(robot)
            self.robots.append(robot)

        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid},
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


def get_grid(model: Model) -> np.ndarray:
    """
    Método para la obtención de la grid y representarla en un notebook
    :param model: Modelo (entorno)
    :return: grid
    """
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        x, y = pos
        for obj in cell_content:
            if isinstance(obj, Robot):
                grid[x][y] = 2
            else:
                grid[x][y] = 1
    return grid

def get_sig_positions(model: Model):
    return [{"unique_id": agent.unique_id, "sig_pos" : agent.sig_pos} for agent in model.robots]

def get_positions(model: Model):
    return [{"unique_id": agent.unique_id, "pos" : agent.pos} for agent in model.robots]