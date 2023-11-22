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
    def __init(self,unique_id,model, agente: Agent = None):
        super().__init(unique_id, model)
        self.agente = Agent


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
        elif on_belt and not delivery and self.pos[1] != 0:
            self.sig_pos = (self.pos[0], self.pos[1] - 1)
        if self.robot != None:
            print(self.robot)
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
            self.robot = self.robot.sig_pos
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
        elif not self.delivery and self.pos[1] == 19 and self.iteration % self.speed_box_arrival == 0:
            box = Box(2000 + int(self.iteration / self.speed_box_arrival), self.model)
            self.model.grid.place_agent(box, self.pos)
            self.model.schedule.add(box)
            self.model.boxes.append(box)
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
        self.camino = []
        self.has_box = False


    def seleccionar_nueva_pos(self, lista_de_vecinos):
        self.sig_pos = self.random.choice(lista_de_vecinos).pos

    def step(self):
        vecinos = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)

        vecinos_disponibles = list()
        for vecino in vecinos:
            if not isinstance(vecino, (Rack, Robot, ConveyorBelt, Box)):
                vecinos_disponibles.append(vecino)

        self.sig_pos = self.random.choice(vecinos_disponibles).pos
        # Calcular ruta a cargador más cercano
        if (len(self.camino) == 0) and self.carga < 30 and not self.isCharging:
            dest = self.model.positions_chargers
            self.camino = self.aStar(dest)
        elif not self.has_box and self.model.box_waiting and len(self.camino) == 0:
            # Posicion rack de llegada de paquetes
            dest = [(0, 20)]
            self.camino = self.aStar(dest)
        
        if (self.isCharging and not isinstance(self.model.grid.__getitem__((self.pos[0], self.pos[1]))[0], Charger)):
            self.isCharging = False

        # Elegir ruta a cargador
        if (len(self.camino) > 0 and not self.isCharging and self.carga <= 30):
            estimated_sig_pos = self.camino[-1]       
            if not (isinstance(self.model.grid.__getitem__((estimated_sig_pos[0], estimated_sig_pos[1]))[0], Robot) and isinstance(self.model.grid.__getitem__((estimated_sig_pos[0], estimated_sig_pos[1]))[0], Charger)):  
                sig_pos = self.camino.pop()
                self.sig_pos = sig_pos
            if len(self.camino) == 0 and isinstance(self.model.grid.__getitem__((self.sig_pos[0], self.sig_pos[1]))[0], Charger):
                self.isCharging = True
        elif not self.model.box_waiting and len(self.camino) > 0:
            self.camino.clear()
        elif len(self.camino) > 0 and self.model.box_waiting:
            estimated_sig_pos = self.camino[-1]
            if not isinstance(self.model.grid.__getitem__((estimated_sig_pos[0], estimated_sig_pos[1]))[0], Robot):
                sig_pos = self.camino.pop()
                self.sig_pos = sig_pos
            if len(self.camino) == 0:
                for item in self.model.grid.__getitem__(self.sig_pos):
                    box: Box = None
                    if isinstance(item, Box): box = item
                if box != None:
                    box.robot = self
                    self.has_box = True
        else:
            self.seleccionar_nueva_pos(vecinos_disponibles)

        if self.carga >= 100:
            self.carga = 100
            self.isCharging = False

    def advance(self):
        robots = get_sig_positions(self.model)
        
        celdas_no_disponibles = list()
        cambio = False
        for robot in robots:
            if self.sig_pos == robot["sig_pos"] and self.unique_id != robot["unique_id"]:
                celdas_no_disponibles.append(robot["sig_pos"])
                cambio = True
        
        vecinos = self.model.grid.get_neighbors(
                        self.pos, moore=True, include_center=False)
        vecinos_disponibles = list()
        
        for vecino in vecinos:
            if not isinstance(vecino, (Rack, ConveyorBelt)) or (isinstance(vecino, Robot) and vecino.sig_pos not in celdas_no_disponibles):
                vecinos_disponibles.append(vecino)
        
        if cambio:
            self.seleccionar_nueva_pos(vecinos_disponibles)

        if self.pos != self.sig_pos:
            self.movimientos += 1

        if self.carga > 0:
            if self.isCharging and isinstance(self.model.grid.__getitem__((self.sig_pos[0], self.sig_pos[1]))[0], Charger): 
                    self.carga += 25
                    self.carga = min(self.carga, 100)
            else:
                self.carga -= 1
            self.model.grid.move_agent(self, self.sig_pos)

    def aStar(self, dest):
        start = self.sig_pos
        # dest = self.model.positions_chargers
        n = len(dest)

        # Priority queue - Procesar nodo con menor f (CellAlgorithm)
        open_list = list()
        # Almacena el path en orden inverso
        path = list()
        # Almacena los nodos visitados para saber si ya se visitaron en O(1)
        visited_nodes = set()
        # Almacena los detalles de cada nodo y se modifica en cada iteracion para reordenar adecuadamente la priority_queue
        cell_details = list()
        for i in range(self.model.grid.width):
            cell_details.append(list())
            for j in range(self.model.grid.height):
                cell_details[i].append(CellAlgorithm(x=i, y=j, n=n))
        
        # Cambiar datos de nodos destinos en cell_details
        destination_cells = list()
        for x, y in dest:
            cell_details[x][y].g = [0] * n
            cell_details[x][y].h = [0] * n
            cell_details[x][y].f = [0] * n
            destination_cells.append(cell_details[x][y])

        # Cambiar datos de nodo inicial en cell_details
        start_cell = cell_details[start[0]][start[1]]
        start_cell.g = [0] * n
        start_cell.h = self.heuristic(start_cell, destination_cells)
        start_cell.f = [start_cell.g[i] + start_cell.h[i] for i in range(n)]
        start_cell.parent_x = [start[0]] * n
        start_cell.parent_y = [start[1]] * n
        
        # Para cada destino (cargador) se calcula el path y se elige el más corto
        for i in range(n):
            visited_nodes.add((start[0], start[1]))
            open_list.append(start_cell)

            path.append(self.aStarHelper(cell_details, open_list, destination_cells, visited_nodes, i))
            open_list.clear()
            visited_nodes.clear()
        
        return min(path, key=len)

    def aStarHelper(self, cell_details, open_list, destination_cells, visited_nodes, i):
        import heapq
        path = list()
        
        # 8 movimientos posibles
        moves = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1),(-1, -1), (-1, 1), (1, -1)]
        # moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        while len(open_list) != 0:
            heapq.heapify(open_list)
            current_cell = heapq.heappop(open_list)
            x = current_cell.x
            y = current_cell.y

            for move in moves:
                new_x = x + move[0]
                new_y = y + move[1]

                if not self.isValid(new_x, new_y):
                    continue
                
                # Recorrer para todos los destinos
                if (new_x, new_y) == (destination_cells[i].x, destination_cells[i].y):
                        cell_details[new_x][new_y].parent_x[i] = x
                        cell_details[new_x][new_y].parent_y[i] = y
                        
                        row = destination_cells[i].x
                        col = destination_cells[i].y

                        while not (cell_details[row][col].parent_x[i] == row and 
                                   cell_details[row][col].parent_y[i] == col):
                            path.append((row, col))
                            temp_row = cell_details[row][col].parent_x[i]
                            temp_col = cell_details[row][col].parent_y[i]
                            row = temp_row
                            col = temp_col
                        return path
                # Checar que no sea un obstaculo y no haya sido visitado
                elif (not isinstance(self.model.grid.__getitem__((new_x, new_y))[0], (Rack, ConveyorBelt)) and
                      (new_x, new_y) not in visited_nodes
                      ):
                    new_G = current_cell.g[i] + 1
                    new_H = self.heuristic(cell_details[new_x][new_y], [destination_cells[i]])
                    new_F = new_G + new_H[0]

                    if (cell_details[new_x][new_y].f[i] == self.FLT_MAX or 
                        cell_details[new_x][new_y].f[i] > new_F):
                        open_list.append(cell_details[new_x][new_y])
                        cell_details[new_x][new_y].f[i] = new_F
                        cell_details[new_x][new_y].g[i] = new_G
                        cell_details[new_x][new_y].h[i] = new_H
                        cell_details[new_x][new_y].parent_x[i] = x
                        cell_details[new_x][new_y].parent_y[i] = y
             
    def heuristic(self, src:CellAlgorithm, dest:[]):
        import math
        # Distancia Euclidiana
        res = list()
        for cell in dest:
            res.append(math.sqrt((src.x - cell.x) ** 2 + (src.y - cell.y) ** 2))

        return res
    
    def isValid(self, row, col):
        return (row >= 0 and 
                row < self.model.grid.width and 
                col >= 0 and col < self.model.grid.height)


class Bodega(Model):
    def __init__(self, M: int = 50, N: int = 25,
                 num_robots: int = 5,
                 modo_pos_inicial: str = 'Aleatoria',
                 speed_box_arrival: int = 10
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