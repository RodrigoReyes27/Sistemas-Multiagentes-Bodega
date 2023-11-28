from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from random import seed
from random import randint
seed(1)
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
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.busy = False


class Picker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.iteration = 0
        self.is_active = False
        maxcap = randint(5,20)
        self.max_capacity = maxcap
        self.capacity = self.max_capacity
        self.orders = self.max_capacity
        self.wait_time = randint(150,300)
    
    def step(self):
        self.iteration += 1
   
        # Determinar si el camion va a estar activo (Cada cierto tiempo)
        if self.iteration % self.wait_time == 0:
            self.is_active = True
            seconds = randint(150,300)
            self.wait_time = seconds
            self.iteration = 0
        # Si el camion está activo y no tiene capacidad para llevar mas paquetes, desactivar
        if self.is_active and self.capacity == 0:
            self.is_active = False
            cap = randint(5,20)
            self.max_capacity = cap
            self.capacity = self.max_capacity
            self.orders = self.max_capacity

    def advance(self):
        pass


class Rack(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.box = None
    
    def step(self):
        # Rack de llegada de paquetes, avisar que hay un paquete esperando al Model
        if self.pos == (0, 20):
            box_waiting = False
            for item in self.model.grid.__getitem__(self.pos):
                if isinstance(item, Box): box_waiting = True
            self.model.box_waiting = box_waiting
        
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
        if on_belt and delivery:
            self.sig_pos = (self.pos[0], self.pos[1] + 1)
        # Si está en una cinta de recolección o rack de recoleccion, avanzar hacia abajo
        elif on_belt and not delivery and self.pos[1] != 0 or self.pos == (self.model.grid.width - 1, 20):
            self.sig_pos = (self.pos[0], self.pos[1] - 1)
        
        # TODO - Cambiar forma en la que se sigue al robot
        if self.robot != None:
            self.sig_pos = self.robot.sig_pos

    def advance(self):
        # Si la caja no se ocupa mover
        if self.sig_pos == None or self.sig_pos == self.pos: return
        
        can_move = True
        # Cuando esta en una banda determinar si no hay una caja en la siguiente posición
        for item in self.model.grid.__getitem__(self.sig_pos):
            if isinstance(item, Box):
                can_move = False
                break
        
        # Checar si el robot que lo lleva cambia de dirección
        # TODO - Cambiar forma en la que se sigue al robot
        if self.robot != None and self.sig_pos != self.robot.sig_pos:
            self.sig_pos = self.robot.sig_pos
        
        if can_move:
            self.model.grid.move_agent(self, self.sig_pos)


class ConveyorBelt(Agent):
    def __init__(self, unique_id, model, speed_box_arrival, delivery=False):
        super().__init__(unique_id, model)
        self.delivery = delivery
        self.speed_box_arrival = speed_box_arrival
        self.iteration = -1
        self.cajas = 0
    
    def step(self):
        # Contador de iteraciones para la llegada de paquetes
        self.iteration += 1

        # Banda de llegada de paquetes - Crear paquete
        if self.delivery and self.pos[1] == 0 and self.iteration % self.speed_box_arrival == 0 and (self.cajas < 800 or self.cajas > 1500):
            # No generar paquete si hay en la primera posición de cinta
            for item in self.model.grid.__getitem__(self.pos):
                if isinstance(item, Box): 
                    self.iteration -= 1
                    return
            # Generar paquete
            box = Box(1500 + int(self.iteration / self.speed_box_arrival), self.model)
            self.cajas += 1
            self.model.grid.place_agent(box, self.pos)
            self.model.schedule.add(box)
            self.model.boxes.append(box)
        # Prueba de banda de recolección de paquetes - Crear paquete y testear recolección
        # elif not self.delivery and self.pos[1] == 19 and self.iteration % self.speed_box_arrival == 0:
        #     box = Box(2000 + int(self.iteration / self.speed_box_arrival), self.model)
        #     self.model.grid.place_agent(box, (self.pos[0], self.pos[1] + 1))
        #     self.model.schedule.add(box)
        #     self.model.boxes.append(box)
        # Banda de recolección de paquetes - eliminar paquete como si fuera de salida
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
                self.model.grid.remove_agent(box)
                self.model.schedule.remove(box)
                self.model.boxes.remove(box)
        else:
            self.cajas += 1
        
        if self.cajas == 1500:
            self.cajas = 0

    def advance(self):
        pass


class Robot(Agent):
    import sys
    FLT_MAX = sys.float_info.max
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.isCharging = False
        self.isGoingToCharge = False
        self.wait = False
        self.charger = None
        self.sig_pos = None
        self.movimientos = 0
        self.carga = 100
        self.box: Box = None
        self.path: list = []
        self.action = None
        

    def seleccionar_nueva_pos(self, lista_de_vecinos):
        self.sig_pos = self.random.choice(lista_de_vecinos).pos

    def step(self):
        vecinos = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)

        vecinos_disponibles = list()
        for vecino in vecinos:
            if not isinstance(vecino, (Rack, Robot, ConveyorBelt, Box)): vecinos_disponibles.append(vecino)

        if (self.isCharging and isinstance(self.model.grid.__getitem__((self.pos[0], self.pos[1]))[0], Charger)):
            self.carga += 1

        # Si ya se completo la tarea que iba a hacer el robot, elegir una nueva
        if self.action != None :
            # 1.2 Si ya no se ocupa llevar cajas a entregar
            if self.action == 'leave_box_deliver' and not self.model.picker.is_active:
                self.action = None
                self.path.clear()
            # 3. Si ya no se ocupa recoger cajas de racks
            elif self.action == 'deliver_box_pick_rack' and not self.model.picker.is_active:
                self.action = None
                self.path.clear()
            # 4. Si ya no se ocupa recoger cajas de banda llegada
            elif self.action == 'pick_box' and not self.model.box_waiting:
                self.action = None
                self.path.clear()
            # 5. Si ya se termino de cargar el robot
            elif self.action == 'cargar' and self.carga >= 100.00:
                self.action = None
                self.path.clear()
                self.isCharging = False
                self.charger = None
                #Hace que el cargador en el que este, tome un estado de que ya no esta ocupado
                agents = self.model.grid.get_cell_list_contents((self.pos[0], self.pos[1]))
                cargador = [agent for agent in agents if isinstance(agent, Charger)]
                cargador[0].busy = False


        if self.isCharging:
            return

        # Determinar que accion debe hacer el robot por importancia
        # 1. Si tiene una caja - dejarla
        #   1.1 Si carga es menor a 30 - dejar caja en rack
        #   1.2 Si hay camion esperando - dejar caja en banda entrega
        #   1.3 De lo contrario - dejar caja en rack
        # 2. Si carga es menor a 30 - cargar
        # 3. Si hay camion esperando y hay cajas en racks - buscar caja en rack y recoger
        # 4. Si hay cajas esperando en banda llegada - recoger
        
        if self.box != None and len(self.path) == 0:
            self.leave_box()
        #Si tiene 50 de bateria o menos y si no tiene camino 
        elif self.carga <= 50 and (len(self.path) == 0 or (not self.isCharging and len(self.path) <= 2)):
            self.charge()
        elif self.model.picker.is_active and len(self.path) == 0 and len(self.model.rack_box) > 0 and self.search_box_deliver():
            pass
        elif self.model.box_waiting and len(self.path) == 0:
            self.search_box_pick()
        else:
            self.sig_pos = self.random.choice(vecinos_disponibles).pos
            self.action == None

        if( self.wait ):
            return


        if len(self.path) > 0: 
            #Si se esta cargado checa si el cargador al que va esta ocupado
            if self.action == 'cargar':
                if len(self.path) <= 2:
                    agents = self.model.grid.get_cell_list_contents((self.path[0][0], self.path[0][1]))
                    cargador = [agent for agent in agents if isinstance(agent, Charger)]
                    #Si esta ocupado saca la posicion del cargador de la lista de camino
                    if cargador[0].busy:
                        self.path.pop(0)  
            #Como se saca un elemento en la condicinal anterior es importante volver a checar si es mayor a 0
            if len(self.path) > 0:
                self.sig_pos = self.path.pop()
            #Revisa si esta en un cargador
            if self.action == 'cargar' and isinstance(self.model.grid.__getitem__((self.sig_pos[0], self.sig_pos[1]))[0], Charger):
                #Si lo esta cambia estado a cargando
                self.isCharging = True
                self.isGoingToCharge = False
                agents = self.model.grid.get_cell_list_contents((self.sig_pos[0], self.sig_pos[1]))
                cargador = [agent for agent in agents if isinstance(agent, Charger)]
                #El cargador cambiaa su estado a ocupado
                cargador[0].busy = True
            
                
        # Si no se eligio una nueva posicion, mover aleatoriamente, sirve cuando se esta en un rack
        #Si se estaba cargando entonces que se quede en la misma posicion
        if self.sig_pos == self.pos and self.action != "cargar": 
            self.seleccionar_nueva_pos(vecinos_disponibles)
            self.action == None
        

        # Manejo de colisiones
        # Basado en preguntar a los demas robots su siguiente movimiento
        # Robot con menor id no cambia su movimiento - los de mayor id cambian y se adaptan
        # Si hay choque - el robot con menor id no cambia su movimiento (no se mueve)
        can_move = True
        for id in range(self.unique_id):
            robot: Robot = self.model.robots[id]
            # Se detecta que no se puede mantener misma ruta o hay choque
            if robot.sig_pos == self.sig_pos:
                can_move = False
                break
        
        if not can_move:
            # Si estaba haciendo una tarea regresar movimiento que tenia pensado hacer
            if self.action != None: self.path.append(self.sig_pos)
            self.sig_pos = self.pos

            # Manejar el caso en el que alguien se quiere mover a donde yo estoy
            is_obstacle = False
            x, y = self.pos
            posible_moves = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
            # Mantener solo movimientos validos
            posible_moves = [pos for pos in posible_moves if self.isValid(pos[0], pos[1])]

            # Checar si alguien se quiere mover a donde yo estoy - si robot es un estorbo
            for id in range(self.unique_id):
                robot: Robot = self.model.robots[id]
                if robot.sig_pos == self.pos: 
                    is_obstacle = True
                if len(posible_moves) > 0 and robot.sig_pos in posible_moves:
                    posible_moves.remove(robot.sig_pos)
            
            # Checar si posiciones a moverse son validas (no hay obstaculos)
            if is_obstacle:
                for pos in posible_moves:
                    if isinstance(self.model.grid.__getitem__(pos), (Rack, Robot, ConveyorBelt, Box)):
                        posible_moves.remove(pos)
        
            # Si es un estorbo y hay movimientos posibles, elegir uno aleatorio
            if len(posible_moves) > 0 and is_obstacle:
                self.sig_pos = self.random.choice(posible_moves)
                # Recalcular path
                if self.action != None:
                    self.path = self.aStar([self.path[0]])
            

    def advance(self):        
        if self.pos != self.sig_pos:
            self.movimientos += 1
            self.carga -= self.model.battery_drain
        
        if self.carga > 0:
            self.model.grid.move_agent(self, self.sig_pos)

    # Regresa cargador en base a posicion dada
    def get_charger_by_pos(self, target_pos):
        for charger in self.model.chargers:
            if charger.pos == target_pos:
                return charger
        return None

    # 1. Si tiene una caja - dejarla
    def leave_box(self):
        def check_rack_leave_box():
        # Determina si esta en rack y deja la caja
            in_rack = False
            rack: Rack = None
            for item in self.model.grid.__getitem__(self.pos):
                if isinstance(item, Rack): 
                    in_rack = True
                    rack = item
            if not in_rack or rack == None: return False

            # Se deja la caja en el rack
            self.box.robot = None
            rack.box = self.box
            self.box.sig_pos = rack.pos # Confirmar que no haya error al dejar cajas en rack
            self.box = None
            # Agregar rack a lista de racks con cajas si no es de pickup o delivery
            if rack.pos != self.model.rack_delivery: self.model.rack_box.append(rack)
            return True

        # 1 Si carga es menor a 30 - dejar caja en rack
        # 2 Si hay camion esperando - dejar caja en banda entrega
        # 3 De lo contrario - dejar caja en rack
        if self.carga <= 50 and len(self.path) == 0:
            # Si no hay path puede ser:
            # - todavia no se define path a rack más cercano
            # - ya se llegó al rack más cercano
            # - que no tome en cuenta el rack de pickup como uno para dejar
            if self.pos != self.model.rack_pickup and check_rack_leave_box(): return
            
            # Si no se ha definido un rack cercano, definirlo
            rack_closest: Rack = self.select_heuristic_rack(empty=True)
            rack_closest.box = self.box
            # Obtiene el path para llegar a un rack
            self.path = self.aStar([rack_closest.pos])
            self.action = 'leave_box_rack'
        elif self.model.picker.is_active and len(self.path) == 0 and self.action == 'deliver_box_pick_rack':
            # que no tome en cuenta el rack de pickup como uno para dejar
            if self.pos != self.model.rack_pickup and check_rack_leave_box(): return
            
            self.path = self.aStar([self.model.rack_delivery])
            self.action = 'leave_box_deliver'
        elif len(self.path) == 0:
            # que no tome en cuenta el rack de pickup como uno para dejar
            if self.pos != self.model.rack_pickup and check_rack_leave_box(): return

            rack_closest: Rack = self.select_heuristic_rack(empty=True)
            rack_closest.box = self.box
            self.path = self.aStar([rack_closest.pos])
            self.action = 'leave_box_rack'

    # 2. Si carga es menor a 50 - cargar
    def charge(self):
        charger_pos = [charger.pos for charger in self.model.chargers]

        if(len(self.path) < 3 and self.isGoingToCharge):

            if(len(self.path) == 0):
                self.path = [self.charger]
                
            if(self.path[0] not in charger_pos):
                self.path = self.aStar(charger_pos)
                  
            charger = self.get_charger_by_pos((self.path[0][0],self.path[0][1]))

            if(charger.busy):
                chargers = []

                for pos in charger_pos:
                    if pos[1] == self.path[0][1] and pos != self.path[0]:
                        charger = self.get_charger_by_pos(pos)
                        if(not charger.busy):
                            chargers.append(pos)

                if(len(chargers)>0):       
                    self.path = self.aStar(chargers)
                    self.charger = self.path[0]
                    self.wait = False
                    return

                for poss in charger_pos:
                    charger = self.get_charger_by_pos(poss)
                    if poss[1] != pos[1] and not charger.busy:
                        chargers.append(poss)
                if(len(chargers)>0):       
                    self.path = self.aStar(chargers)
                    self.charger = self.path[0]
                    self.wait = False
                    return
                else:
                    self.wait = True
                    return
                                
            else:
                self.wait = False

            return
                           
        self.path = self.aStar(charger_pos)
        self.action = 'cargar'
        self.isGoingToCharge = True
        if(len(self.path) > 0):
            self.charger = self.path[0]

    # 3. Si hay camion esperando y hay cajas en racks - buscar caja en rack y recoger
    def search_box_deliver(self):

        def check_rack_pickup_box():
        # Determina si esta en rack y deja la caja
            in_rack = False
            rack: Rack = None
            for item in self.model.grid.__getitem__(self.pos):
                if isinstance(item, Rack): 
                    in_rack = True
                    rack = item
            if not in_rack or rack == None: return False

            # Si estoy en rack - tomar la caja
            self.box = rack.box
            rack.box = None
            self.box.robot = self
            return True

        # Si no hay racks con cajas y que haya camion esperando cajas
        if len(self.model.rack_box) == 0: return False

        # Checar que no sea de rack de pickup o de deilvery 
        if self.pos != self.model.rack_pickup and check_rack_pickup_box(): return False
        elif self.model.picker.orders <= 0: return False

        self.model.picker.orders -= 1
        rack_closest: Rack = self.select_heuristic_rack(empty=False)
        self.model.rack_box.remove(rack_closest) # Eliminar rack de lista de racks con cajas
        self.path = self.aStar([rack_closest.pos])
        self.action = 'deliver_box_pick_rack'

    # 4. Si hay cajas esperando en banda llegada - recoger
    def search_box_pick(self):
        def check_rack_pick_box():
        # Determina si esta en rack y deja la caja
            in_rack = False
            box: Box = None
            for item in self.model.grid.__getitem__(self.pos):
                if isinstance(item, Rack): in_rack = True
                if isinstance(item, Box): box = item
            if not in_rack or box == None: return False
            # Se agarra la caja si la caja aun no ha sido agarrado por otro robot
            if box.robot != None: return True

            self.box = box
            box.robot = self
            return True
        
        # Si hay cajas esperando en llegada - recoger
        if len(self.path) == 0:
            # Checar si estoy en rack de pickup
            if check_rack_pick_box(): return

            # Ruta hacia rack de pickup
            self.action = 'pick_box'
            self.path = self.aStar([self.model.rack_pickup])

    def aStar(self, dest):
        start = self.pos
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
        
        # Caso en el que no hay ruta para alguno de los destinos
        for p in path: 
            if p == None: return []
        
        return min(path, key=len)

    def aStarHelper(self, cell_details, open_list, destination_cells, visited_nodes, i):
        import heapq
        path = list()
        
        # 4 movimientos posibles
        moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
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
                # Determinar si en posición a moverse hay un obstaculo
                obstacle = False
                for item in self.model.grid.__getitem__((new_x, new_y)):
                    if isinstance(item, (Robot, Rack, ConveyorBelt)): obstacle = True
                
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
                elif not obstacle and (new_x, new_y) not in visited_nodes:
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

    def select_heuristic_rack(self, empty: bool)-> Rack:
        # Usa una heuristica para determinar el rack más cercano que no esté ocupado
        rack_min_dist: Rack = None
        min_dist = self.FLT_MAX

        def dist(pos1, pos2):
            import math
            return math.sqrt((pos1[0] - pos2[0]) ** 2 + 8 * (pos1[1] - pos2[1]) ** 2)
        
        racks: list[Rack] = []
        # Rack con cajas o rack vacio
        # - True: rack sin cajas
        # - False: rack con cajas
        if empty: racks = self.model.racks
        else: racks = self.model.rack_box

        for rack in racks:
            # Si rack esta desocupado, no es el de pickup o delivery no tomar en cuenta
            if (empty and rack.box != None or rack.pos == self.model.rack_pickup or rack.pos == self.model.rack_delivery) or (not empty and rack.box == None):
                continue
            curr_dist = dist(self.pos, rack.pos)
            if curr_dist < min_dist:
                min_dist = curr_dist
                rack_min_dist = rack
        return rack_min_dist


class Bodega(Model):
    def __init__(self, M: int = 50, N: int = 25,
                 num_robots: int = 5,
                 modo_pos_inicial: str = 'Aleatoria',
                 speed_box_arrival: int = 12,
                 battery_drain: float = 0.2
                 ):
        # Listas de agentes
        self.robots: list[Robot] = []
        self.chargers: list[Charger] = []
        self.racks: list[Rack] = []
        self.boxes: list[Box] = []
        self.conveyor_belts: list[ConveyorBelt] = []
        self.picker: Picker = None

        self.num_chargers = 4
        self.num_robots = num_robots
        self.battery_drain = battery_drain

        self.grid = MultiGrid(M, N, False)
        self.schedule = SimultaneousActivation(self)

        belt_size = 20
        self.box_waiting = False
        self.rack_pickup = (0, belt_size)
        self.rack_delivery = (M - 1, belt_size)
        self.rack_box: list[Rack] = []

        self.colisiones = 0
        self.curr_step = 0


        posiciones_disponibles = [pos for _, pos in self.grid.coord_iter()]

        # Posicionamiento de cintas
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
        self.picker = picker

        # Posicionamiento de racks
        positions_racks = list()
        for y in [3, 4, 8, 9, 13, 14, 18, 19]:
            positions_racks += [(x, y) for x in list(range(3, 11)) + list(range(14, 22)) + list(range(25, 33)) + list(range(37, 45))]
        positions_racks.append(self.rack_pickup)
        positions_racks.append(self.rack_delivery)

        for id, pos in enumerate(positions_racks):
            mueble = Rack(70 + id, self)
            self.grid.place_agent(mueble, pos)
            posiciones_disponibles.remove(pos)
            if pos == (0, belt_size) or pos == (M - 1, belt_size):
                self.schedule.add(mueble)
            self.racks.append(mueble)

        # Posicionamiento de cargadores
        # Cargadores en mitad superior y inferior
        if (self.num_robots <= 6):
            self.positions_chargers = [(int(M / 2) + i*2, j) for i in [-1, 1] for j in [0, N - 1]]
        else:
            self.positions_chargers = [(int(M / 2) + i*2, j) for i in range(-3, 5, 2) for j in [0, N - 1]]
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
        self.curr_step += 1

        robots_pos = set()
        print("Estados espera")
        for robot in self.robots:
            robots_pos.add(robot.pos)
            print(f"Espera: ",robot.wait, "Esta cargando: ", robot.isCharging, "Esta llendo a cargarse: ", robot.isGoingToCharge)
            
        if len(self.robots) != len(robots_pos):
            self.colisiones += 1
            print(f"Colisiones: {len(self.robots) - len(robots_pos)}, \t Total: {self.colisiones}")

        # Error, racks quedan marcados como ocupados sin cajas
        # Se debe a camio de prioridad de robots
        if self.curr_step % 500 == 0:
            for rack in self.racks:
                has_box = False
                for item in self.grid.__getitem__(rack.pos):
                    if isinstance(item, Box): has_box = True
                if not has_box and rack.box != None: rack.box = None


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
