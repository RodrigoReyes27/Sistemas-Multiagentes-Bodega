import mesa
import math as mt
from .model import Bodega, Robot, Rack, ConveyorBelt, Box, Charger

MAX_NUMBER_ROBOTS = 10

def agent_portrayal(agent):
    if isinstance(agent, Robot):
        return {"Shape": "circle", "Filled": "false", "Color": "black", "Layer": 1, "r": 1.0,
                "text": f"{mt.ceil(agent.carga)}", "text_color": "yellow"}
    elif isinstance(agent, Rack):
        return {"Shape": "rect", "Filled": "true", "Color": "white", "Layer": 0,
                "w": 0.9, "h": 0.9, "text_color": "Black", "text": "🪑"}
    elif isinstance(agent, Box):
        return {"Shape": "rect", "Filled": "true", "Color": "white", "Layer": 0,
                "w": 0.9, "h": 0.9, "text_color": "Black", "text": "📦"}
    elif isinstance(agent, Charger):
        return {"Shape": "rect", "Filled": "true", "Color": "yellow", "Layer": 0, 
                "w": 0.9, "h": 0.9, "text_color": "Black", "text": "🔋"}
    elif isinstance(agent, ConveyorBelt):
        return {"Shape": "rect", "Filled": "true", "Color": "yellow", "Layer": 0, 
                "w": 0.9, "h": 0.9, "text_color": "Black", "text": "🌫"}

grid = mesa.visualization.CanvasGrid(
    agent_portrayal, 50, 25, 600, 600)

chart_deliveries = mesa.visualization.ChartModule(
    [{"Label": "Entregas", "Color": '#36A2EB', "label": "Cajas entregadas"}],
    50, 200,
    data_collector_name="datacollector"
)

chart_movements = mesa.visualization.ChartModule(
    [{"Label": "Movimientos", "Color": '#36A2EB', "label": "Movimientos"}],
    50, 200,
    data_collector_name="datacollector"
)

chart_battery_used = mesa.visualization.ChartModule(
    [{"Label": "Pila", "Color": '#36A2EB', "label": "Pila usada"}],
    50, 200,
    data_collector_name="datacollector"
)

model_params = {
    "num_robots": mesa.visualization.Slider(
        "Número de Robots",
        7,
        1,
        MAX_NUMBER_ROBOTS,
        1,
        description="Escoge cuántos robots deseas implementar en el modelo",
    ),
    "M": 50,
    "N": 25,
}

server = mesa.visualization.ModularServer(
    Bodega, [grid, chart_deliveries, chart_movements, chart_battery_used],
    "Almacén Lobsters", model_params, 8521
)
