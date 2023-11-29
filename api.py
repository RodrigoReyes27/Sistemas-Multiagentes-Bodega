from confluent_kafka import Producer
from flask import Flask, request, jsonify
from bodega.model import Bodega
from dotenv import load_dotenv
import json
import threading
import time
import os

load_dotenv()

app = Flask(__name__)
topic = os.getenv("TOPIC")
conf = {  
    'bootstrap.servers': os.getenv("BOOTSTRAP_SERVERS"),
    'security.protocol': os.getenv("SECURITY_PROTOCOL"),
    'ssl.ca.location': os.getenv("SSL_CA_LOCATION"),
    'sasl.mechanism': os.getenv("SASL_MECHANISM"),
    'sasl.username': os.getenv("SASL_USERNAME"),
    'sasl.password': os.getenv("SASL_PASSWORD"),
}

producer = Producer(**conf)
delivered_records = 0
is_running = False
model: Bodega = None
params = {
    "num_robots": 7,
    "speed_box_arrival": 12,
    "battery_drain": 0.2
}

def acked(err, msg):
    global delivered_records
    """Delivery report handler called on
        successful or failed delivery of message"""
    if err is not None:
        print("Failed to deliver message: {}".format(err))
    else:
        delivered_records += 1
        print("Produced record to topic {} partition [{}] @ offset {}".format(msg.topic(), msg.partition(), msg.offset()))

def get_message_data():
    agents = model.robots
    chargers = model.chargers
    racks = model.racks
    boxes = model.boxes
    conveyor_belts = model.conveyor_belts

    robots = {
        "quantity": len(agents),
        "positions": [{"id": agent.unique_id, "x": agent.pos[0], "y": agent.pos[1]} for agent in agents]
    }

    chargers = {
        "quantity": len(chargers),
        "positions": [{"id": charger.unique_id, "x": charger.pos[0], "y": charger.pos[1]} for charger in chargers]
    }

    racks = {
        "quantity": len(racks),
        "positions": [{"id": rack.unique_id, "x": rack.pos[0], "y": rack.pos[1]} for rack in racks]
    }

    boxes = {
        "quantity": len(boxes),
        "positions": [{"id": box.unique_id, "x": box.pos[0], "y": box.pos[1]} for box in boxes]
    }

    conveyor_belts = {
        "quantity": len(conveyor_belts),
        "positions": [{"id": conveyor_belt.unique_id, "x": conveyor_belt.pos[0], "y": conveyor_belt.pos[1]} for conveyor_belt in conveyor_belts]
    }

    message = {
        "robots": robots,
        "chargers": chargers,
        "racks": racks,
        "boxes": boxes,
        "conveyor_belts": conveyor_belts
    }

    return message

def produce_messages():
    global is_running
    global delivered_records
    if delivered_records == 2500:
        is_running = False
        return
    while is_running:
        record_key = str(delivered_records)

        message = get_message_data()

        producer.produce(topic, key=record_key, value=json.dumps(message, indent=2), on_delivery=acked)
        producer.poll(0)
        time.sleep(0.4)
        model.step()

@app.route('/start', methods=['GET'])
def start_service():
    global is_running, model, params
    if not is_running:
        is_running = True
        model = Bodega(
            num_robots=params["num_robots"],
            speed_box_arrival=params["speed_box_arrival"],
            battery_drain=params["battery_drain"]
        )  # Crear una nueva instancia al iniciar el servicio

        # Enviar datos iniciales de la bodega, para que el frontend pueda renderizar
        message = get_message_data()
        producer.produce(topic, key=str(0), value=json.dumps(message, indent=2), on_delivery=acked)
        producer.poll(0)
        time.sleep(10)
        model.step()

        threading.Thread(target=produce_messages).start()
    return jsonify({"status": "Service started"})


@app.route('/reset', methods=['GET'])
def reset_service():
    global is_running, model, delivered_records, params
    delivered_records = 0
    is_running = False
    time.sleep(1)  # Asegurar que el hilo de producción haya terminado
    model = Bodega(
            num_robots=params["num_robots"],
            speed_box_arrival=params["speed_box_arrival"],
            battery_drain=params["battery_drain"]
        )  # Crear una nueva instancia al iniciar el servicio
    
    start_service()
    return jsonify({"status": "Service reset"})


@app.route('/stop', methods=['GET'])
def stop_service():
    global is_running
    is_running = False
    return jsonify({"status": "Service stopped"})

# Example: http://127.0.0.1:5000/params?num_robots=10&speed_box_arrival=10&battery_drain=0.1
@app.route('/params', methods=['GET'])
def change_params():
    global params
    params["num_robots"] = int(request.args.get("num_robots", params["num_robots"]))
    params["speed_box_arrival"] = int(request.args.get("speed_box_arrival", params["speed_box_arrival"]))
    params["battery_drain"] = float(request.args.get("battery_drain", params["battery_drain"]))
    return jsonify({"status": "Params changed"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
