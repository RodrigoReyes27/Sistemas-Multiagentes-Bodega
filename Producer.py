from confluent_kafka import Producer, KafkaError
from bodega.model import Bodega
from dotenv import load_dotenv
import json
import time
import os

load_dotenv()

if __name__ == '__main__':  
    topic = os.getenv("TOPIC")
    conf = {  
        'bootstrap.servers': os.getenv("BOOTSTRAP_SERVERS"),
        'security.protocol': os.getenv("SECURITY_PROTOCOL"),
        'ssl.ca.location': os.getenv("SSL_CA_LOCATION"),
        'sasl.mechanism': os.getenv("SASL_MECHANISM"),
        'sasl.username': os.getenv("SASL_USERNAME"),
        'sasl.password': os.getenv("SASL_PASSWORD"),
    }
    
    # Create Producer instance  
    producer = Producer(**conf)  
    delivered_records = 0

    model = Bodega()
    
    def acked(err, msg):  
        global delivered_records  
        """Delivery report handler called on  
            successful or failed delivery of message """  
        if err is not None:  
            print("Failed to deliver message: {}".format(err))  
        else:  
            delivered_records += 1  
            print("Produced record to topic {} partition [{}] @ offset {}".format(msg.topic(), msg.partition(), msg.offset()))

    for n in range(1500):
        record_key = str(n)
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
        
        producer.produce(topic, key=record_key, value=json.dumps(message, indent=2), on_delivery=acked)
        print(json.dumps(message, indent=2))
        
        producer.poll(0)
        time.sleep(1)
        model.step()

producer.flush()  
print("{} messages were produced to topic {}!".format(delivered_records, topic))