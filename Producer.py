from confluent_kafka import Producer, KafkaError
from bodega.model import Bodega
import json
import time

if __name__ == '__main__':  
    topic = "SistemasMultiagentes"
    conf = {  
        'bootstrap.servers': "cell-1.streaming.mx-queretaro-1.oci.oraclecloud.com",
        'security.protocol': 'SASL_SSL',
        'ssl.ca.location': 'cert/cacert.pem',
        'sasl.mechanism': 'PLAIN',
        'sasl.username': 'rodrigotots927/rodrigotots927@gmail.com/ocid1.streampool.oc1.mx-queretaro-1.amaaaaaafzzzlhqa7jxd2lhgxboycmw2pbompiddu33tyj6m336h3bhkv33q',
        'sasl.password': 'W_f)4J]TeZ;VgizbUqA5',
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

    for n in range(200):
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
        time.sleep(0.5)
        model.step()

producer.flush()  
print("{} messages were produced to topic {}!".format(delivered_records, topic))