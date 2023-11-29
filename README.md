# Sistema Multiagente
Este sistema multiagente modela una bodega

## Requisitos

- Python 3.x
- Paquetes Python: Flask, mesa, python-dotenv, numpy, confluent_kafka

## Configuración

1. Clona o descarga el repositorio.

2. Instala las dependencias usando pip:

```bash
pip install -r requirements.txt
```


3. Configura la conexión a la base de datos en el archivo `.env`. Reemplaza los valores con tus  datos
```env
BOOTSTRAP_SERVERS=
SECURITY_PROTOCOL=
SSL_CA_LOCATION=
SASL_MECHANISM=
SASL_USERNAME=
SASL_PASSWORD=
TOPIC=
```

## Ejecución

Para iniciar la API, ejecuta el siguiente comando:

```bash
python api.py
```

Iniciar el envío de datos de usando la API
http://127.0.0.1:5000/start
http://127.0.0.1:5000/stop
http://127.0.0.1:5000/restart
[http://127.0.0.1](http://127.0.0.1:5000/params?num_robots=10&speed_box_arrival=10&battery_drain=0.1)

Para iniciar la simulación en Mesa, ejecuta el siguiente comando:

```bash
python run.py
```

Para iniciar generar datos en Kafka, ejecuta el siguiente comando:

```bash
python Producer.py
```


## Equipo 1 - Langostas

- Rodrigo Reyes Gómez      A01284917
- Diego García Minjares    A01284650
- Daniel Eugenio Morales   A01284684
- Jair Santos Gutiérrez    A01026654
- Kraken Domínguez         A00833278
