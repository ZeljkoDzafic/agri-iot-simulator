# Agricultural IoT Simulator

A simple Python-based simulator for agricultural IoT sensors that generates and sends realistic sensor data to a backend API. This tool is designed for testing and demonstrating distributed agricultural monitoring systems as described in the paper "Distributed Agricultural Monitoring Using BigchainDB" by Željko Džafić et al.

## Features

- Simulates multiple agricultural sensor types:
  - Soil moisture sensors
  - Temperature sensors
  - Humidity sensors
  - Light sensors
- Generates realistic data patterns with configurable parameters
- Simulates daily and seasonal variations
- Configurable reporting intervals
- Supports HTTP and MQTT protocols
- Simulates network conditions (packet loss, latency)
- Optional data signing with Ed25519

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/ZeljkoDzafic/agri-iot-simulator.git
cd agri-iot-simulator
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the simulator with default settings:

```bash
python simulator.py
```

### Configuration

Create a custom configuration file:

```bash
cp config/default.json config/my_config.json
```

Edit `my_config.json` to adjust:
- Sensor types and quantities
- Reading frequencies
- Value ranges
- Transmission methods
- Network conditions

Run with your custom configuration:

```bash
python simulator.py --config config/my_config.json
```

### Command-line Options

```
Usage: python simulator.py [OPTIONS]

Options:
  --config TEXT           Path to configuration file
  --duration INTEGER      Simulation duration in seconds (0 for indefinite)
  --output TEXT           Output file for data logging
  --api-url TEXT          Backend API URL
  --mqtt-broker TEXT      MQTT broker address
  --verbose               Enable verbose output
  --help                  Show this message and exit
```

## Configuration Options

### Sample Configuration File

```json
{
  "general": {
    "simulation_name": "farm-123",
    "location": {
      "latitude": 45.2671,
      "longitude": 19.8335
    },
    "log_level": "info"
  },
  "sensors": [
    {
      "id": "temp-1",
      "type": "temperature",
      "field_id": "field-1",
      "name": "Temperature Sensor 1",
      "min_value": -10,
      "max_value": 40,
      "interval": 300,
      "unit": "celsius",
      "noise": 0.2,
      "daily_variation": true,
      "seasonal_variation": true
    },
    {
      "id": "moisture-1",
      "type": "soil_moisture",
      "field_id": "field-1",
      "name": "Soil Moisture Sensor 1",
      "min_value": 0,
      "max_value": 100,
      "interval": 600,
      "unit": "percent",
      "noise": 0.5,
      "drying_rate": 0.05
    }
  ],
  "transmission": {
    "method": "http",
    "endpoint": "http://localhost:8000/measurements/",
    "headers": {
      "Content-Type": "application/json"
    },
    "authentication": {
      "type": "none"
    },
    "network": {
      "packet_loss": 0.01,
      "latency_mean": 100,
      "latency_stddev": 30
    }
  }
}
```

### Sensor Types and Parameters

#### Temperature Sensor

```json
{
  "id": "temp-1",
  "type": "temperature",
  "field_id": "field-1",
  "name": "Temperature Sensor 1",
  "min_value": -10,
  "max_value": 40,
  "interval": 300,
  "unit": "celsius",
  "noise": 0.2,
  "daily_variation": true,
  "seasonal_variation": true
}
```

#### Soil Moisture Sensor

```json
{
  "id": "moisture-1",
  "type": "soil_moisture",
  "field_id": "field-1",
  "name": "Soil Moisture Sensor 1",
  "min_value": 0,
  "max_value": 100,
  "interval": 600,
  "unit": "percent",
  "noise": 0.5,
  "drying_rate": 0.05,
  "rainfall_events": true
}
```

#### Humidity Sensor

```json
{
  "id": "humidity-1",
  "type": "humidity",
  "field_id": "field-1",
  "name": "Humidity Sensor 1",
  "min_value": 20,
  "max_value": 95,
  "interval": 300,
  "unit": "percent",
  "noise": 0.3,
  "daily_variation": true,
  "inverse_temp_correlation": true
}
```

#### Light Sensor

```json
{
  "id": "light-1",
  "type": "light",
  "field_id": "field-1",
  "name": "Light Sensor 1",
  "min_value": 0,
  "max_value": 100000,
  "interval": 300,
  "unit": "lux",
  "noise": 0.1,
  "daily_variation": true
}
```

### Transmission Methods

#### HTTP

```json
{
  "method": "http",
  "endpoint": "http://localhost:8000/measurements/",
  "headers": {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
  },
  "retry_count": 3,
  "timeout": 5
}
```

#### MQTT

```json
{
  "method": "mqtt",
  "broker": "localhost",
  "port": 1883,
  "topic_prefix": "farm/sensors",
  "client_id": "farm-123-simulator",
  "qos": 1,
  "username": "",
  "password": ""
}
```

#### Direct to BigchainDB

```json
{
  "method": "bigchaindb",
  "url": "http://localhost:9984",
  "public_key": "YOUR_PUBLIC_KEY",
  "private_key": "YOUR_PRIVATE_KEY"
}
```

## Examples

### Simulate a single temperature sensor

```bash
python simulator.py --config config/single_temp_sensor.json
```

### Simulate multiple sensors with MQTT transmission

```bash
python simulator.py --config config/mqtt_farm.json --mqtt-broker localhost
```

### Generate a CSV file without transmitting data

```bash
python simulator.py --config config/offline_mode.json --output farm_data.csv
```

## Simulating Weather Events

The simulator can model the effects of various weather events on sensor readings:

### Rainfall Event

```bash
python simulator.py --trigger-event rainfall --duration 120
```

### Temperature Spike

```bash
python simulator.py --trigger-event heatwave --duration 240
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This implementation is based on the research paper "Distributed Agricultural Monitoring Using BigchainDB"
