#!/usr/bin/env python3
"""
Agricultural IoT Sensor Simulator

A comprehensive simulator for generating realistic agricultural sensor data
with multiple transmission methods and advanced simulation features.

Inspired by the research paper "Distributed Agricultural Monitoring Using BigchainDB"
by Željko Džafić et al.
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import threading
import uuid
import math
from datetime import datetime
from typing import Dict, List, Any, Optional

import requests
import paho.mqtt.client as mqtt
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agricultural_simulator.log')
    ]
)
logger = logging.getLogger("agri-iot-simulator")

class Sensor:
    """Base class for agricultural IoT sensors"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize a sensor with the given configuration
        
        Args:
            config: Sensor configuration dictionary
        """
        self.id = config.get('id', str(uuid.uuid4()))
        self.type = config.get('type', 'generic')
        self.field_id = config.get('field_id', 'field-1')
        self.name = config.get('name', f"{self.type.capitalize()} Sensor")
        self.min_value = config.get('min_value', 0)
        self.max_value = config.get('max_value', 100)
        self.interval = config.get('interval', 300)  # seconds
        self.unit = config.get('unit', 'units')
        self.noise = config.get('noise', 0.1)
        
        # Advanced variation settings
        self.daily_variation = config.get('daily_variation', False)
        self.seasonal_variation = config.get('seasonal_variation', False)
        
        # State tracking
        self.last_reading = None
        self.last_reading_time = None
        self.next_reading_time = time.time()
    
    def generate_value(self, timestamp: datetime) -> float:
        """
        Generate a sensor reading with realistic variations
        
        Args:
            timestamp: Timestamp for the reading
        
        Returns:
            Simulated sensor value
        """
        base = (self.max_value + self.min_value) / 2
        amplitude = (self.max_value - self.min_value) / 2
        noise = random.uniform(-self.noise, self.noise) * amplitude

        # Daily variation (sine wave)
        if self.daily_variation:
            seconds = timestamp.timestamp()
            day_phase = math.sin(2 * math.pi * (seconds % 86400) / 86400)
            base += day_phase * amplitude * 0.5

        # Seasonal variation (cosine wave)
        if self.seasonal_variation:
            doy = timestamp.timetuple().tm_yday
            season_phase = math.cos(2 * math.pi * doy / 365)
            base += season_phase * amplitude * 0.3

        return max(self.min_value, min(self.max_value, base + noise))
    
    def read(self) -> Dict[str, Any]:
        """
        Take a reading from the sensor
        
        Returns:
            A dictionary with the reading data
        """
        # Check if it's time for a new reading
        current_time = time.time()
        if current_time < self.next_reading_time:
            return None
        
        # Update next reading time
        self.next_reading_time = current_time + self.interval
        
        # Get the reading
        now = datetime.utcnow()
        value = self.generate_value(now)
        
        # Update last reading
        self.last_reading = value
        self.last_reading_time = current_time
        
        # Create reading object
        reading = {
            "id": str(uuid.uuid4()),
            "sensor_id": self.id,
            "field_id": self.field_id,
            "name": self.name,
            "value": round(value, 2),
            "unit": self.unit,
            "type": self.type,
            "timestamp": now.isoformat() + 'Z'
        }
        
        return reading


class SoilMoistureSensor(Sensor):
    """Soil moisture sensor with advanced drying and rainfall simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.current_value = random.uniform(self.min_value, self.max_value)
        self.drying_rate = config.get('drying_rate', 0.05)
        self.rainfall_events = config.get('rainfall_events', True)
    
    def generate_value(self, timestamp: datetime) -> float:
        """
        Generate a soil moisture reading with drying and rainfall events
        
        Args:
            timestamp: Timestamp for the reading
        
        Returns:
            Simulated soil moisture value
        """
        # Dry out slowly over time
        self.current_value -= self.drying_rate

        # Rainfall event
        if self.rainfall_events and random.random() < 0.01:
            self.current_value += random.uniform(5, 20)

        # Add noise
        self.current_value += random.uniform(-self.noise, self.noise) * (self.max_value - self.min_value)
        
        # Constrain to valid range
        self.current_value = max(self.min_value, min(self.max_value, self.current_value))
        return self.current_value


class HumiditySensor(Sensor):
    """Humidity sensor with inverse temperature correlation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.inverse_temp_correlation = config.get('inverse_temp_correlation', True)
    
    def generate_value(self, timestamp: datetime) -> float:
        """
        Generate a humidity reading with inverse temperature correlation
        
        Args:
            timestamp: Timestamp for the reading
        
        Returns:
            Simulated humidity value
        """
        value = super().generate_value(timestamp)
        
        if self.inverse_temp_correlation:
            hour = timestamp.hour
            # Adjust humidity based on time of day (less humidity during hot hours)
            value *= 1.2 - (hour / 24)
        
        return max(self.min_value, min(self.max_value, value))


class LightSensor(Sensor):
    """Light sensor with daily variation and cloud effects"""
    
    def generate_value(self, timestamp: datetime) -> float:
        """
        Generate a light reading with day/night cycle and cloud effects
        
        Args:
            timestamp: Timestamp for the reading
        
        Returns:
            Simulated light intensity value
        """
        seconds = timestamp.timestamp()
        hour = timestamp.hour
        
        # Day/night cycle for light sensors
        if 6 <= hour <= 18:  # Daytime
            day_phase = math.sin((hour - 6) * (math.pi / 12))  # Peaks at noon
            base_value = self.min_value + day_phase * (self.max_value - self.min_value)
            
            # Cloud cover effect
            if random.random() < 0.3:  # 30% chance of clouds
                cloud_factor = random.uniform(0.3, 0.9)  # 30-90% reduction
                base_value *= cloud_factor
        else:
            base_value = self.min_value  # Minimal light at night
        
        # Add noise
        base_value += random.uniform(-self.noise, self.noise) * (self.max_value - self.min_value)
        
        return max(self.min_value, min(self.max_value, base_value))


class DataTransmitter:
    """Base class for data transmission methods"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize a data transmitter with the given configuration
        
        Args:
            config: Transmission configuration dictionary
        """
        self.config = config
        self.network = config.get('network', {})
        self.packet_loss = self.network.get('packet_loss', 0)
        self.latency_mean = self.network.get('latency_mean', 0)
        self.latency_stddev = self.network.get('latency_stddev', 0)
    
    def send(self, data: Dict[str, Any]) -> bool:
        """
        Send data using the configured transmission method.
        
        Args:
            data: The data to send
            
        Returns:
            True if transmission was successful, False otherwise
        """
        # Simulate network conditions
        if random.random() < self.packet_loss:
            logger.warning("Simulated packet loss - data not sent")
            return False
        
        # Simulate network latency
        if self.latency_mean > 0:
            latency = max(0, random.normalvariate(self.latency_mean, self.latency_stddev))
            time.sleep(latency / 1000)  # Convert to seconds
        
        return True


class HttpTransmitter(DataTransmitter):
    """Transmitter that sends data using HTTP POST requests"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get('endpoint', 'http://localhost:8000/measurements/')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.retry_count = config.get('retry_count', 3)
        self.timeout = config.get('timeout', 5)
        
        # Authentication
        self.auth = None
        auth_config = config.get('authentication', {})
        auth_type = auth_config.get('type', 'none')
        
        if auth_type == 'basic':
            username = auth_config.get('username', '')
            password = auth_config.get('password', '')
            self.auth = (username, password)
        elif auth_type == 'bearer':
            token = auth_config.get('token', '')
            self.headers['Authorization'] = f"Bearer {token}"
    
    def send(self, data: Dict[str, Any]) -> bool:
        """Send data using HTTP POST"""
        # First check network conditions
        if not super().send(data):
            return False
        
        # Send the data
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.endpoint,
                    json=data,
                    headers=self.headers,
                    auth=self.auth,
                    timeout=self.timeout
                )
                
                if response.status_code < 300:
                    logger.info(f"Data sent successfully to {self.endpoint}")
                    return True
                else:
                    logger.warning(f"HTTP error: {response.status_code} - {response.text}")
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{self.retry_count}): {e}")
            
            # Wait before retrying with exponential backoff
            time.sleep(2 ** attempt)
        
        return False


class SensorSimulator:
    """Main simulator class to manage sensor simulation and data transmission"""
    
    def __init__(self, config_path: str):
        """
        Initialize the simulator with a configuration file
        
        Args:
            config_path: Path to the JSON configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create sensors
        self.sensors = self._create_sensors()
        
        # Create transmitter
        transmission_config = self.config.get('transmission', {})
        method = transmission_config.get('method', 'http')
        
        if method == 'http':
            self.transmitter = HttpTransmitter(transmission_config)
        else:
            logger.warning(f"Unsupported transmission method: {method}. Defaulting to HTTP.")
            self.transmitter = HttpTransmitter(transmission_config)
    
    def _create_sensors(self) -> List[Sensor]:
        """
        Create sensor instances based on configuration
        
        Returns:
            List of sensor objects
        """
        sensors = []
        sensor_configs = self.config.get('sensors', [])
        
        # Sensor type mapping
        sensor_classes = {
            'temperature': Sensor,
            'soil_moisture': SoilMoistureSensor,
            'humidity': HumiditySensor,
            'light': LightSensor
        }
        
        for sensor_config in sensor_configs:
            sensor_type = sensor_config.get('type', 'generic')
            sensor_class = sensor_classes.get(sensor_type, Sensor)
            sensors.append(sensor_class(sensor_config))
        
        return sensors
    
    def run(self, duration: int = 0, output_file: Optional[str] = None, verbose: bool = False):
        """
        Run the sensor simulation
        
        Args:
            duration: Simulation duration in seconds (0 for indefinite)
            output_file: Optional file to log sensor readings
            verbose: Enable verbose logging
        """
        start_time = time.time()
        
        while True:
            # Check simulation duration
            if duration and time.time() - start_time > duration:
                logger.info("Simulation duration reached. Stopping.")
                break
            
            # Collect and transmit readings
            for sensor in self.sensors:
                try:
                    reading = sensor.read()
                    
                    if reading:
                        # Log to file if specified
                        if output_file:
                            with open(output_file, 'a') as f:
                                f.write(json.dumps(reading) + '\n')
                        
                        # Transmit data
                        self.transmitter.send(reading)
                        
                        # Optional verbose logging
                        if verbose:
                            logger.info(f"Sensor {reading['sensor_id']} reading: {reading}")
                
                except Exception as e:
                    logger.error(f"Error processing sensor {sensor.id}: {e}")
            
            # Wait to respect sensor intervals
            time.sleep(1)


def main():
    """Main entry point for the Agricultural IoT Simulator"""
    parser = argparse.ArgumentParser(description='Agricultural IoT Sensor Simulator')
    parser.add_argument('--config', type=str, default='config.json', 
                        help='Path to configuration file')
    parser.add_argument('--duration', type=int, default=0,
                        help='Simulation duration in seconds (0 for indefinite)')
    parser.add_argument('--output', type=str, 
                        help='Output file for data logging')
    parser.add_argument('--api-url', type=str, 
                        help='Override HTTP API endpoint URL')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create default configuration if not exists
    if not os.path.exists(args.config):
        default_config = {
            "general": {
                "simulation_name": "farm-simulator",
                "location": {
                    "latitude": 45.2671,
                    "longitude": 19.8335
                }
            },
            "sensors": [
                {
                    "id": "temp-1",
                    "type": "temperature",
                    "field_id": "field-1",
                    "name": "Temperature Sensor",
                    "min_value": -10,
                    "max_value": 40,
                    "interval": 300,
                    "unit": "celsius",
                    "noise": 0.2,
                    "daily_variation": True,
                    "seasonal_variation": True
                },
                {
                    "id": "moisture-1",
                    "type": "soil_moisture",
                    "field_id": "field-1",
                    "name": "Soil Moisture Sensor",
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
                "network": {
                    "packet_loss": 0.01,
                    "latency_mean": 100,
                    "latency_stddev": 30
                }
            }
        }
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        
        # Write default configuration
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default configuration at {args.config}")
    
    # Override API URL if provided
    try:
        simulator = SensorSimulator(args.config)
        
        # Update transmission endpoint if API URL is provided
        if args.api_url:
            simulator.transmitter.endpoint = args.api_url
        
        # Run simulation
        logger.info("Starting Agricultural IoT Sensor Simulation")
        simulator.run(
            duration=args.duration, 
            output_file=args.output, 
            verbose=args.verbose
        )
    
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)