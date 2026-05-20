import pandas as pd
import random
import time
from datetime import datetime

# CSV file path
FILE_NAME = "sensor_data.csv"

# Function to generate sensor data
def generate_sensor_data():

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "soil_moisture": random.choice([
            random.randint(20, 30),   # Dry condition
            random.randint(31, 60),   # Normal condition
            random.randint(61, 80)    # Wet condition
        ]),

        "temperature": random.randint(20, 40),
        "humidity": random.randint(40, 90),
        "soil_pH": round(random.uniform(5.5, 7.5), 2),
        "light_intensity": random.randint(200, 1000)
    }

    return data


# Create CSV file with headers if file is missing or empty
try:
    df = pd.read_csv(FILE_NAME)

except (FileNotFoundError, pd.errors.EmptyDataError):

    df = pd.DataFrame(columns=[
        "timestamp",
        "soil_moisture",
        "temperature",
        "humidity",
        "soil_pH",
        "light_intensity"
    ])

    df.to_csv(FILE_NAME, index=False)


print("Generating live sensor data...\n")

# Infinite loop for live simulation
while True:

    sensor_data = generate_sensor_data()

    print(sensor_data)

    # Smart alerts
    if sensor_data["soil_moisture"] < 30:
        print("ALERT: Irrigation Required!")

    if sensor_data["temperature"] > 35:
        print("ALERT: High Temperature Detected!")

    if sensor_data["humidity"] < 45:
        print("ALERT: Low Humidity Warning!")

    print("-" * 50)

    # Append data to CSV
    df = pd.DataFrame([sensor_data])

    df.to_csv(
        FILE_NAME,
        mode='a',
        header=False,
        index=False
    )

    # Wait 5 seconds
    time.sleep(5)