from azure.storage.blob import BlobServiceClient

# Azure connection string
CONNECTION_STRING = "YOUR_CONNECTION_STRING"
# Container name
CONTAINER_NAME = "farm-data"

# File path
FILE_PATH = "../sensor_simulation/sensor_data.csv"

# Blob name
BLOB_NAME = "sensor_data.csv"

# Create blob service client
blob_service_client = BlobServiceClient.from_connection_string(
    CONNECTION_STRING
)

# Create blob client
blob_client = blob_service_client.get_blob_client(
    container=CONTAINER_NAME,
    blob=BLOB_NAME
)

# Upload file
with open(FILE_PATH, "rb") as data:

    blob_client.upload_blob(
        data,
        overwrite=True
    )

print("✅ File uploaded to Azure Blob Storage!")