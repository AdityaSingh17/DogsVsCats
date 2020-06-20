import requests
import json

# Define API Endpoint and Headers.
ApiEndpoint = "http://dogorcat.pythonanywhere.com/query"
Headers = {"Content-Type": "application/json"}

# Specify the URL.
URL = "<URL HERE>"

# Convert the data into JSON.
Data = json.dumps({"url": URL})
# Generate a POST request at the API Endpoint with Headers and JSON data.
response = requests.post(ApiEndpoint, headers=Headers, data=Data)

try:
    # A text response is returned, convert it to JSON.
    JSONResponse = json.loads(response.text)
    # Print the result.
    print("Predicted result:", JSONResponse["Prediction"])
except:
    # In case of any error, print the response text as is.
    print(response.text)
