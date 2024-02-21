import http.client
import common.constants as CM

endpoint = CM.ENDPOINT
conn = http.client.HTTPSConnection(endpoint)
headers = {'Content-type': 'application/x-www-form-urlencoded'}

conn.request('POST', '/api/v3/time', headers=headers)
response = conn.getresponse()

print(response.read().decode())

# Output:
# The server's response to your POST request
