import requests
import json
import sseclient

def test_mcp_sse():
    url = "https://repolens-mcp.onrender.com/sse"
    print(f"Connecting to {url}...")
    
    response = requests.get(url, stream=True, headers={'Accept': 'text/event-stream'})
    print(f"GET status: {response.status_code}")
    
    client = sseclient.SSEClient(response)
    for event in client.events():
        print(f"Event: {event.event}")
        print(f"Data: {event.data}")
        
        if event.event == "endpoint":
            post_url = event.data
            print(f"Found POST endpoint: {post_url}")
            
            # Now let's try to send a POST request
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "smithery-cli", "version": "1.0"}
                }
            }
            
            import urllib.parse
            full_post_url = urllib.parse.urljoin(url, post_url)
            print(f"Sending POST to {full_post_url}...")
            post_response = requests.post(
                full_post_url,
                json=init_payload,
                headers={'Content-Type': 'application/json'}
            )
            print(f"POST status: {post_response.status_code}")
            print(f"POST response: {post_response.text}")
            break

if __name__ == "__main__":
    test_mcp_sse()
