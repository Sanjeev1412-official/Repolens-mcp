import socket

def run_proxy():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', 9999))
    server.listen(1)
    print("Listening on port 9999...")
    
    while True:
        client, addr = server.accept()
        data = client.recv(4096)
        print("--- RECEIVED REQUEST ---")
        print(data.decode('utf-8', errors='ignore'))
        
        # Send a basic 200 OK
        response = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
        client.send(response)
        client.close()

if __name__ == "__main__":
    run_proxy()
