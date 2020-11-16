import socket
import threading

#Variables for holding information about connections
connections = []
total_connections = 0


def server_receive(socket, client):
    while client.signal:
        try:
            data = socket.recv(32)
            print(str(data.decode("utf-8")))
        except:
            print("Client " + str(address) + " has disconnected")
            client.signal = False
            connections.remove(client)
            total_connections -= 1
            break

#Client class, new instance created for each connected client
#Each instance has the socket and address that is associated with items
#Along with an assigned ID and a name chosen by the client
class Client(threading.Thread):
    def __init__(self, socket, address, id, name, signal):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = address
        self.id = id
        self.name = name
        self.signal = signal

    def __str__(self):
        return str(self.id) + " " + str(self.address)
    
    #Attempt to get data from client
    #If unable to, assume client has disconnected and remove him from server data
    #If able to and we get data back, print it in the server and send it back to every
    #client aside from the client that has sent it
    #.decode is used to convert the byte data into a printable string
    def run(self):
        receiveThread = threading.Thread(target=server_receive, args=(self.socket, self))
        receiveThread.start()
        while self.signal:
            message = input()
            for client in connections:
                client.socket.sendall(str.encode(message))


#Wait for new connections
def newConnections(socket):
    while True:
        sock, address = socket.accept()
        global total_connections
        connections.append(Client(sock, address, total_connections, "Name", True))
        connections[len(connections) - 1].start()
        print("New connection at ID " + str(connections[len(connections) - 1]))
        total_connections += 1


def main():
    #Get host and port
    # host = input("Host: ")
    # port = int(input("Port: "))
    host = '127.0.0.1'
    port = 33334

    #Create new server socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(5)

    #Create new thread to wait for connections
    newConnectionsThread = threading.Thread(target=newConnections, args=(sock,))
    newConnectionsThread.start()
    
main()