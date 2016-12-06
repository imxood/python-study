import socket
import sys

print 'hello, python!'
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error, msg:
    print 'failed to create socket. Error code: ' + str(msg[0]), 'Error message: ' + msg[1]
    sys.exit();

print 'socket created!'

host = 'www.zhihu.com'
port = 80

try:
    remote_ip = socket.gethostbyname( host )
except socket.gaierror:
    print 'Hostname could not be resoled. Exiting.'
    sys.exit();

print 'Ip address of ' + host + ' is ' + remote_ip;

s.connect((remote_ip , port))
 
print 'Socket Connected to ' + host + ' on ip ' + remote_ip

message = "GET / HTTP/1.1\r\n\r\n"

try:
    s.sendall(message)
except socket.error:
    print 'Send failed'
    sys.exit()

print 'Message send successfully'

reply = s.recv(4096)

print reply