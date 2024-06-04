#ifndef SOCKET_H
#define SOCKET_H

#include <unistd.h>
#include <stdexcept>

#include <fcntl.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

class Socket {
public:
    int socketId;
    struct sockaddr_in addr;
    socklen_t addrLen;

    explicit Socket(int domain, int type, int protocol, bool nonBlocking = false) {
        socketId = socket(domain, type, protocol);
        if (socketId < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        if (nonBlocking) {
            setNonBlocking();
        }
    }

    void setNonBlocking() {
        int flags = fcntl(socketId, F_GETFL, 0);
        if (flags == -1) {
            throw std::runtime_error("Error getting socket flags");
        }
        if (fcntl(socketId, F_SETFL, flags | O_NONBLOCK) == -1) {
            throw std::runtime_error("Error setting socket to non-blocking");
        }
    }

    void setRecvSize(int size) {
        if (setsockopt(socketId, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)) < 0) {
            throw std::runtime_error("Failed to set receive buffer size");
        }
    }

    void bind(const struct sockaddr* addr, socklen_t addrLen) {
        if (::bind(socketId, addr, addrLen) < 0) {
            throw std::runtime_error("Failed to bind socket");
        }
    }

    void bind(std::string ipAddress, int port) {
        addrLen = sizeof(addr);
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr(ipAddress.c_str());
        addr.sin_port = htons(port);
        bind((struct sockaddr*)&addr, addrLen);
    }

    void bind(std::string ipAddressAndPort) {
        size_t pos = ipAddressAndPort.find(':');
        std::string ipAddress = ipAddressAndPort.substr(0, pos);
        std::string portStr = ipAddressAndPort.substr(pos + 1);
        int port = std::stoi(portStr);
        bind(ipAddress, port);
    }

    void listen(int backlog) {
        if (::listen(socketId, backlog) < 0) {
            throw std::runtime_error("Failed to listen on socket");
        }
    }

    int accept(struct sockaddr* addr, socklen_t* addrLen) {
        return ::accept(socketId, addr, addrLen);
    }

    void connect(const struct sockaddr* addr, socklen_t addrLen) {
        if (::connect(socketId, addr, addrLen) < 0) {
            throw std::runtime_error("Failed to connect to socket");
        }
    }

    void connect(std::string ipAddress, int port) {
        addrLen = sizeof(addr);
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr(ipAddress.c_str());
        addr.sin_port = htons(port);
        connect((struct sockaddr*)&addr, addrLen);
    }

    void connect(std::string ipAddressAndPort) {
        size_t pos = ipAddressAndPort.find(':');
        std::string ipAddress = ipAddressAndPort.substr(0, pos);
        std::string portStr = ipAddressAndPort.substr(pos + 1);
        int port = std::stoi(portStr);
        connect(ipAddress, port);
    }

    virtual int send(const void* buf, size_t len, int flags) {
        return ::send(socketId, buf, len, flags);
    }

    virtual int recv(void* buf, size_t len, int flags) {
        return ::recv(socketId, buf, len, flags);
    }

    void close() {
        ::close(socketId);
    }
};

class SocketUDP : public Socket {
public:
    explicit SocketUDP(bool nonBlocking = false) : Socket(AF_INET, SOCK_DGRAM, 0, nonBlocking) {}

    int send(const void* buf, size_t len, int flags) override {
        return ::sendto(socketId, buf, len, flags, (struct sockaddr*)&addr, addrLen);
    }

    int recv(void* buf, size_t len, int flags) override {
        return ::recvfrom(socketId, buf, len, flags, (struct sockaddr*)&addr, &addrLen);
    }
};

class SocketTCP : public Socket {
public:
    explicit SocketTCP(bool nonBlocking = false) : Socket(AF_INET, SOCK_STREAM, 0, nonBlocking) {}
};

#endif // SOCKET_H
