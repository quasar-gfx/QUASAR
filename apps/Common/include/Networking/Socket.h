#ifndef SOCKET_H
#define SOCKET_H

#include <unistd.h>
#include <stdexcept>
#include <cstring>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string>
#include <iostream>
#include <cerrno>

class Socket {
public:
    int socketId;
    struct sockaddr_in addr;
    socklen_t addrLen;

    Socket(int domain, int type, int protocol, bool nonBlocking = false) {
        socketId = socket(domain, type, protocol);
        if (socketId < 0) {
            throw std::runtime_error("Failed to create socket: " + std::string(std::strerror(errno)));
        }

        if (nonBlocking) {
            setNonBlocking();
        }
    }
    ~Socket() {
        close();
    }

    void setNonBlocking() {
        int flags = fcntl(socketId, F_GETFL, 0);
        if (flags == -1) {
            throw std::runtime_error("Error getting socket flags: " + std::string(std::strerror(errno)));
        }
        if (fcntl(socketId, F_SETFL, flags | O_NONBLOCK) == -1) {
            throw std::runtime_error("Error setting socket to non-blocking: " + std::string(std::strerror(errno)));
        }
    }

    void setRecvSize(int size) {
        if (setsockopt(socketId, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)) < 0) {
            throw std::runtime_error("Failed to set receive buffer size: " + std::string(std::strerror(errno)));
        }
    }

    void setSendSize(int size) {
        if (setsockopt(socketId, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size)) < 0) {
            throw std::runtime_error("Failed to set send buffer size: " + std::string(std::strerror(errno)));
        }
    }

    void setAddress(const std::string &ipAddress, int port) {
        addrLen = sizeof(addr);
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr(ipAddress.c_str());
        addr.sin_port = htons(port);
    }

    void setAddress(const std::string &ipAddressAndPort) {
        size_t pos = ipAddressAndPort.find(':');
        if (pos == std::string::npos) {
            throw std::invalid_argument("Invalid address format, expected ip:port");
        }
        std::string ipAddress = ipAddressAndPort.substr(0, pos);
        int port;
        try {
            port = std::stoi(ipAddressAndPort.substr(pos + 1));
        } catch (const std::exception &e) {
            throw std::invalid_argument("Invalid port number");
        }
        setAddress(ipAddress, port);
    }

    void bind(const struct sockaddr* addr, socklen_t addrLen) {
        if (::bind(socketId, addr, addrLen) < 0) {
            throw std::runtime_error("Failed to bind socket: " + std::string(std::strerror(errno)));
        }
    }

    void bind(const std::string &ipAddress, int port) {
        setAddress(ipAddress, port);
        bind((struct sockaddr*)&addr, addrLen);
    }

    void bind(const std::string &ipAddressAndPort) {
        setAddress(ipAddressAndPort);
        bind((struct sockaddr*)&addr, addrLen);
    }

    virtual int send(const void* buf, size_t len, int flags) {
        return ::send(socketId, buf, len, flags);
    }

    virtual int recv(void* buf, size_t len, int flags) {
        return ::recv(socketId, buf, len, flags);
    }

    void close() {
        if (socketId != -1) {
            ::close(socketId);
            socketId = -1;
        }
    }
};

class SocketUDP : public Socket {
public:
    SocketUDP(bool nonBlocking = false) : Socket(AF_INET, SOCK_DGRAM, 0, nonBlocking) {}

    int send(const void* buf, size_t len, int flags) override {
        return ::sendto(socketId, buf, len, flags, (struct sockaddr*)&addr, addrLen);
    }

    int recv(void* buf, size_t len, int flags) override {
        return ::recvfrom(socketId, buf, len, flags, (struct sockaddr*)&addr, &addrLen);
    }
};

class SocketTCP : public Socket {
public:
    SocketTCP(bool nonBlocking = false) : Socket(AF_INET, SOCK_STREAM, 0, nonBlocking) {}

    void setReuseAddrPort() {
        int opt = 1;
        if (setsockopt(socketId, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
            throw std::runtime_error("Failed to set reuse address: " + std::string(std::strerror(errno)));
        }
    }

    void listen(int backlog) {
        if (::listen(socketId, backlog) < 0) {
            throw std::runtime_error("Failed to listen on socket: " + std::string(std::strerror(errno)));
        }
    }

    int accept(struct sockaddr* addr, socklen_t* addrLen) {
        clientSocketID = ::accept(socketId, addr, addrLen);
        if (clientSocketID < 0) {
            return -1;
        }
        return 0;
    }

    int accept() {
        return accept((struct sockaddr*)&addr, &addrLen);
    }

    int connect(const struct sockaddr* addr, socklen_t addrLen) {
        return ::connect(socketId, addr, addrLen);
    }

    int connect(const std::string &ipAddress, int port) {
        setAddress(ipAddress, port);
        return connect((struct sockaddr*)&addr, addrLen);
    }

    int connect(const std::string &ipAddressAndPort) {
        setAddress(ipAddressAndPort);
        return connect((struct sockaddr*)&addr, addrLen);
    }

    int send(const void* buf, size_t len, int flags) override {
        return ::send(clientSocketID, buf, len, flags);
    }

    int recv(void* buf, size_t len, int flags) override {
        return ::recv(socketId, buf, len, flags);
    }

private:
    int clientSocketID = -1;
};

#endif // SOCKET_H
