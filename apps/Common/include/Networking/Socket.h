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
#include <cerrno>

namespace quasar {

class Socket {
public:
    int socketID;
    struct sockaddr_in addr;
    socklen_t addrLen;

    Socket(int domain, int type, int protocol, bool nonBlocking = false) {
        socketID = socket(domain, type, protocol);
        if (socketID < 0) {
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
        int flags = fcntl(socketID, F_GETFL, 0);
        if (flags == -1) {
            throw std::runtime_error("Error getting socket flags: " + std::string(std::strerror(errno)));
        }
        if (fcntl(socketID, F_SETFL, flags | O_NONBLOCK) == -1) {
            throw std::runtime_error("Error setting socket to non-blocking: " + std::string(std::strerror(errno)));
        }
    }

    void setRecvSize(int size) {
        if (setsockopt(socketID, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)) < 0) {
            throw std::runtime_error("Failed to set receive buffer size: " + std::string(std::strerror(errno)));
        }
    }

    void setSendSize(int size) {
        if (setsockopt(socketID, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size)) < 0) {
            throw std::runtime_error("Failed to set send buffer size: " + std::string(std::strerror(errno)));
        }
    }

    void setRecvTimeout(int timeout_seconds) {
        struct timeval tv;
        tv.tv_sec = timeout_seconds;
        tv.tv_usec = 0;

        if (setsockopt(socketID, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
            throw std::runtime_error("Failed to set socket receive timeout: " + std::string(std::strerror(errno)));
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
        if (::bind(socketID, addr, addrLen) < 0) {
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
        return ::send(socketID, buf, len, flags);
    }

    virtual int recv(void* buf, size_t len, int flags) {
        return ::recv(socketID, buf, len, flags);
    }

    void close() {
        if (socketID != -1) {
            ::close(socketID);
            socketID = -1;
        }
    }
};

class SocketTCP final : public Socket {
public:
    SocketTCP(bool nonBlocking = false) : Socket(AF_INET, SOCK_STREAM, 0, nonBlocking) {}
    ~SocketTCP() {
        close();
    }

    void setReuseAddr() {
        int opt = 1;
        if (setsockopt(socketID, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            throw std::runtime_error("Failed to set reuse address: " + std::string(std::strerror(errno)));
        }
    }

    void listen(int backlog) {
        if (::listen(socketID, backlog) < 0) {
            throw std::runtime_error("Failed to listen on socket: " + std::string(std::strerror(errno)));
        }
    }

    int accept(struct sockaddr* addr, socklen_t* addrLen) {
        return ::accept(socketID, addr, addrLen);
    }

    int accept() {
        return accept((struct sockaddr*)&addr, &addrLen);
    }

    int connect(const struct sockaddr* addr, socklen_t addrLen) {
        return ::connect(socketID, addr, addrLen);
    }

    int connect(const std::string &ipAddress, int port) {
        setAddress(ipAddress, port);
        return connect((struct sockaddr*)&addr, addrLen);
    }

    int connect(const std::string &ipAddressAndPort) {
        setAddress(ipAddressAndPort);
        return connect((struct sockaddr*)&addr, addrLen);
    }

    int sendToClient(int clientSocketID, const void* buf, size_t len, int flags) {
        return ::send(clientSocketID, buf, len, flags);
    }

    int recvFromClient(int clientSocketID, void* buf, size_t len, int flags) {
        return ::recv(clientSocketID, buf, len, flags);
    }

    int recv(void* buf, size_t len, int flags) override {
        return ::recv(socketID, buf, len, flags);
    }
};

class SocketUDP final : public Socket {
public:
    SocketUDP(bool nonBlocking = false) : Socket(AF_INET, SOCK_DGRAM, 0, nonBlocking) {}
    ~SocketUDP() {
        close();
    }

    int send(const void* buf, size_t len, int flags) override {
        return ::sendto(socketID, buf, len, flags, (struct sockaddr*)&addr, addrLen);
    }

    int recv(void* buf, size_t len, int flags) override {
        return ::recvfrom(socketID, buf, len, flags, (struct sockaddr*)&addr, &addrLen);
    }
};

} // namespace quasar

#endif // SOCKET_H
