#ifndef QUAD_RECEIVER_H
#define QUAD_RECEIVER_H

#include <Networking/Socket.h>
#include <vector>
#include <cstring>
#include <spdlog/spdlog.h>
#include <Utils/TimeUtils.h>

namespace quasar {

class QuadReceiver {
public:
    struct Stats {
        unsigned int quadPacketsReceived = 0;
        double totalQuadNetworkTimeMs = 0.0;
        unsigned int totalQuadsSizeBytes = 0;
        unsigned int totalDepthOffsetsSizeBytes = 0;
    } stats;
    
    QuadReceiver(const std::string& serverAddress, int serverPort)
        : serverAddress(serverAddress)
        , serverPort(serverPort)
        , socket(nullptr)
        , connected(false)
        , currentPoseID(0)
    {
        spdlog::info("Initializing QuadReceiver: Server={}:{}", serverAddress, serverPort);
        //connectToServer();
    }
    
    ~QuadReceiver() {
        if (socket) {
            delete socket;
            socket = nullptr;
        }
    }
    
    bool receiveQuadData(std::vector<char>& quads, std::vector<char>& depthOffsets) {
        if (!connected || !socket) {
            if (!connectToServer()) {
                return false;
            }
        }
        
        double startTime = timeutils::getTimeMicros();
        
        try {
            // Receive the header first
            struct QuadPacketHeader {
                uint64_t poseID;
                uint32_t quadsSize;
                uint32_t depthOffsetsSize;
            } header;
            
            // Receive header
            spdlog::debug("Receiving header ({} bytes)...", sizeof(header));
            if (!receiveAll(&header, sizeof(header))) {
                spdlog::error("Failed to receive header");
                return false;
            }
            
            // Update pose ID
            currentPoseID = header.poseID;
            spdlog::info("Received pose ID: {}", currentPoseID);
            spdlog::info("Quads size: {} bytes", header.quadsSize);
            spdlog::info("Depth offsets size: {} bytes", header.depthOffsetsSize);
            
            // Handle edge case: zero-sized data (possible initial frame)
            if (header.quadsSize == 0 && header.depthOffsetsSize == 0) {
                spdlog::info("Received frame with no quads or depth offsets data");
                stats.quadPacketsReceived++;
                return false;  // Return false to indicate no actual data to process
            }
            
            // Resize vectors to accommodate incoming data
            try {
                if (header.quadsSize > 0) {
                    spdlog::debug("Resizing quads vector to {} bytes", header.quadsSize);
                    quads.resize(header.quadsSize);
                } else {
                    quads.clear();
                }
                
                if (header.depthOffsetsSize > 0) {
                    spdlog::debug("Resizing depth offsets vector to {} bytes", header.depthOffsetsSize);
                    depthOffsets.resize(header.depthOffsetsSize);
                } else {
                    depthOffsets.clear();
                }
            } catch (const std::exception& e) {
                spdlog::error("Failed to resize data buffers: {}", e.what());
                return false;
            }
            
            // Receive quads data with robust handling
            if (header.quadsSize > 0) {
                spdlog::debug("Receiving quads data ({} bytes)...", header.quadsSize);
                if (!receiveAll(quads.data(), header.quadsSize)) {
                    spdlog::error("Failed to receive complete quads data");
                    return false;
                }
                spdlog::debug("Quads data received successfully");
            }
            
            // Receive depth offsets data with robust handling
            if (header.depthOffsetsSize > 0) {
                spdlog::debug("Receiving depth offsets data ({} bytes)...", header.depthOffsetsSize);
                if (!receiveAll(depthOffsets.data(), header.depthOffsetsSize)) {
                    spdlog::error("Failed to receive complete depth offsets data");
                    return false;
                }
                spdlog::debug("Depth offsets data received successfully");
            }
            
            // Update stats
            stats.quadPacketsReceived++;
            stats.totalQuadsSizeBytes += header.quadsSize;
            stats.totalDepthOffsetsSizeBytes += header.depthOffsetsSize;
            
            double endTime = timeutils::getTimeMicros();
            stats.totalQuadNetworkTimeMs += timeutils::microsToMillis(endTime - startTime);
            
            spdlog::info("Quad data received successfully: {} bytes total", header.quadsSize + header.depthOffsetsSize);
            return true;
            
        } catch (const std::exception& e) {
            spdlog::error("Error receiving quad data: {}", e.what());
            connected = false;
            delete socket;
            socket = nullptr;
            return false;
        }
    }
    
    uint64_t getCurrentPoseID() const {
        return currentPoseID;
    }
    
private:
    std::string serverAddress;
    int serverPort;
    SocketTCP* socket;
    bool connected;
    uint64_t currentPoseID;
    
public:
    bool connectToServer() {
        try {
            // Create TCP socket
            socket = new SocketTCP();
            
            // Set receive buffer size
            try {
                socket->setRecvSize(16 * 1024 * 1024); // 16MB buffer
                spdlog::debug("Set receive buffer size to 16MB");
            } catch (const std::exception& e) {
                spdlog::warn("Failed to set receive buffer size: {}", e.what());
                // Continue anyway, as this is not critical
            }
            
            // Set receive timeout
            try {
                socket->setRecvTimeout(10); // 10 seconds timeout
                spdlog::debug("Set receive timeout to 10 seconds");
            } catch (const std::exception& e) {
                spdlog::warn("Failed to set receive timeout: {}", e.what());
                // Continue anyway, as this is not critical
            }
            
            // Connect to server
            spdlog::debug("Connecting to server at {}:{}...", serverAddress, serverPort);
            int result = socket->connect(serverAddress, serverPort);
            if (result < 0) {
                spdlog::error("Failed to connect to quad server at {}:{}: {}", 
                           serverAddress, serverPort, std::strerror(errno));
                delete socket;
                socket = nullptr;
                connected = false;
                return false;
            }
            
            connected = true;
            spdlog::info("Connected to quad server at {}:{}", serverAddress, serverPort);
            return true;
            
        } catch (const std::exception& e) {
            spdlog::error("Error connecting to quad server: {}", e.what());
            if (socket) {
                delete socket;
                socket = nullptr;
            }
            connected = false;
            return false;
        }
    }
    
    // Helper method to receive exactly 'size' bytes into buffer
    bool receiveAll(void* buffer, size_t size) {
        if (size == 0) {
            spdlog::warn("Attempted to receive 0 bytes");
            return true; // Nothing to receive
        }
        
        if (buffer == nullptr) {
            spdlog::error("Receive buffer is null");
            return false;
        }
        
        char* ptr = static_cast<char*>(buffer);
        size_t bytesRemaining = size;
        int bytesReceived;
        int retryCount = 0;
        const int maxRetries = 10;
        
        while (bytesRemaining > 0) {
            // Receive data
            bytesReceived = socket->recv(ptr, bytesRemaining, 0);
            
            // Check for errors
            if (bytesReceived <= 0) {
                if (bytesReceived == 0) {
                    spdlog::warn("Server disconnected");
                    connected = false;
                    delete socket;
                    socket = nullptr;
                    return false;
                } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // Socket would block, retry a few times with a short delay
                    retryCount++;
                    if (retryCount > maxRetries) {
                        spdlog::error("Maximum retry count reached while receiving data, giving up");
                        return false;
                    }
                    spdlog::debug("Socket would block, retrying ({}/{})", retryCount, maxRetries);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                } else {
                    spdlog::error("Socket receive error: {}", std::strerror(errno));
                    return false;
                }
            }
            
            // Reset retry counter on successful receive
            retryCount = 0;
            
            // Update pointers and counters
            ptr += bytesReceived;
            bytesRemaining -= bytesReceived;
            
            // Log progress for large transfers
            if (size > 1024*1024 && bytesRemaining > 0 && (size - bytesRemaining) % (1024*1024) < bytesReceived) {
                double percent = 100.0 * (size - bytesRemaining) / size;
                spdlog::debug("Receiving data: {:.1f}% ({}/{} bytes)", 
                           percent, size - bytesRemaining, size);
            }
        }
        
        return true;
    }
};

} // namespace quasar

#endif // QUAD_RECEIVER_H