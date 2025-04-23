#ifndef QUAD_STREAMER_H
#define QUAD_STREAMER_H

#include <Quads/QuadsSimulator.h>
#include <Networking/Socket.h>
#include <VideoStreamer.h>

namespace quasar {

class QuadStreamer: public QuadsSimulator {
public:
    struct Stats {
        unsigned int quadFramesSent = 0;
        unsigned int keyFramesSent = 0;
        unsigned int quadPacketsSent = 0;
        
        double totalQuadCompressTimeMs = 0.0;
        double totalQuadNetworkTimeMs = 0.0;
        
        double videoCopyTimeMs = 0.0;
        double videoEncodeTimeMs = 0.0;
        double videoSendTimeMs = 0.0;
        
        unsigned int totalQuadsSizeBytes = 0;
        unsigned int totalDepthOffsetsSizeBytes = 0;
        
        float videoBitrateMbps = 0.0;
        float videoFrameRate = 0.0;


        double totalRenderTime = 0.0;
        double totalCreateProxiesTime = 0.0;
        double totalGenQuadMapTime = 0.0;
        double totalSimplifyTime = 0.0;
        double totalGatherQuadsTime = 0.0;
        double totalCreateMeshTime = 0.0;
        double totalAppendQuadsTime = 0.0;
        double totalFillQuadsIndiciesTime = 0.0;
        double totalCreateVertIndTime = 0.0;
        double totalGenDepthTime = 0.0;
        double totalCompressTime = 0.0;

        unsigned int totalProxies = 0;
        unsigned int totalDepthOffsets = 0;
        double compressedSizeBytes = 0;
    } stats;
    
    QuadStreamer(const PerspectiveCamera &remoteCamera, FrameGenerator &frameGenerator, 
                const std::string& quadServerAddress, int quadPort, const std::string& videoURL,
                int targetFramerate = 30, int targetBitrate = 50, const std::string& videoFormat = "mpegts")
        : QuadsSimulator(remoteCamera, frameGenerator)
        , quadServerAddress(quadServerAddress)
        , quadPort(quadPort)
        , videoURL(videoURL)
        , videoFormat(videoFormat)
        , targetBitrate(targetBitrate)
        , videoStreamerRT({
            .width = quadsGenerator.remoteWindowSize.x,
            .height = quadsGenerator.remoteWindowSize.y,
            .internalFormat = GL_SRGB8,
            .format = GL_RGB,
            .type = GL_UNSIGNED_BYTE,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_LINEAR,
            .magFilter = GL_LINEAR
        }, videoURL, targetFramerate, targetBitrate, videoFormat)
        , quadSocketConnected(false)
        , quadSocket(nullptr)
        , poseID(0)
    {
        spdlog::info("Initializing QuadStreamer: Video URL={}, Quad Server={}:{}", 
                     videoURL, quadServerAddress, quadPort);
        setupQuadSocket();
    }
    
    ~QuadStreamer() {
        if (clientSocketID != -1) {
            ::close(clientSocketID);
        }
        if (quadSocket) {
            delete quadSocket;
            quadSocket = nullptr;
        }
    }

    unsigned int saveToFile(const std::string &outputPath) {
        // Call parent method to get byte size
        unsigned int totalBytes = QuadsSimulator::saveToFile(outputPath);

                // Prepare geometry data
        std::vector<char> compressedQuadsData;
        std::vector<char> compressedDepthOffsetsData;
        
        // Save data to memory
        quadsGenerator.outputQuadBuffers.saveToMemory(compressedQuadsData, true);
        
        #if !defined(__APPLE__) && !defined(__ANDROID__)
        quadsGenerator.depthOffsets.saveToMemory(compressedDepthOffsetsData, true);
        #endif

        // Stream the data over socket if connected
        if (quadSocketConnected && quadSocket) {
            streamQuadData(compressedQuadsData, compressedDepthOffsetsData);
        }
        
        return totalBytes;
    }
    
    void generateFrame(const PerspectiveCamera& remoteCamera, const Scene& remoteScene,
                      DeferredRenderer& remoteRenderer, bool generateResFrame = false, 
                      bool showNormals = false, bool showDepth = false)  {
        // Call the parent method to generate frame
        QuadsSimulator::generateFrame(remoteCamera, remoteScene, remoteRenderer, 
                                     generateResFrame, showNormals, showDepth);
        
        // Stream the frame data
        sendFrame();
        
        // Increment frame counters
        if (generateResFrame) {
            stats.quadFramesSent++;
        } else {
            stats.keyFramesSent++;
        }
    }
    
    unsigned int sendFrame() {
        double startTime = timeutils::getTimeMicros();
    
        startTime = timeutils::getTimeMicros();
        copyRT.blitToRenderTarget(videoStreamerRT);
        videoStreamerRT.sendFrame(poseID); // from constructor
        stats.videoSendTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Prepare geometry data
        std::vector<char> compressedQuadsData;
        std::vector<char> compressedDepthOffsetsData;
        
        // Save data to memory
        quadsGenerator.outputQuadBuffers.saveToMemory(compressedQuadsData, true);
        
        #if !defined(__APPLE__) && !defined(__ANDROID__)
        quadsGenerator.depthOffsets.saveToMemory(compressedDepthOffsetsData, true);
        #endif
        
        // Stream quad data if socket is connected
        if (quadSocketConnected && quadSocket) {
            streamQuadData(compressedQuadsData, compressedDepthOffsetsData);
        }
        
        // Update stats
        stats.videoFrameRate = videoStreamerRT.getFrameRate(); 
        // Increment pose ID for next frame
        poseID++;
        
        return quads.size() + depthOffsets.size();
    }

private:
    std::string quadServerAddress;
    int quadPort;
    std::string videoURL;
    std::string videoFormat;
    int targetBitrate;
    
    VideoStreamer videoStreamerRT;
    
    SocketTCP* quadSocket;
    bool quadSocketConnected;
    
    uint64_t poseID;
    
    bool setupQuadSocket() {
        try {
            // Create TCP socket
            quadSocket = new SocketTCP(true); // non-blocking
            quadSocket->setReuseAddr();
            
            // Bind to specified address and port
            quadSocket->bind(quadServerAddress, quadPort);
            
            // Listen for connections
            quadSocket->listen(1);  // Accept only one client
            
            quadSocketConnected = true;
            spdlog::info("Quad socket created and listening on {}:{}", quadServerAddress, quadPort);
            
            return true;
        } catch (const std::exception& e) {
            spdlog::error("Failed to start quad server: {}", e.what());
            if (quadSocket) {
                delete quadSocket;
                quadSocket = nullptr;
            }
            quadSocketConnected = false;
            return false;
        }
    }
    
    void streamQuadData(std::vector<char> compressedQuadsData, std::vector<char> compressedDepthOffsetsData) {
        double startTime = timeutils::getTimeMicros();
        
        // Check for new connections if we don't have a client
        if (clientSocketID == -1) {
            clientSocketID = quadSocket->accept();
            if (clientSocketID >= 0) {
                spdlog::info("Quad client connected");
            } else if (errno != EWOULDBLOCK && errno != EAGAIN) {
                spdlog::error("Error accepting client: {}", std::strerror(errno));
            }
        }
        
        if (clientSocketID >= 0) {
            try {
                // Prepare header with sizes
                struct QuadPacketHeader {
                    uint64_t poseID;
                    uint32_t quadsSize;
                    uint32_t depthOffsetsSize;
                } header;
                
                header.poseID = poseID;
                header.quadsSize = compressedQuadsData.size();
                header.depthOffsetsSize = compressedDepthOffsetsData.size();
                
                // Send header
                if (::send(clientSocketID, &header, sizeof(header), 0) < 0) {
                    throw std::runtime_error("Failed to send header");
                }
                
                // Send quads data
                if (quads.size() > 0) {
                    spdlog::info("quad data size {}", compressedQuadsData.size());
                    if (::send(clientSocketID, compressedQuadsData.data(), compressedQuadsData.size(), 0) < 0) {
                        throw std::runtime_error("Failed to send quads data");
                    }
                }
                
                // Send depth offsets data
                if (depthOffsets.size() > 0) {
                    spdlog::info("depthoffset data size {}", compressedDepthOffsetsData.size());
                    if (::send(clientSocketID, compressedDepthOffsetsData.data(), compressedDepthOffsetsData.size(), 0) < 0) {
                        throw std::runtime_error("Failed to send depth offsets");
                    }
                }
                
                stats.quadPacketsSent++;
                stats.totalQuadsSizeBytes += compressedQuadsData.size();
                stats.totalDepthOffsetsSizeBytes += compressedDepthOffsetsData.size();
                
            } catch (const std::exception& e) {
                spdlog::error("Error sending quad data: {}", e.what());
                ::close(clientSocketID);
                clientSocketID = -1;
            }
        }
        
        stats.totalQuadNetworkTimeMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    // Add this member variable
    int clientSocketID = -1;

};

} // namespace quasar

#endif // QUAD_STREAMER_H