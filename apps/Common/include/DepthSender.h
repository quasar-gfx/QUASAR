#ifndef DEPTH_SENDER_H
#define DEPTH_SENDER_H

#include <DataStreamer.h>

#include <CameraPose.h>

class DepthSender {
public:
    std::string receiverURL;

    DataStreamerTCP* streamer;

    RenderTarget* renderTarget;

    int imageSize;

    explicit DepthSender(RenderTarget* renderTarget, std::string receiverURL)
            : receiverURL(receiverURL)
            , renderTarget(renderTarget)
            , imageSize(sizeof(pose_id_t) + renderTarget->width * renderTarget->height * sizeof(GLushort)) {
        streamer = new DataStreamerTCP(receiverURL, imageSize);
        data = new uint8_t[imageSize];

        CUdevice device = findCudaDevice();
        if (device == -1) {
            throw std::runtime_error("No CUDA device found");
        }

        cudaError_t cudaErr = cudaGraphicsGLRegisterImage(&cudaResource, renderTarget->colorBuffer.ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
        if (cudaErr != cudaSuccess) {
            throw std::runtime_error("Failed to register GL image with CUDA");
        }
    }
    ~DepthSender() {
#ifndef __APPLE__
            cudaDeviceSynchronize();
            cudaError_t cudaErr = cudaGraphicsUnregisterResource(cudaResource);
            if (cudaErr != cudaSuccess) {
                av_log(nullptr, AV_LOG_ERROR, "Error: Couldn't unregister CUDA resource: %s\n", cudaGetErrorString(cudaErr));
            }
#endif

        delete streamer;
        delete[] data;
    }

    void sendFrame(unsigned int poseID) {
        memcpy(data, &poseID, sizeof(pose_id_t));

#ifndef __APPLE__
        cudaError_t cudaErr;

        cudaErr = cudaGraphicsMapResources(1, &cudaResource);
        if (cudaErr != cudaSuccess) {
            throw std::runtime_error("Failed to map CUDA resources");
            return;
        }

        cudaErr = cudaGraphicsSubResourceGetMappedArray(&cudaBuffer, cudaResource, 0, 0);
        if (cudaErr != cudaSuccess) {
            throw std::runtime_error("Failed to get CUDA buffer");
            return;
        }

        cudaErr = cudaMemcpy2DFromArray(data + sizeof(pose_id_t), renderTarget->width * sizeof(GLushort),
                                        cudaBuffer,
                                        0, 0, renderTarget->width * sizeof(GLushort), renderTarget->height,
                                        cudaMemcpyDeviceToHost);
        if (cudaErr != cudaSuccess) {
            throw std::runtime_error("Failed to copy CUDA buffer to host");
            return;
        }

        cudaErr = cudaGraphicsUnmapResources(1, &cudaResource);
        if (cudaErr != cudaSuccess) {
            throw std::runtime_error("Failed to unmap CUDA resources");
            return;
        }
#else
        renderTarget->bind();
        glReadPixels(0, 0, renderTarget->width, renderTarget->height, GL_RED, GL_UNSIGNED_SHORT, data + sizeof(pose_id_t));
        renderTarget->unbind();
#endif

        streamer->send((const uint8_t*)data);
    }

private:
    uint8_t* data;

#ifndef __APPLE__
    cudaGraphicsResource* cudaResource;
    cudaArray* cudaBuffer;

    CUdevice findCudaDevice() {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);

        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found" << std::endl;
            return -1;
        }

        CUdevice device;

        char name[100];
        cuDeviceGet(&device, 0);
        cuDeviceGetName(name, 100, device);
        std::cout << "CUDA Device: " << name << std::endl;

        return device;
    }
#endif
};

#endif // DEPTH_SENDER_H
