#ifndef DEPTH_SENDER_H
#define DEPTH_SENDER_H

#include <thread>

#include <DataStreamer.h>

#include <CameraPose.h>

#ifndef __APPLE__
#include <cuda_gl_interop.h>
#include <CudaUtils.h>
#endif

class DepthSender : public RenderTarget {
public:
    std::string receiverURL;

    int imageSize;

    struct Stats {
        float timeToCopyFrame = -1.0f;
    } stats;

    explicit DepthSender(const RenderTargetCreateParams &params, std::string receiverURL)
            : receiverURL(receiverURL)
            , imageSize(sizeof(pose_id_t) + params.width * params.height * sizeof(GLushort))
            , streamer(receiverURL)
            , RenderTarget(params) {
        data = std::vector<uint8_t>(imageSize);

        renderTargetCopy = new RenderTarget({
            .width = width,
            .height = height,
            .internalFormat = colorBuffer.internalFormat,
            .format = colorBuffer.format,
            .type = colorBuffer.type,
            .wrapS = colorBuffer.wrapS,
            .wrapT = colorBuffer.wrapT,
            .minFilter = colorBuffer.minFilter,
            .magFilter = colorBuffer.magFilter,
            .multiSampled = colorBuffer.multiSampled
        });

#ifndef __APPLE__
        CUdevice device = CudaUtils::findCudaDevice();
        // register opengl texture with cuda
        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&cudaResource,
                                                     renderTargetCopy->colorBuffer.ID, GL_TEXTURE_2D,
                                                     cudaGraphicsRegisterFlagsReadOnly));

        // start data sending thread
        running = true;
        dataSendingThread = std::thread(&DepthSender::sendData, this);
#endif
    }
    ~DepthSender() {
        close();
    }

    void close() {
#ifndef __APPLE__
        running = false;

        // send dummy to unblock thread
        dataReady = true;
        cv.notify_one();

        if (dataSendingThread.joinable()) {
            dataSendingThread.join();
        }

        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
#endif
    }

    void sendFrame(pose_id_t poseID) {
        renderTargetCopy->bind();
        blitToRenderTarget(*renderTargetCopy);
        renderTargetCopy->unbind();

#ifndef __APPLE__
        std::lock_guard<std::mutex> lock(m);

        // update cuda buffer
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource));
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&cudaBuffer, cudaResource, 0, 0));
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));
#endif

        this->poseID = poseID;

#ifndef __APPLE__
        // tell thread to send data
        dataReady = true;
        cv.notify_one();
#else
        memcpy(data, &poseID, sizeof(pose_id_t));

        bind();
        glReadPixels(0, 0, width, height, GL_RED, GL_UNSIGNED_SHORT, data + sizeof(pose_id_t));
        unbind();

        streamer.send((const uint8_t*)data, imageSize);
#endif
    }

private:
    DataStreamerTCP streamer;

    pose_id_t poseID;
    std::vector<uint8_t> data;
    RenderTarget* renderTargetCopy;

#ifndef __APPLE__
    cudaGraphicsResource* cudaResource;
    cudaArray* cudaBuffer;

    std::thread dataSendingThread;
    std::mutex m;
    std::condition_variable cv;
    bool dataReady = false;

    std::atomic_bool running = false;

    void sendData() {
        while (true) {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [this] { return dataReady; });

            if (running) {
                dataReady = false;
            }
            else {
                break;
            }

            auto startCopyFrame = std::chrono::high_resolution_clock::now();

            memcpy(data.data(), &poseID, sizeof(pose_id_t));

            // copy depth buffer to data
            cudaError_t cudaErr;
            CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(data.data() + sizeof(pose_id_t), width * sizeof(GLushort),
                                                   cudaBuffer,
                                                   0, 0, width * sizeof(GLushort), height,
                                                   cudaMemcpyDeviceToHost));

            auto endCopyFrame = std::chrono::high_resolution_clock::now();

            stats.timeToCopyFrame = std::chrono::duration<float, std::milli>(endCopyFrame - startCopyFrame).count();

            streamer.send(data);
        }
    }
#endif
};

#endif // DEPTH_SENDER_H
