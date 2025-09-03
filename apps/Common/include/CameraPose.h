#ifndef CAMERA_POSE_H
#define CAMERA_POSE_H

#include <glm/glm.hpp>

#include <Path.h>
#include <Utils/FileIO.h>
#include <Cameras/PerspectiveCamera.h>
#include <Cameras/VRCamera.h>

namespace quasar {

typedef uint32_t pose_id_t;

struct Pose {
    pose_id_t id;
    union {
        struct {
            glm::mat4 viewL;
            glm::mat4 viewR;
            glm::mat4 projL;
            glm::mat4 projR;
        } stereo;
        struct {
            glm::mat4 view;
            glm::mat4 pad1;
            glm::mat4 proj;
            glm::mat4 pad2;
        } mono;
    };
    time_t timestamp;

    Pose() = default;
    Pose(const glm::mat4& view, const glm::mat4& proj, time_t timestamp)
            : mono{view, glm::mat4(1.0f), proj, glm::mat4(1.0f)}
            , timestamp(timestamp) {}

    void setViewMatrix(const glm::mat4& view) {
        mono.view = view;
    }

    void setProjectionMatrix(const glm::mat4& proj) {
        mono.proj = proj;
    }

    void setViewMatrices(const glm::mat4 (&views)[2]) {
        stereo.viewL = views[0];
        stereo.viewR = views[1];
    }

    void setProjectionMatrices(const glm::mat4 (&projs)[2]) {
        stereo.projL = projs[0];
        stereo.projR = projs[1];
    }

    void copyPoseToCamera(PerspectiveCamera& camera, bool copyProj = true) {
        if (copyProj) {
            camera.setProjectionMatrix(mono.proj);
        }
        camera.setViewMatrix(mono.view);
    }

    void copyPoseToCamera(VRCamera& camera) {
        camera.setProjectionMatrices({ stereo.projL, stereo.projR });
        camera.setViewMatrices({ stereo.viewL, stereo.viewR });
    }

    void copyPoseFromCamera(const PerspectiveCamera& camera) {
        mono.proj = camera.getProjectionMatrix();
        mono.view = camera.getViewMatrix();
    }

    void copyPoseFromCamera(const VRCamera& camera) {
        stereo.projL = camera.left.getProjectionMatrix();
        stereo.viewL = camera.left.getViewMatrix();
        stereo.projR = camera.right.getProjectionMatrix();
        stereo.viewR = camera.right.getViewMatrix();
    }

    size_t writeToFile(const Path& outputPath) const {
        return FileIO::writeToBinaryFile(outputPath, this, sizeof(Pose));
    }

    size_t writeToMemory(std::vector<char>& outputData) const {
        outputData.resize(sizeof(Pose));
        std::memcpy(outputData.data(), this, sizeof(Pose));
        return sizeof(Pose);
    }

    size_t loadFromFile(const Path& inputPath) {
        const std::vector<char>& data = FileIO::loadFromBinaryFile(inputPath);
        std::memcpy(this, data.data(), sizeof(Pose));
        return sizeof(Pose);
    }

    size_t loadFromMemory(const char* inputData, size_t inputSize) {
        if (inputSize < sizeof(Pose)) {
            throw std::runtime_error("Input data size " +
                                      std::to_string(inputSize) +
                                      " is smaller than Pose size " +
                                      std::to_string(sizeof(Pose)));
        }
        std::memcpy(this, inputData, sizeof(Pose));
        return sizeof(Pose);
    }
};

} // namespace quasar

#endif // CAMERA_POSE_H
