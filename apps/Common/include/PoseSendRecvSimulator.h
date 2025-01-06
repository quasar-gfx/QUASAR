#ifndef POSE_SIM_H
#define POSE_SIM_H

#include <deque>

#include <Cameras/PerspectiveCamera.h>
#include <CameraPose.h>

class PoseSendRecvSimulator {
public:
    double networkLatency = 0.0;

    PoseSendRecvSimulator(double networkLatency) : networkLatency(networkLatency) {}

    void setNetworkLatency(double networkLatency) {
        this->networkLatency = networkLatency;
    }

    void addPose(const PerspectiveCamera &camera, double now) {
        poses.push_back({camera.getViewMatrix(), camera.getProjectionMatrix(), 0});
        timestamps.push_back(now);
    }

    bool getPose(Pose &pose, double now) {
        if (poses.empty() || timestamps.empty()) {
            return false;
        }

        if (now - timestamps.front() >= networkLatency / MILLISECONDS_IN_SECOND) {
            pose = poses.front();
            poses.pop_front();
            timestamps.pop_front();
            return true;
        }
        else {
            return false;
        }
    }

private:
    std::deque<Pose> poses;
    std::deque<double> timestamps;
};

#endif // POSE_SIM_H
