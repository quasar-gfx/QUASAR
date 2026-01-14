#ifndef NETWORK_SIMULATOR_H
#define NETWORK_SIMULATOR_H

#include <random>
#include <deque>
#include <vector>

#include <CameraPose.h>

namespace quasar {

struct NetworkSimulatorCreateParams {
    double networkLatencyMs;
    double networkJitterMs;
    double renderTimeMs;
    uint seed = 42;
};

class NetworkSimulator {
public:
    NetworkSimulator(NetworkSimulatorCreateParams params);

    void setNetworkLatency(double networkLatencyMs);
    void setNetworkJitter(double networkJitterMs);
    void setRenderTime(double renderTimeMs);
    void clear();

    void sendPose(const Pose& pose, double now);
    void update(double now);
    bool recvPose(Pose& pose, double now, double& originalTimestamp);

    double getNetworkLatency() const { return networkLatencyS; }
    double getRenderTime() const { return renderTimeS; }

    const std::vector<double>& getRTTs() const { return rtts; }
    double getRTTMean() const;
    double getRTTStdDev() const;

private:
    double networkLatencyS;
    double networkJitterS;
    double renderTimeS;

    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution;

    double lastUpdateTimeS = -1.0;
    std::deque<Pose> incomingPoses;
    std::deque<Pose> outPoses;
    std::deque<double> outOrigTimestamps;

    std::vector<double> rtts;

    double actualInJitter;
    double actualOutJitter;

    double randomJitter();
};

} // namespace quasar

#endif // NETWORK_SIMULATOR_H
