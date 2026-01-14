#include <numeric>
#include <cmath>

#include <NetworkSimulator.h>
#include <Utils/TimeUtils.h>

using namespace quasar;

NetworkSimulator::NetworkSimulator(NetworkSimulatorCreateParams params)
    : networkLatencyS(timeutils::millisToSeconds(params.networkLatencyMs))
    , networkJitterS(timeutils::millisToSeconds(params.networkJitterMs))
    , renderTimeS(timeutils::millisToSeconds(params.renderTimeMs))
    , generator(params.seed)
    , distribution(-networkJitterS, networkJitterS)
    , actualInJitter(randomJitter())
    , actualOutJitter(randomJitter())
{}

void NetworkSimulator::setNetworkLatency(double networkLatencyMs) {
    networkLatencyS = timeutils::millisToSeconds(networkLatencyMs);
    clear();
}

void NetworkSimulator::setNetworkJitter(double networkJitterMs) {
    networkJitterS = timeutils::millisToSeconds(networkJitterMs);
    distribution = std::uniform_real_distribution<double>(-networkJitterS, networkJitterS);
    clear();
}

void NetworkSimulator::setRenderTime(double renderTimeMs) {
    renderTimeS = timeutils::millisToSeconds(renderTimeMs);
    clear();
}

void NetworkSimulator::clear() {
    incomingPoses.clear();
    outPoses.clear();
    outOrigTimestamps.clear();
    rtts.clear();
}

void NetworkSimulator::sendPose(const Pose& pose, double now) {
    Pose poseCopy = pose;
    poseCopy.timestamp = static_cast<double>(timeutils::secondsToMicros(now));
    incomingPoses.push_back(poseCopy);
    update(now);
}

void NetworkSimulator::update(double now) {
    if (now <= lastUpdateTimeS) return;
    lastUpdateTimeS = now;

    if (!incomingPoses.empty()) {
        double dtFuture = networkLatencyS;
        Pose poseToRecv = incomingPoses.front();
        double timestampS = timeutils::microsToSeconds(poseToRecv.timestamp);
        if (networkLatencyS > 0 && now - timestampS < dtFuture + actualInJitter) return;

        poseToRecv.timestamp = static_cast<double>(timeutils::secondsToMicros(now));
        actualInJitter = randomJitter();

        outPoses.push_back(poseToRecv);
        outOrigTimestamps.push_back(timestampS);
        incomingPoses.pop_front();
    }
}

bool NetworkSimulator::recvPose(Pose& pose, double now, double& originalTimestamp) {
    if (outPoses.empty() && outOrigTimestamps.empty()) return false;

    double dtFuture = renderTimeS;
    double timestampS = timeutils::microsToSeconds(outPoses.front().timestamp);
    if (networkLatencyS > 0 && now - timestampS < dtFuture + actualOutJitter) return false;

    actualOutJitter = randomJitter();

    double oldTimestampS = outOrigTimestamps.front();
    rtts.push_back(timeutils::secondsToMillis(now - oldTimestampS));
    originalTimestamp = oldTimestampS;

    pose = (networkLatencyS != 0) ? outPoses.front() : outPoses.back();

    outPoses.pop_front();
    outOrigTimestamps.pop_front();

    return true;
}

double NetworkSimulator::randomJitter() {
    return distribution(generator);
}

double NetworkSimulator::getRTTMean() const {
    if (rtts.empty()) return 0.0;
    return std::accumulate(rtts.begin(), rtts.end(), 0.0) / rtts.size();
}

double NetworkSimulator::getRTTStdDev() const {
    if (rtts.size() < 2) return 0.0;
    double mean = getRTTMean();
    double sumSquaredDiffs = 0.0;
    for (double val : rtts) {
        sumSquaredDiffs += (val - mean) * (val - mean);
    }
    return std::sqrt(sumSquaredDiffs / (rtts.size() - 1));
}
