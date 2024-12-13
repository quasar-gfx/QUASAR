#include <fstream>
#include <sstream>
#include <iostream>

#include <Animator.h>
#include <Utils/TimeUtils.h>

Animator::Animator(const std::string& pathFile, bool tween) : tween(tween) {
    if (!pathFile.empty()) {
        loadAnimation(pathFile);
    }
}

void Animator::loadAnimation(const std::string& pathFile) {
    running = true;

    currentIndex = 0;

    std::ifstream file(pathFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open camera path file: " << pathFile << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        float px, py, pz;
        float rx, ry, rz;
        int64_t timestampMillis;
        ss >> px >> py >> pz >> rx >> ry >> rz >> timestampMillis;

        glm::vec3 position = glm::vec3(px, py, pz);
        glm::vec3 rotationEuler = glm::radians(glm::vec3(rx, ry, rz));
        glm::quat rotationQuat = glm::quat(rotationEuler);

        waypoints.push_back({ position, rotationQuat, static_cast<double>(timestampMillis) });
    }
    file.close();
}

void Animator::update(double dt) {
    if (!running || waypoints.size() < 2)
        return;

    if (tween) {
        now += timeutils::secondsToMillis(dt);
        while (currentIndex < waypoints.size() - 1 && now >= waypoints[currentIndex + 1].timestamp) {
            currentIndex++;
        }
    }
    else {
        now = waypoints[currentIndex].timestamp;
        currentIndex++;
    }

    if (currentIndex >= waypoints.size()) {
        currentIndex = waypoints.size() - 1;
        running = false;
    }

    if (currentIndex > 0) {
        auto deltaMillis = waypoints[currentIndex].timestamp - waypoints[currentIndex - 1].timestamp;
        auto deltaSeconds = timeutils::millisToSeconds(deltaMillis);
        this->dt = glm::max(deltaSeconds, 0.0);
    }
}

void Animator::copyPoseToCamera(PerspectiveCamera &camera) const {
    if (waypoints.empty())
        return;

    if (!running || currentIndex >= waypoints.size())
        return;

    camera.setPosition(getCurrentPosition());
    camera.setRotationQuat(getCurrentRotation());
    camera.updateViewMatrix();
}

const glm::vec3 Animator::getCurrentPosition() const {
    if (!running || currentIndex >= waypoints.size())
        return waypoints.back().position;

    if (tween) {
        const Waypoint& start = waypoints[currentIndex];
        const Waypoint& end = waypoints[currentIndex + 1];

        double segmentDuration = end.timestamp - start.timestamp;
        double segmentTime = now - start.timestamp;
        float t = static_cast<float>(segmentTime / segmentDuration);

        return glm::mix(start.position, end.position, t);
    }
    else {
        return waypoints[currentIndex].position;
    }
}

const glm::quat Animator::getCurrentRotation() const {
    if (waypoints.empty())
        return glm::quat();

    if (!running || currentIndex >= waypoints.size())
        return waypoints.back().rotation;

    if (tween) {
        const Waypoint& start = waypoints[currentIndex];
        const Waypoint& end = waypoints[currentIndex + 1];

        double segmentDuration = end.timestamp - start.timestamp;
        double segmentTime = now - start.timestamp;
        float t = static_cast<float>(segmentTime / segmentDuration);

        return glm::slerp(start.rotation, end.rotation, t);
    }
    else {
        return waypoints[currentIndex].rotation;
    }
}
