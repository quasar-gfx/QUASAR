#include <fstream>
#include <sstream>

#include <spdlog/spdlog.h>

#include <CameraAnimator.h>
#include <Utils/TimeUtils.h>

CameraAnimator::CameraAnimator(const std::string &pathFile, bool tween) : tween(tween) {
    if (!pathFile.empty()) {
        loadAnimation(pathFile);
    }
}

void CameraAnimator::loadAnimation(const std::string &pathFile) {
    running = true;
    currentIndex = 0;

    std::ifstream file(pathFile);
    if (!file.is_open()) {
        spdlog::error("Failed to open camera path file: {}", pathFile);
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

        waypoints.push_back({ position, rotationQuat, timeutils::millisToSeconds(timestampMillis) });
    }
    file.close();
}

bool CameraAnimator::update(double dt) {
    static bool firstUpdate = true;

    if (!running || waypoints.size() < 2)
        return false;

    bool waypointUpdated = firstUpdate;
    firstUpdate = false;

    if (tween) {
        this->dt = dt;
        now += this->dt;

        while (currentIndex + 1 < waypoints.size() && now >= waypoints[currentIndex + 1].timestamp) {
            currentIndex++;
            waypointUpdated = true;
        }

        if (currentIndex >= waypoints.size() - 1) {
            running = false;
        }
    }
    else {
        currentIndex++;
        if (currentIndex < waypoints.size()) {
            auto deltaSeconds = waypoints[currentIndex].timestamp - waypoints[currentIndex - 1].timestamp;
            this->dt = glm::max(deltaSeconds, 0.0);
            now = waypoints[currentIndex].timestamp;
            waypointUpdated = true;
        }
        else {
            running = false;
        }
    }

    return waypointUpdated;
}

const glm::vec3 CameraAnimator::getCurrentPosition() const {
    if (!running || currentIndex >= waypoints.size())
        return waypoints.back().position;

    if (tween) {
        const CameraPose& start = waypoints[currentIndex];
        const CameraPose& end = waypoints[currentIndex + 1];

        double segmentDuration = end.timestamp - start.timestamp;
        double segmentTime = now - start.timestamp;
        float t = static_cast<float>(segmentTime / segmentDuration);

        return glm::mix(start.position, end.position, t);
    }
    else {
        return waypoints[currentIndex].position;
    }
}

const glm::quat CameraAnimator::getCurrentRotation() const {
    if (waypoints.empty())
        return glm::quat();

    if (!running || currentIndex >= waypoints.size())
        return waypoints.back().rotation;

    if (tween) {
        const CameraPose& start = waypoints[currentIndex];
        const CameraPose& end = waypoints[currentIndex + 1];

        double segmentDuration = end.timestamp - start.timestamp;
        double segmentTime = now - start.timestamp;
        float t = static_cast<float>(segmentTime / segmentDuration);

        return glm::slerp(start.rotation, end.rotation, t);
    }
    else {
        return waypoints[currentIndex].rotation;
    }
}

void CameraAnimator::copyPoseToCamera(PerspectiveCamera &camera) const {
    if (waypoints.empty())
        return;

    if (!running || currentIndex >= waypoints.size())
        return;

    camera.setPosition(getCurrentPosition());
    camera.setRotationQuat(getCurrentRotation());
    camera.updateViewMatrix();
}
