#include <fstream>
#include <sstream>
#include <iostream>

#include <Animator.h>

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
    while(std::getline(file, line)) {
        std::stringstream ss(line);
        float px, py, pz;
        float rx, ry, rz;
        int64_t timestamp;
        ss >> px >> py >> pz >> rx >> ry >> rz >> timestamp;

        glm::vec3 position = glm::vec3(px, py, pz);
        glm::vec3 eulerAnglesRadians = glm::radians(glm::vec3(rx, ry, rz));
        glm::quat rotation = glm::quat(eulerAnglesRadians);

        waypoints.push_back({ position, rotation, static_cast<double>(timestamp) });
    }
    file.close();
}

void Animator::update(double dt) {
    if (!running || waypoints.size() < 2)
        return;

    if (tween) {
        timeAccumulator += (dt * 1000.0);
        while(currentIndex < waypoints.size() - 1 && timeAccumulator >= waypoints[currentIndex + 1].timestamp) {
            currentIndex++;
        }
    }
    else {
        currentIndex++;
    }

    if (currentIndex >= waypoints.size()) {
        running = false;
        currentIndex = waypoints.size() - 1;
    }
}

glm::vec3 Animator::getCurrentPosition() const {
    if (waypoints.empty())
        return glm::vec3(0.0f);

    if (!running || currentIndex >= waypoints.size())
        return waypoints.back().position;

    if (tween) {
        const Waypoint& start = waypoints[currentIndex];
        const Waypoint& end = waypoints[currentIndex + 1];

        double segmentDuration = end.timestamp - start.timestamp;
        double segmentTime = timeAccumulator - start.timestamp;
        float t = static_cast<float>(segmentTime / segmentDuration);

        return glm::mix(start.position, end.position, t);
    }
    else {
        return waypoints[currentIndex].position;
    }
}

glm::quat Animator::getCurrentRotation() const {
    if (waypoints.empty())
        return glm::quat();

    if (!running || currentIndex >= waypoints.size())
        return waypoints.back().rotation;

    if (tween) {
        const Waypoint& start = waypoints[currentIndex];
        const Waypoint& end = waypoints[currentIndex + 1];

        double segmentDuration = end.timestamp - start.timestamp;
        double segmentTime = timeAccumulator - start.timestamp;
        float t = static_cast<float>(segmentTime / segmentDuration);

        return glm::slerp(start.rotation, end.rotation, t);
    }
    else {
        return waypoints[currentIndex].rotation;
    }
}
