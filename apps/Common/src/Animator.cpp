#include <fstream>
#include <sstream>
#include <iostream>

#include <Animator.h>

Animator::Animator(const std::string& pathFile) {
    if (!pathFile.empty()) {
        loadAnimation(pathFile);
    }
}

void Animator::loadAnimation(const std::string& pathFile) {
    running = true;

    std::ifstream file(pathFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << pathFile << std::endl;
        return;
    }

    std::string line;
    while(std::getline(file, line)) {
        std::stringstream ss(line);
        float px, py, pz;
        float rx, ry, rz;
        int64_t timestamp;
        ss >> px >> py >> pz >> rx >> ry >> rz >> timestamp;

        glm::vec3 position(px, py, pz);
        glm::vec3 eulerAngles(glm::radians(rx), glm::radians(ry), glm::radians(rz));
        glm::quat rotation = glm::quat(eulerAngles);

        waypoints.push_back({ position, rotation, static_cast<double>(timestamp) });
    }
    file.close();
}

void Animator::update(double deltaTime) {
    if (!running || waypoints.size() < 2)
        return;

    timeAccumulator += (deltaTime * 1000.0);

    while (currentIndex < waypoints.size() - 1 &&
           timeAccumulator >= waypoints[currentIndex + 1].timestamp) {
        currentIndex++;
    }

    if (currentIndex >= waypoints.size() - 1) {
        running = false;
        currentIndex = waypoints.size() - 1;
    }
}

glm::vec3 Animator::getCurrentPosition() const {
    if (waypoints.empty())
        return glm::vec3(0.0f);

    if (!running || currentIndex >= waypoints.size() - 1)
        return waypoints.back().position;

    const Waypoint& start = waypoints[currentIndex];
    const Waypoint& end = waypoints[currentIndex + 1];

    double segmentDuration = end.timestamp - start.timestamp;
    double segmentTime = timeAccumulator - start.timestamp;
    float t = static_cast<float>(segmentTime / segmentDuration);

    return glm::mix(start.position, end.position, t);
}

glm::quat Animator::getCurrentRotation() const {
    if (waypoints.empty())
        return glm::quat();

    if (!running || currentIndex >= waypoints.size() - 1)
        return waypoints.back().rotation;

    const Waypoint& start = waypoints[currentIndex];
    const Waypoint& end = waypoints[currentIndex + 1];

    double segmentDuration = end.timestamp - start.timestamp;
    double segmentTime = timeAccumulator - start.timestamp;
    float t = static_cast<float>(segmentTime / segmentDuration);

    return glm::slerp(start.rotation, end.rotation, t);
}
