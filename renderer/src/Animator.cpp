#include "Animator.h"
#include <fstream>
#include <sstream>
#include <iostream>

Animator::Animator(const std::string& pathFile)
    : currentIndex(0), timeAccumulator(0.0), finished(false) {
    loadPath(pathFile);
    totalDuration = 20.0; // Total animation duration in seconds
    segmentDuration = totalDuration / (waypoints.size() - 1);
}

void Animator::loadPath(const std::string& pathFile) {
    std::ifstream file(pathFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open path file: " << pathFile << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        float px, py, pz;
        float rx, ry, rz;
        ss >> px >> py >> pz >> rx >> ry >> rz;
        glm::vec3 position(px, py, pz);
        glm::vec3 eulerAngles(glm::radians(rx), glm::radians(ry), glm::radians(rz)); // Convert degrees to radians
        glm::quat rotation = glm::quat(eulerAngles);
        waypoints.push_back({ position, rotation });
    }
    file.close();
}

void Animator::update(double deltaTime) {
    if (finished || waypoints.size() < 2)
        return;

    timeAccumulator += deltaTime;

    if (timeAccumulator >= segmentDuration) {
        timeAccumulator -= segmentDuration;
        currentIndex++;

        if (currentIndex >= waypoints.size() - 1) {
            finished = true;
            currentIndex = waypoints.size() - 1; // Ensure we don't go out of bounds
        }
    }
}

glm::vec3 Animator::getCurrentPosition() const {
    if (waypoints.empty())
        return glm::vec3(0.0f);

    if (finished)
        return waypoints.back().position;

    const Waypoint& start = waypoints[currentIndex];
    const Waypoint& end = waypoints[currentIndex + 1];
    float t = static_cast<float>(timeAccumulator / segmentDuration);
    return glm::mix(start.position, end.position, t);
}

glm::quat Animator::getCurrentRotation() const {
    if (waypoints.empty())
        return glm::quat();

    if (finished)
        return waypoints.back().rotation;

    const Waypoint& start = waypoints[currentIndex];
    const Waypoint& end = waypoints[currentIndex + 1];
    float t = static_cast<float>(timeAccumulator / segmentDuration);
    return glm::slerp(start.rotation, end.rotation, t);
}

bool Animator::isFinished() const {
    return finished;
}
