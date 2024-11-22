#include "Animator.h"
#include <fstream>
#include <sstream>
#include <iostream>

Animator::Animator(const std::string& pathFile)
    : currentIndex(0), timeAccumulator(0.0), finished(false) {
    loadPath(pathFile);
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
        int64_t timestamp;
        ss >> px >> py >> pz >> rx >> ry >> rz >> timestamp;
        
        glm::vec3 position(px, py, pz);
        glm::vec3 eulerAngles(glm::radians(rx), glm::radians(ry), glm::radians(rz));
        glm::quat rotation = glm::quat(eulerAngles);
        
        waypoints.push_back({ position, rotation, static_cast<double>(timestamp) });
    }
    file.close();

    std::cout << "Loaded " << waypoints.size() << " waypoints" << std::endl;
    if (!waypoints.empty()) {
        std::cout << "First timestamp: " << waypoints.front().timestamp << std::endl;
        std::cout << "Last timestamp: " << waypoints.back().timestamp << std::endl;
    }
}

void Animator::update(double deltaTime) {
    if (finished || waypoints.size() < 2)
        return;

    timeAccumulator += (deltaTime * 1000.0 * playbackSpeed);
    
    std::cout << "Time: " << timeAccumulator << " / " << waypoints.back().timestamp << std::endl;
    
    while (currentIndex < waypoints.size() - 1 && 
           timeAccumulator >= waypoints[currentIndex + 1].timestamp) {
        currentIndex++;
    }

    if (currentIndex >= waypoints.size() - 1) {
        finished = true;
        currentIndex = waypoints.size() - 1;
        return;
    }
}

glm::vec3 Animator::getCurrentPosition() const {
    if (waypoints.empty())
        return glm::vec3(0.0f);

    if (finished || currentIndex >= waypoints.size() - 1)
        return waypoints.back().position;

    const Waypoint& start = waypoints[currentIndex];
    const Waypoint& end = waypoints[currentIndex + 1];
    
    double segmentDuration = end.timestamp - start.timestamp;
    double segmentTime = timeAccumulator - start.timestamp;
    float t = static_cast<float>(segmentTime / segmentDuration);
    
    std::cout << "Interpolating position: t = " << t << std::endl;
    
    return glm::mix(start.position, end.position, t);
}

glm::quat Animator::getCurrentRotation() const {
    if (waypoints.empty())
        return glm::quat();

    if (finished || currentIndex >= waypoints.size() - 1)
        return waypoints.back().rotation;

    const Waypoint& start = waypoints[currentIndex];
    const Waypoint& end = waypoints[currentIndex + 1];
    
    double segmentDuration = end.timestamp - start.timestamp;
    double segmentTime = timeAccumulator - start.timestamp;
    float t = static_cast<float>(segmentTime / segmentDuration);
    
    return glm::slerp(start.rotation, end.rotation, t);
}

bool Animator::isFinished() const {
    return finished;
}
