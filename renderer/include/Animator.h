#ifndef ANIMATOR_H
#define ANIMATOR_H

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp> 

class Animator {
public:
    Animator(const std::string& pathFile);

    // Update the animator's state
    void update(double deltaTime);

    // Get current camera position and rotation
    glm::vec3 getCurrentPosition() const;
    glm::quat getCurrentRotation() const;

    // Check if the animation has finished
    bool isFinished() const;

private:
    // Load path from file
    void loadPath(const std::string& pathFile);

    // Waypoint structure
    struct Waypoint {
        glm::vec3 position;
        glm::quat rotation;
    };

    std::vector<Waypoint> waypoints;
    size_t currentIndex;
    double timeAccumulator;
    double totalDuration;
    double segmentDuration;

    bool finished;
};

#endif // ANIMATOR_H
