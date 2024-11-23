#ifndef ANIMATOR_H
#define ANIMATOR_H

#include <vector>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

class Animator {
public:
    struct Waypoint {
        glm::vec3 position;
        glm::quat rotation;
        double timestamp;
    };

    bool running = false;

    Animator() = default;

    void loadAnimation(const std::string& pathFile);
    void update(double deltaTime);

    glm::vec3 getCurrentPosition() const;
    glm::quat getCurrentRotation() const;

    void setPlaybackSpeed(float speed) { playbackSpeed = speed; }
    float getPlaybackSpeed() const { return playbackSpeed; }

private:
    std::vector<Waypoint> waypoints;
    size_t currentIndex = 0;
    double timeAccumulator = 0.0;
    float playbackSpeed = 1.0f;
};

#endif // ANIMATOR_H
