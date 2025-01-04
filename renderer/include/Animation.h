#ifndef ANIMATION_H
#define ANIMATION_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

class Animation {
public:
    struct Keyframe {
        glm::vec3 value;
        double time;
    };

    class AnimationProperty {
    public:
        std::vector<Keyframe> keyframes;
        glm::vec3 currentState = glm::vec3(0.0f);
        glm::vec3 initialState = glm::vec3(0.0f);

        double elapsedTime = 0.0f;

        bool isRotation = false;

        bool reverse = false;
        bool loop = true;

        AnimationProperty() = default;
        AnimationProperty(const glm::vec3& initialState, bool isRotation = false);

        void addKeyframe(const glm::vec3& value, double time);
        void setProperties(bool reverse, bool loop);
        void update(double deltaTime);
        void reset();

    private:
        bool shouldReverse = false;
    };

    Animation() = default;

    void addPositionKey(const glm::vec3& value, double time);
    void addRotationKey(const glm::vec3& value, double time);
    void addScaleKey(const glm::vec3& value, double time);

    void setPositionProperties(bool reverse, bool loop);
    void setRotationProperties(bool reverse, bool loop);
    void setScaleProperties(bool reverse, bool loop);

    void update(double deltaTime);

    const glm::mat4& getTransformation() const;

private:
    AnimationProperty translation{glm::vec3(0.0f, 0.0f, 0.0f), false};
    AnimationProperty scale{glm::vec3(1.0f, 1.0f, 1.0f), false};
    AnimationProperty rotation{glm::vec3(0.0f, 0.0f, 0.0f), true};

    glm::mat4 transformationMatrix = glm::mat4(1.0f);

    void updateTransformation();
};

#endif // ANIMATION_H
