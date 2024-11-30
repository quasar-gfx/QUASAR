#ifndef ANIMATION_H
#define ANIMATION_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

class AnimationProperty {
public:
    glm::vec3 initialState;
    glm::vec3 finalState;
    glm::vec3 currentState;

    bool loop = false;
    bool reverse = false;
    float duration = 1.0f;
    float elapsedTime = 0.0f;

    AnimationProperty();
    AnimationProperty(const glm::vec3& initialState);

    void set(const glm::vec3& from, const glm::vec3& to, float duration, bool reverse = false, bool loop = false);

    void update(float deltaTime);

    void reset();
};

class Animation {
public:
    Animation() = default;

    void setTranslation(const glm::vec3& from, const glm::vec3& to, float duration, bool reverse = false, bool loop = false);
    void setRotation(const glm::vec3& fromEuler, const glm::vec3& toEuler, float duration, bool reverse = false, bool loop = false);
    void setScale(const glm::vec3& from, const glm::vec3& to, float duration, bool reverse = false, bool loop = false);

    void update(float deltaTime);

    const glm::mat4& getTransformation() const;

private:
    AnimationProperty translation{glm::vec3(0.0f, 0.0f, 0.0f)};
    AnimationProperty scale{glm::vec3(1.0f, 1.0f, 1.0f)};
    AnimationProperty rotation{glm::vec3(0.0f, 0.0f, 0.0f)};

    glm::mat4 transformationMatrix = glm::mat4(1.0f);

    void updateTransformation();
};

#endif // ANIMATION_H
