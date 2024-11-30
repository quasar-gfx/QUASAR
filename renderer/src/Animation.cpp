#include <iostream>

#include <Animation.h>

AnimationProperty::AnimationProperty()
    : initialState(0.0f), finalState(0.0f), currentState(0.0f) {}

AnimationProperty::AnimationProperty(const glm::vec3& initialState)
    : initialState(initialState), finalState(initialState), currentState(initialState) {
}

void AnimationProperty::set(const glm::vec3& from, const glm::vec3& to, float duration, bool reverse, bool loop) {
    this->initialState = from;
    this->finalState = to;
    this->currentState = from;
    this->duration = duration;
    this->reverse = reverse;
    this->loop = loop;
    this->elapsedTime = 0.0f;
}

void AnimationProperty::update(float deltaTime) {
    if (duration <= 0.0f) return;

    elapsedTime += deltaTime;
    float t = elapsedTime / duration;

    if (t > 1.0f) {
        if (reverse) {
            elapsedTime = 0.0f;
            std::swap(initialState, finalState);
            t = 0.0f;
        }
        else if (loop) {
            elapsedTime = 0.0f;
            t = 0.0f;
        }
        else {
            t = 1.0f;
        }
    }

    currentState = glm::mix(initialState, finalState, t);
}

void AnimationProperty::reset() {
    elapsedTime = 0.0f;
    currentState = initialState;
}

void Animation::setTranslation(const glm::vec3& from, const glm::vec3& to, float duration, bool reverse, bool loop) {
    translation.set(from, to, duration, reverse, loop);
}

void Animation::setRotation(const glm::vec3& fromEuler, const glm::vec3& toEuler, float duration, bool reverse, bool loop) {
    rotation.set(fromEuler, toEuler, duration, reverse, loop);
}

void Animation::setScale(const glm::vec3& from, const glm::vec3& to, float duration, bool reverse, bool loop) {
    scale.set(from, to, duration, reverse, loop);
}

void Animation::update(float deltaTime) {
    translation.update(deltaTime);
    rotation.update(deltaTime);
    scale.update(deltaTime);

    updateTransformation();
}

void Animation::updateTransformation() {
    transformationMatrix = glm::mat4(1.0f);

    transformationMatrix = glm::translate(transformationMatrix, translation.currentState);

    glm::quat rotationQuat = glm::quat(glm::radians(rotation.currentState));
    transformationMatrix *= glm::mat4_cast(rotationQuat);

    transformationMatrix = glm::scale(transformationMatrix, scale.currentState);
}

const glm::mat4& Animation::getTransformation() const {
    return transformationMatrix;
}
