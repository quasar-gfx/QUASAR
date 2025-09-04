#include <iostream>
#include <algorithm>

#include <spdlog/spdlog.h>

#include <Animation.h>

using namespace quasar;

Animation::AnimationProperty::AnimationProperty(const glm::vec3& initialState, bool isRotation)
    : initialState(initialState)
    , currentState(initialState)
    , isRotation(isRotation)
{}

void Animation::AnimationProperty::setProperties(bool reverse, bool loop) {
    this->reverse = reverse;
    this->loop = loop;
}

void Animation::AnimationProperty::addKeyframe(const glm::vec3& value, double time) {
    keyframes.push_back({value, time});
    std::sort(keyframes.begin(), keyframes.end(), [](const Keyframe& a, const Keyframe& b) {
        return a.time < b.time;
    });
}

void Animation::AnimationProperty::update(double deltaTime) {
    if (keyframes.size() <= 1) return;

    if (shouldReverse) {
        elapsedTime -= deltaTime;
    }
    else {
        elapsedTime += deltaTime;
    }

    float totalTime = keyframes.back().time;

    if (elapsedTime > totalTime) {
        if (reverse) {
            elapsedTime = totalTime;
            shouldReverse = true;
        }
        else if (loop) {
            elapsedTime = fmod(elapsedTime, totalTime);
        }
        else {
            elapsedTime = totalTime;
        }
    }
    else if (elapsedTime < 0.0f) {
        if (reverse) {
            elapsedTime = 0.0f;
            shouldReverse = false;
        }
        else if (loop) {
            elapsedTime = totalTime + fmod(elapsedTime, totalTime);
        }
        else {
            elapsedTime = 0.0f;
        }
    }

    Keyframe* prev = nullptr;
    Keyframe* next = nullptr;

    for (size_t i = 0; i < keyframes.size(); i++) {
        if (keyframes[i].time >= elapsedTime) {
            next = &keyframes[i];
            prev = (i > 0) ? &keyframes[i - 1] : &keyframes.back();
            break;
        }
    }

    if (!next) {
        prev = &keyframes.back();
        next = &keyframes.front();
    }

    if (prev && next) {
        float t = 0.0f;

        if (next == &keyframes.front() && elapsedTime > keyframes.back().time) {
            float wrappedTime = elapsedTime - keyframes.back().time;
            t = wrappedTime / (keyframes.front().time + totalTime - keyframes.back().time);
        }
        else {
            if (prev->time != next->time) {
                t = (elapsedTime - prev->time) / (next->time - prev->time);
            }
        }

        if (!isRotation) {
            currentState = glm::mix(prev->value, next->value, t);
        }
        else {
            if (keyframes.size() == 2) {
                glm::vec3 prevEuler = prev->value;
                glm::vec3 nextEuler = next->value;

                glm::vec3 interpolatedEuler = glm::mix(prevEuler, nextEuler, t);
                glm::quat interpolatedQuat = glm::quat(glm::radians(interpolatedEuler));
                currentState = glm::degrees(glm::eulerAngles(interpolatedQuat));
            }
            else {
                glm::quat prevQuat = glm::quat(glm::radians(prev->value));
                glm::quat nextQuat = glm::quat(glm::radians(next->value));
                glm::quat interpolatedQuat = glm::slerp(prevQuat, nextQuat, t);
                currentState = glm::degrees(glm::eulerAngles(interpolatedQuat));
            }
        }
    }
    else {
        currentState = keyframes.empty() ? initialState : keyframes.back().value;
    }
}

void Animation::AnimationProperty::reset() {
    elapsedTime = 0.0f;
    currentState = keyframes.empty() ? initialState : keyframes.front().value;
}

void Animation::addPositionKey(const glm::vec3& value, double time) {
    translation.addKeyframe(value, time);
}

void Animation::addRotationKey(const glm::vec3& value, double time) {
    rotation.addKeyframe(value, time);
}

void Animation::addScaleKey(const glm::vec3& value, double time) {
    scale.addKeyframe(value, time);
}

void Animation::setPositionProperties(bool reverse, bool loop) {
    translation.setProperties(reverse, loop);
}

void Animation::setRotationProperties(bool reverse, bool loop) {
    rotation.setProperties(reverse, loop);
}

void Animation::setScaleProperties(bool reverse, bool loop) {
    scale.setProperties(reverse, loop);
}

void Animation::update(double deltaTime) {
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
