#ifndef NODE_H
#define NODE_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <vector>

#include <Materials/Material.h>
#include <Culling/AABB.h>
#include <Culling/BoundingSphere.h>

class Node;
class Scene;
class Camera;

enum class EntityType {
    EMPTY,
    MESH
};

class Entity {
public:
    Node* parentNode = nullptr;

    AABB aabb;

    explicit Entity() : ID(nextID++), aabb() {}

    int getID() { return ID; }

    virtual void bindSceneAndCamera(Scene &scene, Camera &camera, const glm::mat4 &model, Material* overrideMaterial = nullptr) = 0;
    virtual unsigned int draw(Scene &scene, Camera &camera, const glm::mat4 &model, bool frustumCull = true, Material* overrideMaterial = nullptr) = 0;
    virtual unsigned int draw(Scene &scene, Camera &camera, const glm::mat4 &model, const BoundingSphere &boundingSphere, Material* overrideMaterial = nullptr) = 0;

    virtual EntityType getType() { return EntityType::EMPTY; }

private:
    unsigned int ID;

    static unsigned int nextID;
};

class Node {
public:
    Node* parent = nullptr;
    Entity* entity = nullptr;
    std::vector<Node*> children;

    bool frustumCulled = true;

    int getID() { return ID; }

    explicit Node() {
        ID = nextID++;
    }

    explicit Node(Entity* entity) {
        ID = nextID++;
        setEntity(entity);
    }

    void setEntity(Entity* entity) {
        this->entity = entity;
        entity->parentNode = this;
    }

    void addChildNode(Node* node) {
        children.push_back(node);
        node->parent = this;
    }

    void setTranslation(glm::vec3 translation) {
        this->translation = translation;
    }

    void setRotationQuat(glm::quat quat) {
        this->rotationQuat = quat;
    }

    void setRotationEuler(glm::vec3 euler) {
        this->rotationQuat = glm::quat(glm::radians(euler));
    }

    void setScale(glm::vec3 scale) {
        this->scale = scale;
    }

    glm::vec3 getTranslation() {
        return translation;
    }

    glm::quat getRotationQuat() {
        return rotationQuat;
    }

    glm::vec3 getRotationEuler() {
        glm::vec3 euler = glm::eulerAngles(rotationQuat);
        return euler;
    }

    glm::vec3 getScale() {
        return scale;
    }

    void setTransformParentFromLocal(glm::mat4 transform) {
        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(transform, scale, rotationQuat, translation, skew, perspective);
    }

    glm::mat4 getTransformParentFromLocal() {
        return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotationQuat) * glm::scale(glm::mat4(1.0f), scale);
    }

    glm::mat4 getTransformLocalFromParent() {
        return glm::scale(glm::mat4(1.0f), 1.0f/scale) * glm::mat4_cast(glm::conjugate(rotationQuat)) * glm::translate(glm::mat4(1.0f), -translation);
    }

    glm::mat4 getTransformLocalFromWorld() {
        glm::mat4 transformLocalFromWorld = getTransformLocalFromParent();

        Node* parent = this->parent;
        while (parent != nullptr) {
            // have to multiply in reverse order, since parents goes from child to root
            transformLocalFromWorld = transformLocalFromWorld * parent->getTransformLocalFromParent();
            parent = parent->parent;
        }

        return transformLocalFromWorld;
    }

private:
    unsigned int ID;

    glm::vec3 translation = glm::vec3(0.0f);
    glm::quat rotationQuat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = glm::vec3(1.0f);

    static unsigned int nextID;
};

#endif // NODE_H
