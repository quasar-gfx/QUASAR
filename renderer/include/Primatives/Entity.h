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

    int getID() const { return ID; }

    virtual void bindMaterial(const Scene &scene, const Camera &camera, const glm::mat4 &model, const Material* overrideMaterial = nullptr) = 0;
    virtual unsigned int draw(const Scene &scene, const Camera &camera, const glm::mat4 &model, bool frustumCull = true, const Material* overrideMaterial = nullptr) = 0;
    virtual unsigned int draw(const Scene &scene, const Camera &camera, const glm::mat4 &model, const BoundingSphere &boundingSphere, const Material* overrideMaterial = nullptr) = 0;

    virtual EntityType getType() const { return EntityType::EMPTY; }

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
    bool visible = true;

    int getID() const { return ID; }

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

    void setPosition(glm::vec3 position) {
        this->position = position;
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

    glm::vec3 getPosition() const {
        return position;
    }

    glm::quat getRotationQuat() const {
        return rotationQuat;
    }

    glm::vec3 getRotationEuler() const {
        glm::vec3 euler = glm::eulerAngles(rotationQuat);
        return euler;
    }

    glm::vec3 getScale() const {
        return scale;
    }

    void setTransformParentFromLocal(glm::mat4 transform) {
        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(transform, scale, rotationQuat, position, skew, perspective);
    }

    glm::mat4 getTransformParentFromLocal() const {
        return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotationQuat) * glm::scale(glm::mat4(1.0f), scale);
    }

    glm::mat4 getTransformLocalFromParent() const {
        return glm::scale(glm::mat4(1.0f), 1.0f/scale) * glm::mat4_cast(glm::conjugate(rotationQuat)) * glm::translate(glm::mat4(1.0f), -position);
    }

    glm::mat4 getTransformLocalFromWorld() const {
        glm::mat4 transformLocalFromWorld = getTransformLocalFromParent();

        Node* parent = this->parent;
        while (parent != nullptr) {
            // have to multiply in reverse order, since parents goes from child to root
            transformLocalFromWorld = transformLocalFromWorld * parent->getTransformLocalFromParent();
            parent = parent->parent;
        }

        return transformLocalFromWorld;
    }

protected:
    unsigned int ID;

    glm::vec3 position = glm::vec3(0.0f);
    glm::quat rotationQuat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = glm::vec3(1.0f);

    static unsigned int nextID;
};

#endif // NODE_H
