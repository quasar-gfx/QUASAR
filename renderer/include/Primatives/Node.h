#ifndef NODE_H
#define NODE_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <Primatives/Entity.h>
#include <vector>

class Node {
public:
    Node* parent = nullptr;
    Entity* entity = nullptr;
    std::vector<Node*> children;

    bool frustumCulled = true;
    bool wireframe = false;
    float wireframeLineWidth = 1.5f;
    bool visible = true;

    Node();
    Node(Entity* entity);

    int getID() const;
    void setEntity(Entity* entity);
    void addChildNode(Node* node);

    void setPosition(glm::vec3 position);
    void setRotationQuat(glm::quat quat);
    void setRotationEuler(glm::vec3 euler, bool radians = false);
    void setScale(glm::vec3 scale);

    virtual glm::vec3 getPosition() const;
    glm::quat getRotationQuat() const;
    glm::vec3 getRotationEuler(bool radians = false) const;
    glm::vec3 getScale() const;

    void setTransformParentFromLocal(const glm::mat4 &pose);
    void setTransformLocalFromParent(const glm::mat4 &view);

    glm::mat4 getTransformParentFromLocal() const;
    glm::mat4 getTransformLocalFromParent() const;
    glm::mat4 getTransformLocalFromWorld() const;

protected:
    unsigned int ID;
    glm::vec3 position = glm::vec3(0.0f);
    glm::quat rotationQuat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = glm::vec3(1.0f);

    static unsigned int nextID;
};

#endif // NODE_H
