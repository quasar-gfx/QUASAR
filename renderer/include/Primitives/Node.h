#ifndef NODE_H
#define NODE_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <Primitives/Entity.h>
#include <vector>

class Node {
public:
    std::string name;

    Node* parent = nullptr;
    Entity* entity = nullptr;
    std::vector<Node*> children;

    bool frustumCulled = true;
    bool visible = true;

    GLenum primativeType = GL_TRIANGLES;
    float pointSize = 5.0f;

    bool wireframe = false;
    float wireframeLineWidth = 1.5f;

    Material* overrideMaterial = nullptr;

    std::vector<int> meshIndices;

    Node();
    Node(const std::string &name);
    Node(Entity* entity);
    Node(const std::string &name, Entity* entity);

    Node* findNodeByName(const std::string &name);

    int getID() const;
    void setEntity(Entity* entity);
    void setName(const std::string &name);
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

    const glm::mat4 getTransformParentFromLocal() const;
    const glm::mat4 getTransformLocalFromParent() const;
    const glm::mat4 getTransformLocalFromWorld() const;

    void setTransformAnimation(const glm::mat4 &transform);

    const glm::mat4 getTransformAnimation() const;

protected:
    uint32_t ID;
    glm::vec3 position = glm::vec3(0.0f);
    glm::quat rotationQuat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = glm::vec3(1.0f);

    glm::mat4 transformAnimation = glm::mat4(1.0f);

    static uint32_t nextID;
};

#endif // NODE_H
