#ifndef NODE_H
#define NODE_H

#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>

#include <Primitives/Entity.h>

#include <Animation.h>

namespace quasar {

class Node {
public:
    Node* parent = nullptr;
    std::vector<Entity*> entities;
    std::vector<Node*> children;

    bool visible = true;
    bool frustumCulled = true;

    bool isGLTF = false;

    GLenum primitiveType = GL_TRIANGLES;
    float pointSize = 5.0f;

    bool wireframe = false;
    float wireframeLineWidth = 1.0f;

    const Material* overrideMaterial = nullptr;

    Node();
    Node(const std::string& name);
    Node(Entity* entity);
    Node(const std::string& name, Entity* entity);

    Node* findNodeByName(const std::string& name);

    int getID() const;
    void addEntity(Entity* entity);
    void setName(const std::string& name);
    void addChildNode(Node* node);

    void setPosition(const glm::vec3& position);
    void setRotationQuat(const glm::quat& quat);
    void setRotationEuler(const glm::vec3& euler, bool radians = false);
    void setScale(const glm::vec3& scale);

    const std::string& getName() const { return name; }
    virtual glm::vec3 getPosition() const;
    glm::quat getRotationQuat() const;
    glm::vec3 getRotationEuler(bool radians = false) const;
    glm::vec3 getScale() const;

    const glm::mat4 getTransformParentFromLocal() const;
    const glm::mat4 getTransformLocalFromParent() const;
    const glm::mat4 getTransformLocalFromWorld() const;
    const glm::mat4 getTransformAnimation() const;

    void setTransformParentFromLocal(const glm::mat4& pose);
    void setTransformLocalFromParent(const glm::mat4& view);

    bool hasAnimation() const { return animation != nullptr; }
    std::shared_ptr<Animation> addAnimation();

    void updateAnimations(double dt);

protected:
    uint32_t ID;
    static uint32_t nextID;

    std::string name;

    glm::vec3 position = glm::vec3(0.0f);
    glm::quat rotationQuat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = glm::vec3(1.0f);

    std::shared_ptr<Animation> animation;
};

} // namespace quasar

#endif // NODE_H
