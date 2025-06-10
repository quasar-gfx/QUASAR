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
    std::string name;

    Node* parent = nullptr;
    Entity* entity = nullptr;
    std::vector<Node*> children;

    Animation* animation = nullptr;

    bool frustumCulled = true;
    bool visible = true;

    GLenum primativeType = GL_TRIANGLES;
    float pointSize = 5.0f;

    bool wireframe = false;
    float wireframeLineWidth = 1.0f;

    Material* overrideMaterial = nullptr;

    std::vector<int> meshIndices;

    Node();
    Node(const std::string& name);
    Node(Entity* entity);
    Node(const std::string& name, Entity* entity);

    Node* findNodeByName(const std::string& name);

    int getID() const;
    void setEntity(Entity* entity);
    void setName(const std::string& name);
    void addChildNode(Node* node);

    void setPosition(const glm::vec3& position);
    void setRotationQuat(const glm::quat& quat);
    void setRotationEuler(const glm::vec3& euler, bool radians = false);
    void setScale(const glm::vec3& scale);

    virtual glm::vec3 getPosition() const;
    glm::quat getRotationQuat() const;
    glm::vec3 getRotationEuler(bool radians = false) const;
    glm::vec3 getScale() const;

    void setTransformParentFromLocal(const glm::mat4& pose);
    void setTransformLocalFromParent(const glm::mat4& view);

    const glm::mat4 getTransformParentFromLocal() const;
    const glm::mat4 getTransformLocalFromParent() const;
    const glm::mat4 getTransformLocalFromWorld() const;

    const glm::mat4 getTransformAnimation() const;

    void updateAnimations(double dt);

protected:
    uint32_t ID;
    glm::vec3 position = glm::vec3(0.0f);
    glm::quat rotationQuat = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 scale = glm::vec3(1.0f);

    static uint32_t nextID;
};

} // namespace quasar

#endif // NODE_H
