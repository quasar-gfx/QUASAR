#include <Primatives/Node.h>

unsigned int Node::nextID = 0;

Node::Node() {
    ID = nextID++;
}

Node::Node(Entity* entity) {
    ID = nextID++;
    setEntity(entity);
}

int Node::getID() const {
    return ID;
}

void Node::setEntity(Entity* entity) {
    this->entity = entity;
    entity->parentNode = this;
}

void Node::addChildNode(Node* node) {
    children.push_back(node);
    node->parent = this;
}

void Node::setPosition(glm::vec3 position) {
    this->position = position;
}

void Node::setRotationQuat(glm::quat quat) {
    this->rotationQuat = quat;
}

void Node::setRotationEuler(glm::vec3 euler, bool radians) {
    if (!radians) {
        euler = glm::radians(euler);
    }
    this->rotationQuat = glm::quat(euler);
}

void Node::setScale(glm::vec3 scale) {
    this->scale = scale;
}

glm::vec3 Node::getPosition() const {
    return position;
}

glm::quat Node::getRotationQuat() const {
    return rotationQuat;
}

glm::vec3 Node::getRotationEuler(bool radians) const {
    glm::vec3 euler = glm::eulerAngles(rotationQuat);
    if (!radians) {
        euler = glm::degrees(euler);
    }
    return euler;
}

glm::vec3 Node::getScale() const {
    return scale;
}

void Node::setTransformParentFromLocal(const glm::mat4 &pose) {
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(pose, scale, rotationQuat, position, skew, perspective);
}

void Node::setTransformLocalFromParent(const glm::mat4 &view) {
    glm::mat3 rotScaleMat = glm::mat3(view);

    glm::mat3 rotationMatrix = glm::transpose(rotScaleMat);

    position = -rotationMatrix * glm::vec3(view[3]);
    rotationQuat = glm::quat_cast(rotationMatrix);

    scale.x = glm::length(glm::vec3(view[0]));
    scale.y = glm::length(glm::vec3(view[1]));
    scale.z = glm::length(glm::vec3(view[2]));
}

glm::mat4 Node::getTransformParentFromLocal() const {
    return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotationQuat) * glm::scale(glm::mat4(1.0f), scale);
}

glm::mat4 Node::getTransformLocalFromParent() const {
    return glm::scale(glm::mat4(1.0f), 1.0f/scale) * glm::mat4_cast(glm::conjugate(rotationQuat)) * glm::translate(glm::mat4(1.0f), -position);
}

glm::mat4 Node::getTransformLocalFromWorld() const {
    glm::mat4 transformLocalFromWorld = getTransformLocalFromParent();

    Node* parent = this->parent;
    while (parent != nullptr) {
        // have to multiply in reverse order, since parents goes from child to root
        transformLocalFromWorld = transformLocalFromWorld * parent->getTransformLocalFromParent();
        parent = parent->parent;
    }

    return transformLocalFromWorld;
}
