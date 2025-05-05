#include <Primitives/Node.h>

using namespace quasar;

uint Node::nextID = 0;

Node::Node() {
    ID = nextID++;
    this->name = "Node" + std::to_string(ID);
}

Node::Node(const std::string &name) {
    ID = nextID++;
    this->name = name;
}

Node::Node(Entity* entity) {
    ID = nextID++;
    this->name = "Node" + std::to_string(ID);
    setEntity(entity);
}

Node::Node(const std::string &name, Entity* entity) {
    ID = nextID++;
    this->name = name;
    setEntity(entity);
}

Node* Node::findNodeByName(const std::string &name) {
    if (this->name == name) {
        return this;
    }

    for (Node* child : children) {
        Node* found = child->findNodeByName(name);
        if (found != nullptr) {
            return found;
        }
    }

    return nullptr;
}

int Node::getID() const {
    return ID;
}

void Node::setEntity(Entity* entity) {
    this->entity = entity;
    entity->parentNode = this;
}

void Node::setName(const std::string &name) {
    this->name = name;
}

void Node::addChildNode(Node* node) {
    children.push_back(node);
    node->parent = this;
}

void Node::setPosition(const glm::vec3 &position) {
    this->position = position;
}

void Node::setRotationQuat(const glm::quat &quat) {
    this->rotationQuat = quat;
}

void Node::setRotationEuler(const glm::vec3 &euler, bool radians) {
    glm::vec3 eulerCopy = euler;
    if (!radians) {
        eulerCopy = glm::radians(eulerCopy);
    }
    this->rotationQuat = glm::quat(eulerCopy);
}

void Node::setScale(const glm::vec3 &scale) {
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

const glm::mat4 Node::getTransformParentFromLocal() const {
    const glm::mat4& transform = glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotationQuat) * glm::scale(glm::mat4(1.0f), scale);
    return transform;
}

const glm::mat4 Node::getTransformLocalFromParent() const {
    const glm::mat4& transformInv = glm::scale(glm::mat4(1.0f), 1.0f/scale) * glm::mat4_cast(glm::conjugate(rotationQuat)) * glm::translate(glm::mat4(1.0f), -position);
    return transformInv;
}

const glm::mat4 Node::getTransformLocalFromWorld() const {
    glm::mat4 transformLocalFromWorld = getTransformLocalFromParent();

    Node* parent = this->parent;
    while (parent != nullptr) {
        // have to multiply in reverse order, since parents goes from child to root
        transformLocalFromWorld = transformLocalFromWorld * parent->getTransformLocalFromParent();
        parent = parent->parent;
    }

    return transformLocalFromWorld;
}

const glm::mat4 Node::getTransformAnimation() const {
    glm::mat4 transformAnimation = glm::mat4(1.0f);

    if (animation != nullptr) {
        transformAnimation = animation->getTransformation();
    }

    return transformAnimation;
}

void Node::updateAnimations(double dt) {
    if (animation != nullptr) {
        animation->update(dt);
    }

    if (entity != nullptr) {
        entity->updateAnimations(dt);
    }

    for (auto* child : children) {
        child->updateAnimations(dt);
    }
}
