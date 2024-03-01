#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include <Entity.h>

class Scene {
public:
    std::vector<Node*> children;

    void addChildNode(Node* node) {
        children.push_back(node);
    }
};

#endif // SCENE_H
