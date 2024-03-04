#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include <Entity.h>
#include <Lights.h>
#include <CubeMap.h>

class Scene {
public:
    CubeMap* skyBox = nullptr;
    AmbientLight* ambientLight = nullptr;
    DirectionalLight* directionalLight = nullptr;

    std::vector<Node*> children;

    void addChildNode(Node* node) {
        children.push_back(node);
    }

    void setSkyBox(CubeMap* skyBox) {
        this->skyBox = skyBox;
    }

    void setAmbientLight(AmbientLight* ambientLight) {
        this->ambientLight = ambientLight;
    }

    void setDirectionalLight(DirectionalLight* directionalLight) {
        this->directionalLight = directionalLight;
    }
};

#endif // SCENE_H
