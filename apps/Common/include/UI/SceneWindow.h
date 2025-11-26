#ifndef SCENE_WINDOW_H
#define SCENE_WINDOW_H

#include <string>

#include <glm/glm.hpp>
#include <imgui/imgui.h>

#include <Scene.h>
#include <Primitives/Node.h>
#include <Primitives/Entity.h>

namespace quasar {

class SceneWindow {
public:
    bool visible = false;

    SceneWindow(Scene& scene, const ImVec2 size, ImGuiWindowFlags flags = ImGuiWindowFlags_None)
        : scene(scene)
        , savedSkybox(scene.skybox)
        , size(size)
        , flags(flags)
    {}

    void draw(double now, double dt) {
        if (!visible) {
            return;
        }

        ImGui::SetNextWindowSize(size, ImGuiCond_FirstUseEver);
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImVec2 pos = ImVec2(vp->WorkPos.x + vp->WorkSize.x - size.x - 10.0f, vp->WorkPos.y + 10.0f);
        ImGui::SetNextWindowPos(pos, ImGuiCond_FirstUseEver);
        ImGui::Begin("Scene Settings", &visible, flags);

        // Background Section
        if (ImGui::TreeNodeEx("Background", ImGuiTreeNodeFlags_SpanAvailWidth)) {
            if (ImGui::Checkbox("Show Sky Box", &showSkyBox)) {
                scene.skybox = showSkyBox ? savedSkybox : nullptr;
            }

            if (ImGui::Button("Change Background Color", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
                ImGui::OpenPopup("Background Color Popup");
            }
            if (ImGui::BeginPopup("Background Color Popup")) {
                ImGui::ColorPicker3("Background Color", (float*)&scene.backgroundColor);
                ImGui::EndPopup();
            }
            ImGui::TreePop();
        }

        ImGui::Separator();

        // Lights Section
        if (ImGui::TreeNodeEx("Lights", ImGuiTreeNodeFlags_SpanAvailWidth)) {
            // Ambient Light
            if (scene.ambientLight) {
                if (ImGui::TreeNodeEx((void*)scene.ambientLight, ImGuiTreeNodeFlags_SpanAvailWidth, "Ambient Light")) {
                    ImGui::ColorEdit3("Color##Ambient", (float*)&scene.ambientLight->color);
                    ImGui::DragFloat("Intensity##Ambient", &scene.ambientLight->intensity, 0.05f, 0.0f, 10.0f);
                    ImGui::TreePop();
                }
            }
            else {
                ImGui::TextDisabled("No Ambient Light");
            }

            // Directional Light
            if (scene.directionalLight) {
                if (ImGui::TreeNodeEx((void*)scene.directionalLight, ImGuiTreeNodeFlags_SpanAvailWidth, "Directional Light")) {
                    ImGui::ColorEdit3("Color##Directional", (float*)&scene.directionalLight->color);
                    ImGui::DragFloat("Intensity##Directional", &scene.directionalLight->intensity, 0.05f, 0.0f, 10.0f);
                    ImGui::DragFloat3("Direction##Directional", (float*)&scene.directionalLight->direction, 0.1f, -5.0f, 5.0f);
                    ImGui::DragFloat("Distance##Directional", &scene.directionalLight->distance, 0.1f, 0.0f, 10000.0f);
                    ImGui::TreePop();
                }
            }
            else {
                ImGui::TextDisabled("No Directional Light");
            }

            // Point Lights
            if (!scene.pointLights.empty()) {
                if (ImGui::TreeNodeEx("Point Lights", ImGuiTreeNodeFlags_SpanAvailWidth)) {
                    for (size_t i = 0; i < scene.pointLights.size(); ++i) {
                        PointLight* pl = scene.pointLights[i];
                        std::string label = "Point Light " + std::to_string(i);
                        if (ImGui::TreeNodeEx((void*)pl, ImGuiTreeNodeFlags_SpanAvailWidth, "%s", label.c_str())) {
                            ImGui::ColorEdit3(("Color##Point" + std::to_string(i)).c_str(), (float*)&pl->color);
                            ImGui::DragFloat(("Intensity##Point" + std::to_string(i)).c_str(), &pl->intensity, 0.1f, 0.0f, 1000.0f);
                            ImGui::DragFloat3(("Position##Point" + std::to_string(i)).c_str(), (float*)&pl->position, 0.01f);
                            ImGui::DragFloat(("Constant##Point" + std::to_string(i)).c_str(), &pl->constant, 0.01f, 0.0f, 10.0f);
                            ImGui::DragFloat(("Linear##Point" + std::to_string(i)).c_str(), &pl->linear, 0.001f, 0.0f, 1.0f);
                            ImGui::DragFloat(("Quadratic##Point" + std::to_string(i)).c_str(), &pl->quadratic, 0.001f, 0.0f, 1.0f);
                            ImGui::Checkbox(("Debug##Point" + std::to_string(i)).c_str(), &pl->debug);
                            ImGui::TreePop();
                        }
                    }
                    ImGui::TreePop();
                }
            }
            else {
                ImGui::TextDisabled("No Point Lights");
            }
            ImGui::TreePop();
        }

        ImGui::Separator();

        // Node Graph Section
        if (ImGui::TreeNodeEx("Node Graph", ImGuiTreeNodeFlags_SpanAvailWidth)) {
            for (auto* child : scene.children) {
                displayNode(child);
            }
            ImGui::TreePop();
        }

        ImGui::End();
    }

private:
    ImGuiWindowFlags flags;
    const ImVec2 size;

    bool showSkyBox = true;

    Scene& scene;
    SkyBox* savedSkybox = nullptr;

    void displayNode(Node* node) {
        ImGuiTreeNodeFlags tflags = ImGuiTreeNodeFlags_SpanAvailWidth;

        std::string label = node->getName().empty() ? (std::string("Node ") + std::to_string(node->getID())) : node->getName();
        bool open = ImGui::TreeNodeEx((void*)node, tflags, "Node: %s", label.c_str());
        if (open) {
            ImGui::Checkbox(("Visible##" + std::to_string(node->getID())).c_str(), &node->visible);

            glm::vec3 pos = node->getPosition();
            if (ImGui::DragFloat3(("Position##" + std::to_string(node->getID())).c_str(), (float*)&pos, 0.01f)) {
                node->setPosition(pos);
            }
            glm::vec3 rot = node->getRotationEuler();
            if (ImGui::DragFloat3(("Rotation##" + std::to_string(node->getID())).c_str(), (float*)&rot, 0.1f)) {
                node->setRotationEuler(rot);
            }
            glm::vec3 scl = node->getScale();
            if (ImGui::DragFloat3(("Scale##" + std::to_string(node->getID())).c_str(), (float*)&scl, 0.01f)) {
                node->setScale(scl);
            }

            for (auto* e : node->entities) {
                std::string name = e->getName().empty() ? (std::string("Entity ") + std::to_string(e->getID())) : e->getName();
                ImGuiTreeNodeFlags leaf = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_Bullet | ImGuiTreeNodeFlags_SpanAvailWidth;
                ImGui::TreeNodeEx((void*)e, leaf, "Mesh: %s", name.c_str());
            }

            for (auto* child : node->children) {
                displayNode(child);
            }

            ImGui::TreePop();
        }
    }
};

} // namespace quasar

#endif // SCENE_WINDOW_H
