#ifndef CAMERA_HEADER_H
#define CAMERA_HEADER_H

#include <glm/glm.hpp>
#include <imgui.h>

#include <Cameras/PerspectiveCamera.h>

namespace quasar {

class CameraHeader {
public:
    bool visible = true;

    CameraHeader(Camera& camera, const std::string& title = "Main Camera", bool readOnly = false)
        : camera(camera)
        , title(title)
        , readOnly(readOnly)
    {}

    void draw(double now, double dt) {
        if (!visible) {
            return;
        }

        if (ImGui::CollapsingHeader(title.c_str())) {
            if (readOnly) {
                ImGui::BeginDisabled();
            }

            float fovY = camera.getFovyDegrees();
            if (ImGui::DragFloat("FOV (deg)", &fovY, 0.1f, 1.0f, 170.0f)) {
                camera.setFovyDegrees(fovY);
            }
            float near = camera.getNear();
            if (ImGui::DragFloat("Near Plane (m)", &near, 0.01f, 0.01f, camera.getFar() - 0.1f, "%.2f")) {
                camera.setNear(near);
            }
            float far = camera.getFar();
            if (ImGui::DragFloat("Far Plane (m)", &far, 1.0f, camera.getNear() + 0.1f, 10000.0f, "%.1f")) {
                camera.setFar(far);
            }

            ImGui::Separator();

            glm::vec3 position = camera.getPosition();
            if (ImGui::DragFloat3("Position (m)", reinterpret_cast<float*>(&position), 0.01f)) {
                camera.setPosition(position);
            }
            glm::vec3 rotation = camera.getRotationEuler();
            if (ImGui::DragFloat3("Rotation (deg)", reinterpret_cast<float*>(&rotation), 0.1f)) {
                camera.setRotationEuler(rotation);
            }
            glm::quat rotationQuat = camera.getRotationQuat();
            ImGui::InputFloat4("Rotation (quat)", reinterpret_cast<float*>(&rotationQuat), "%.3f", ImGuiInputTextFlags_ReadOnly);
            ImGui::DragFloat("Speed (m/s)", &camera.movementSpeed, 0.05f, 0.1f, 20.0f);

            if (readOnly) {
                ImGui::EndDisabled();
            }
        }
    }

private:
    Camera& camera;

    const std::string title;
    bool readOnly;
};

} // namespace quasar

#endif // CAMERA_HEADER_H
