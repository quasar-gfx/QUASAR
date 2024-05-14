#ifndef IMGUI_MANAGER_H
#define IMGUI_MANAGER_H

#include <memory>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <imgui/backends/imgui_impl_glfw.h>

#include <Windowing/GLFWWindow.h>
#include <GUI/GUIManager.h>

class ImGuiManager : public GUIManager {
public:
    ImGuiManager(std::shared_ptr<GLFWWindow> glfwWindow);
    ~ImGuiManager();

    void setStyle();

    void predraw() override;
    void postdraw() override;
};

#endif // IMGUI_MANAGER_H
