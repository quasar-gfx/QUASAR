#ifndef IMGUI_MANAGER_H
#define IMGUI_MANAGER_H

#ifndef __ANDROID__

#include <memory>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <imgui/backends/imgui_impl_glfw.h>

#include <Windowing/GLFWWindow.h>
#include <GUI/GUIManager.h>

namespace quasar {

class ImGuiManager : public GUIManager {
public:
    ImGuiManager(std::shared_ptr<GLFWWindow> glfwWindow);
    ~ImGuiManager();

    void setStyle() const;

    void beginDrawing() const override;
    void endDrawing() const override;
};

#endif

} // namespace quasar

#endif // IMGUI_MANAGER_H
