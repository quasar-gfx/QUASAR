#ifndef IMGUI_MANAGER_H
#define IMGUI_MANAGER_H

#ifndef __ANDROID__

#include <imgui/imgui.h>

#include <Windowing/GLFWWindow.h>
#include <GUI/GUIManager.h>

namespace quasar {

class ImGuiManager : public GUIManager {
public:
    ImGuiManager(std::shared_ptr<GLFWWindow> glfwWindow);
    ~ImGuiManager();

    void setStyle(float fontSize) const;

    void beginDrawing() const override;
    void endDrawing() const override;

private:
    float fontSize = 24.0f;
};

#endif

} // namespace quasar

#endif // IMGUI_MANAGER_H
