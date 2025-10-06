#ifndef FRAME_RATE_WINDOW_H
#define FRAME_RATE_WINDOW_H

#include <imgui/imgui.h>

namespace quasar {

class FrameRateWindow {
public:
    bool visible = true;

    FrameRateWindow(ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)
        : flags(flags)
    {}

    void draw(double now, double dt) {
        if (!visible) {
            return;
        }

        ImGui::SetNextWindowPos(ImVec2(10, 40), ImGuiCond_FirstUseEver);
        ImGui::Begin("", 0, flags);
        ImGui::Text("%.1f FPS (%.3f ms/frame)", ImGui::GetIO().Framerate, 1000.0f / ImGui::GetIO().Framerate);
        ImGui::End();
    }

private:
    ImGuiWindowFlags flags;
};

} // namespace quasar

#endif // FRAME_RATE_WINDOW_H
