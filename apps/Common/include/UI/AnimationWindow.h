#ifndef ANIMATION_WINDOW_H
#define ANIMATION_WINDOW_H

#include <imgui/imgui.h>
#include <glm/glm.hpp>

namespace quasar {

class AnimationWindow {
public:
    bool visible = false;

    AnimationWindow(glm::vec2 size, ImGuiWindowFlags flags = ImGuiWindowFlags_None)
        : size(size)
        , flags(flags)
    {}

    void setPlaying(bool playing) { runAnimations = playing; }
    bool isPlaying() const { return runAnimations; }

    double getAnimationIntervalMs() const {
        int fr = animationFramerates[animationFramerateIndex];
        return fr > 0 ? (MILLISECONDS_IN_SECOND / static_cast<double>(fr)) : 0.0;
    }

    double getCurrentTime() const { return currentTime; }
    void resetTime() { currentTime = 0.0; }

    void draw(double now, double dt) {
        if (!visible) return;

        if (runAnimations) {
            currentTime += dt;
        }

        ImGui::SetNextWindowSize(ImVec2(size.x, size.y), ImGuiCond_FirstUseEver);
        ImGuiViewport* vp = ImGui::GetMainViewport();
        ImVec2 pos = ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.4f, vp->WorkPos.y + 90.0f);
        ImGui::SetNextWindowPos(pos, ImGuiCond_FirstUseEver);
        ImGui::Begin("Animations", &visible, flags);

        ImGui::TextColored(ImVec4(0,1,0,1), "Current Time: %.3f s", currentTime);

        ImGui::Separator();

        ImGui::Text("Animation Framerate:");
        ImGui::Combo("", &animationFramerateIndex, animationFramerateLabels, IM_ARRAYSIZE(animationFramerateLabels));

        ImGui::Separator();

        if (!runAnimations) {
            if (ImGui::Button("Play", ImVec2(ImGui::GetContentRegionAvail().x, 0))) { runAnimations = true; }
        }
        else {
            if (ImGui::Button("Pause", ImVec2(ImGui::GetContentRegionAvail().x, 0))) { runAnimations = false; }
        }

        ImGui::End();
    }

private:
    ImGuiWindowFlags flags;
    glm::vec2 size;

    bool runAnimations = false;
    int animationFramerates[6] = {1, 5, 10, 24, 30, 60};
    const char* animationFramerateLabels[6] = {"1 FPS", "5 FPS", "10 FPS", "24 FPS", "30 FPS", "60 FPS"};
    int animationFramerateIndex = 4; // default 30 FPS
    double currentTime = 0.0;
};

} // namespace quasar

#endif // ANIMATION_WINDOW_H
