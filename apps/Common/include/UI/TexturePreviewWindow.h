#ifndef TEXTURE_PREVIEW_WINDOW_H
#define TEXTURE_PREVIEW_WINDOW_H

#include <imgui/imgui.h>
#include <glm/glm.hpp>

#include <Texture.h>

namespace quasar {

class TexturePreviewWindow {
public:
    bool visible = false;

    TexturePreviewWindow(const std::string& title, const Texture& texture, const glm::uvec2& initialSize,
                         ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse)
        : title(title)
        , texture(texture)
        , initialSize(initialSize)
        , flags(flags)
    {}

    void draw(double now, double dt) {
        if (!visible) {
            return;
        }

        ImGui::SetNextWindowPos(ImVec2(10, 40), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(initialSize.x, initialSize.y), ImGuiCond_FirstUseEver);
        ImGui::Begin(title.c_str(), &visible, flags);

        // Maintain aspect ratio
        float aspect = static_cast<float>(texture.width) / static_cast<float>(texture.height);
        ImVec2 drawSize = ImGui::GetContentRegionAvail();

        if (drawSize.x / drawSize.y > aspect) {
            drawSize.x = drawSize.y * aspect;
        }
        else {
            drawSize.y = drawSize.x / aspect;
        }

        ImGui::Image((void*)(intptr_t)texture, drawSize, ImVec2(0, 1), ImVec2(1, 0));
        ImGui::End();
    }

private:
    ImGuiWindowFlags flags;

    const std::string title;
    const Texture& texture;
    const glm::uvec2& initialSize;
};

} // namespace quasar

#endif // TEXTURE_PREVIEW_WINDOW_H
