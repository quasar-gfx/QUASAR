#ifndef FRAME_CAPTURE_WINDOW_H
#define FRAME_CAPTURE_WINDOW_H

#include <imgui/imgui.h>
#include <glm/glm.hpp>

#include <Path.h>
#include <Recorder.h>

namespace quasar {

class FrameCaptureWindow {
public:
    bool visible = false;

    FrameCaptureWindow(Recorder& recorder, const glm::uvec2& size, const Path& outputPath,
                       ImGuiWindowFlags flags = ImGuiCond_FirstUseEver)
        : recorder(recorder)
        , size(size)
        , outputPath(outputPath)
        , flags(flags)
    {}

    void draw(double now, double dt) {
        if (!visible) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(size.x, size.y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10, 40), ImGuiCond_FirstUseEver);
        ImGui::Begin("Frame Capture", &visible);

        ImGui::Text("Base File Name:");
        ImGui::InputText("##base file name", fileNameBase, IM_ARRAYSIZE(fileNameBase));
        std::string time = std::to_string(static_cast<int>(now * 1000.0f));
        Path filename = (outputPath / fileNameBase).appendToName("." + time);

        ImGui::Checkbox("Save as HDR", &writeToHDR);

        ImGui::Separator();

        if (ImGui::Button("Capture Current Frame")) {
            recorder.saveScreenshotToFile(filename, writeToHDR);
        }

        ImGui::End();
    }

private:
    ImGuiWindowFlags flags;

    Recorder& recorder;
    const glm::uvec2& size;
    const Path& outputPath;

    bool writeToHDR = false;
    char fileNameBase[256] = "screenshot";
};

} // namespace quasar

#endif // FRAME_CAPTURE_WINDOW_H
