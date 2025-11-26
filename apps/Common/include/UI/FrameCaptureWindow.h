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

    FrameCaptureWindow(Recorder& recorder, const ImVec2 size, const Path& outputPath,
                       ImGuiWindowFlags flags = ImGuiWindowFlags_None)
        : recorder(recorder)
        , size(size)
        , outputPath(outputPath)
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
        ImGui::Begin("Screenshot", &visible, flags);

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
    const ImVec2 size;

    Recorder& recorder;
    const Path& outputPath;

    bool writeToHDR = false;
    char fileNameBase[256] = "screenshot";
};

} // namespace quasar

#endif // FRAME_CAPTURE_WINDOW_H
