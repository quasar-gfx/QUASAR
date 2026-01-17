#ifndef SCREENSHOT_WINDOW_H
#define SCREENSHOT_WINDOW_H

#include <imgui/imgui.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>

#include <Path.h>
#include <Recorder.h>

namespace quasar {

class ScreenshotWindow {
public:
    bool visible = false;

    ScreenshotWindow(Recorder& recorder, const ImVec2 size, const Path& outputPath,
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

        const char* formatNames[] = { "PNG", "JPG", "EXR" };
        ImGui::Combo("Format", &selectedFormatIndex, formatNames, IM_ARRAYSIZE(formatNames));

        ImGui::Separator();

        if (ImGui::Button("Capture Current Frame")) {
            currentFormat = static_cast<ScreenshotFormat>(selectedFormatIndex);

            std::string time = std::to_string(static_cast<int>(now * 1000.0f));

            std::string extension;
            switch (currentFormat) {
                case ScreenshotFormat::PNG: extension = ".png"; break;
                case ScreenshotFormat::JPG: extension = ".jpg"; break;
                case ScreenshotFormat::EXR: extension = ".exr"; break;
            }

            Path fullPath = (outputPath / fileNameBase).appendToName("_" + time).withExtension(extension);
            recorder.saveScreenshotToFile(fullPath);

            spdlog::info("Screenshot saved to {}", fullPath.str());
        }

        ImGui::End();
    }

private:
    enum class ScreenshotFormat {
        PNG,
        JPG,
        EXR
    };

    ImGuiWindowFlags flags;
    const ImVec2 size;

    Recorder& recorder;
    const Path& outputPath;

    int selectedFormatIndex = 0; // Default to 0 (PNG)
    ScreenshotFormat currentFormat = ScreenshotFormat::PNG;
    char fileNameBase[256] = "screenshot";
};

} // namespace quasar

#endif // SCREENSHOT_WINDOW_H
