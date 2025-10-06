#ifndef RECORD_WINDOW_H
#define RECORD_WINDOW_H

#include <imgui/imgui.h>
#include <glm/glm.hpp>

#include <Path.h>
#include <Recorder.h>

namespace quasar {

class RecordWindow {
public:
    bool visible = false;

    RecordWindow(Recorder& recorder, const glm::uvec2& initialSize, const Path& outputPath,
                 ImGuiWindowFlags flags = ImGuiCond_FirstUseEver)
        : recorder(recorder)
        , initialSize(initialSize)
        , outputPath(outputPath)
        , flags(flags)
    {}

    bool isRecording() const { return recording; }

    void draw(double now, double dt) {
        if (!visible) {
            return;
        }

        ImGui::SetNextWindowSize(ImVec2(initialSize.x, initialSize.y), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10, 40), ImGuiCond_FirstUseEver);
        ImGui::Begin("Record", &visible);

        if (isRecording()) {
            ImGui::TextColored(ImVec4(1,0,0,1), "Recording in progress...");
        }

        ImGui::Text("Output Directory:");
        ImGui::InputText("##output directory", recordingDirBase, IM_ARRAYSIZE(recordingDirBase));

        ImGui::Text("FPS:");
        if (ImGui::InputInt("##fps", &recorder.targetFrameRate)) {
            recorder.setTargetFrameRate(recorder.targetFrameRate);
        }

        ImGui::Text("Format:");
        if (ImGui::Combo("##format", &recordingFormatIndex, recorder.getFormatCStrArray(), recorder.getFormatCount())) {
            Recorder::OutputFormat selectedFormat = Recorder::OutputFormat::MP4;
            switch (recordingFormatIndex) {
                case 0: selectedFormat = Recorder::OutputFormat::MP4; break;
                case 1: selectedFormat = Recorder::OutputFormat::PNG; break;
                case 2: selectedFormat = Recorder::OutputFormat::JPG; break;
                default: break;
            }
            recorder.setFormat(selectedFormat);
        }

        if (ImGui::Button("Start")) {
            recording = true;
            std::string time = std::to_string(static_cast<int>(now * 1000.0f));
            Path recordingDir = (outputPath / recordingDirBase).appendToName("." + time);
            recorder.setOutputPath(recordingDir);
            recorder.start();
        }
        ImGui::SameLine();
        if (ImGui::Button("Stop")) {
            recorder.stop();
            recording = false;
        }

        ImGui::End();
    }

private:
    ImGuiWindowFlags flags;

    bool recording = false;

    Recorder& recorder;
    const glm::uvec2& initialSize;
    const Path& outputPath;

    int recordingFormatIndex = 0;
    char recordingDirBase[256] = "recordings";
};

} // namespace quasar

#endif // RECORD_WINDOW_H
