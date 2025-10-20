#ifndef PLATFORM_ANDROID

#include <imgui/backends/imgui_impl_opengl3.h>
#include <imgui/backends/imgui_impl_glfw.h>

#include <GUI/ImGuiManager.h>
#include <GUI/fonts/trebucbd.h>

using namespace quasar;

ImGuiManager::ImGuiManager(std::shared_ptr<GLFWWindow> glfwWindow) {
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    // Setup ImGui OpenGL backend
    ImGui_ImplGlfw_InitForOpenGL(glfwWindow->window, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    setStyle(fontSize);
}

ImGuiManager::~ImGuiManager() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiManager::setStyle(float fontSize) const {
    auto& fonts = ImGui::GetIO().Fonts;
    ImFontConfig cfg;
    cfg.FontDataOwnedByAtlas = false;
    fonts->AddFontFromMemoryTTF(
        static_cast<void*>(trebucbd_ttf),
        static_cast<int>(trebucbd_ttf_len),
        fontSize,
        &cfg
    );

    auto& style = ImGui::GetStyle();
    style.TabRounding = 5;
    style.WindowRounding = 8;
}

void ImGuiManager::beginDrawing() const {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiManager::endDrawing() const {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

#endif
