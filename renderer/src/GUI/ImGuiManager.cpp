#include <GUI/ImGuiManager.h>

ImGuiManager::ImGuiManager(std::shared_ptr<GLFWWindow> glfwWindow) {
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    // Setup ImGui OpenGL backend
    ImGui_ImplGlfw_InitForOpenGL(glfwWindow->window, true);
    ImGui_ImplOpenGL3_Init("#version 410");

    setStyle();
}

ImGuiManager::~ImGuiManager() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiManager::setStyle() const {
    auto& fonts = ImGui::GetIO().Fonts;
    fonts->AddFontFromFileTTF("../assets/fonts/trebucbd.ttf", 24.0f);

    auto& style = ImGui::GetStyle();
    style.TabRounding = 5;
    style.WindowRounding = 8;
}

void ImGuiManager::beginDrawing() const {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
}

void ImGuiManager::endDrawing() const {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
