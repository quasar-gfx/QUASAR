#ifndef GUI_MANAGER_H
#define GUI_MANAGER_H

#include <functional>

class GUIManager {
public:
    using GuiCallback = std::function<void(double now, double dt)>;

    void onRender(GuiCallback callback) { guiCallback = callback; }

    virtual void predraw() = 0;
    virtual void postdraw() = 0;

    void draw(double now, double dt) {
        predraw();

        if (guiCallback) {
            guiCallback(now, dt);
        }

        postdraw();
    }

private:
    GuiCallback guiCallback;
};

#endif // GUI_MANAGER_H
