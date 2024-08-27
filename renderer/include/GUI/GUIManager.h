#ifndef GUI_MANAGER_H
#define GUI_MANAGER_H

#include <functional>

class GUIManager {
public:
    using GuiCallback = std::function<void(double now, double dt)>;

    void onRender(GuiCallback callback) { guiCallback = callback; }

    virtual void beginDrawing() const = 0;
    virtual void endDrawing() const = 0;

    void draw(double now, double dt) const {
        beginDrawing();

        if (guiCallback) {
            guiCallback(now, dt);
        }

        endDrawing();
    }

private:
    GuiCallback guiCallback;
};

#endif // GUI_MANAGER_H
