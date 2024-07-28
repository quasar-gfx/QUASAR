#ifndef WINDOW_H
#define WINDOW_H

#include <functional>

struct Mouse {
    bool LEFT_PRESSED = false;
    bool MIDDLE_PRESSED = false;
    bool RIGHT_PRESSED = false;
};

struct Keys {
    bool W_PRESSED = false;
    bool A_PRESSED = false;
    bool S_PRESSED = false;
    bool D_PRESSED = false;
    bool Q_PRESSED = false;
    bool E_PRESSED = false;
    bool ESC_PRESSED = false;
};

struct CursorPos {
    double x;
    double y;
};

class Window {
public:
    virtual ~Window() = default;

    virtual void getSize(unsigned int &width, unsigned int &height) = 0;
    virtual bool resized() = 0;

    virtual Mouse getMouseButtons() = 0;
    virtual CursorPos getCursorPos() = 0;
    virtual Keys getKeys() = 0;
    virtual void setMouseCursor(bool enabled) = 0;
    virtual double getTime() = 0;

    virtual void swapBuffers() = 0;
    virtual bool tick() = 0;
    virtual void close() = 0;
};

#endif // WINDOW_H
