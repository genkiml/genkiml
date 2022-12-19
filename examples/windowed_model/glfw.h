#pragma once

#include <cassert>
#include <GLFW/glfw3.h>

struct GLFW
{
    struct Point { double x, y; };

    static auto normalize(const Point& p)
    {
        static const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
        assert(mode != nullptr);

        return Point {
                p.x / static_cast<double>(mode->width),
                p.y / static_cast<double>(mode->height),
        };
    };

    static auto get_mouse_pos()
    {
        assert(wnd != nullptr);

        double x {}, y {};
        glfwGetCursorPos(wnd, &x, &y);

        int wnd_x{}, wnd_y{};
        glfwGetWindowPos(wnd, &wnd_x, &wnd_y);

        return Point{wnd_x + x, wnd_y + y};
    }

private:
    static inline GLFWwindow* wnd = []
    {
#if defined(__APPLE__) && __APPLE__
        glfwInitHint(GLFW_COCOA_MENUBAR, GLFW_FALSE);
#endif
        glfwInit();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

        return glfwCreateWindow(400, 300, "", nullptr, nullptr);
    }();
};
