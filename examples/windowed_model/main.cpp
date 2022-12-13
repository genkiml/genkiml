#include <atomic>
#include <csignal>
#include <fmt/format.h>
#include <thread>

#include <genkiml/windowed_model.h>
#include "glfw.h"

std::atomic_bool is_running = true;
static void signal_handler(int) { is_running = false; }

int main()
{
    namespace sc = std::chrono;

    for (auto sig: {SIGINT, SIGTERM, SIGABRT})
        signal(sig, signal_handler);

    constexpr size_t window_size = 128;
    constexpr size_t num_signals = 2;
    constexpr double inference_interval = 0.1;
    constexpr double sample_rate_hz = 100.0;

    // TODO: Update model file and feed with mouse x/y velocity
    genki::ml::WindowedModel<window_size, num_signals, float> model(genki::ml::load_model(), inference_interval, 1.0 / sample_rate_hz);

    sc::time_point<sc::steady_clock> prev_update {};

    while (is_running)
    {
        const auto now = sc::steady_clock::now();
        const auto dt  = sc::duration_cast<sc::milliseconds>(now - prev_update);
        prev_update = now;

        const auto [norm_x, norm_y] = GLFW::normalize(GLFW::get_mouse_pos());
        fmt::print("({:.2f} {:.2f}), dt: {} ms\n", norm_x, norm_y, dt.count());

        // Just do a busy-loop. Windows' sleep/wakeup overhead is ~15ms
        while (sc::steady_clock::now() < now + sc::milliseconds(10)) {}
    }
}
