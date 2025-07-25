#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <chrono>

namespace quasar {

constexpr double MILLISECONDS_IN_SECOND = 1e3;
constexpr double MICROSECONDS_IN_SECOND = 1e6;
constexpr double MICROSECONDS_IN_MILLISECOND = 1e3;
constexpr double NANOSECONDS_IN_SECOND = 1e9;
constexpr double NANOSECONDS_IN_MILLISECOND = 1e6;

#define BYTES_PER_MEGABYTE (1000 * 1000)

namespace timeutils {

inline time_t getTimeSeconds() {
    // Get unix timestamp in seconds
    std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return s.count();
}

inline time_t getTimeMillis() {
    // Get unix timestamp in ms
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return ms.count();
}

inline time_t getTimeMicros() {
    // Get unix timestamp in us
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return us.count();
}

inline time_t getTimeNanos() {
    // Get unix timestamp in ns
    std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return ns.count();
}

inline double nanoToMillis(double nanos) {
    return nanos / NANOSECONDS_IN_MILLISECOND;
}

inline double microsToMillis(double micros) {
    return micros / MICROSECONDS_IN_MILLISECOND;
}

inline double microsToSeconds(double micros) {
    return micros / MICROSECONDS_IN_SECOND;
}

inline double millisToMicros(double millis) {
    return millis * MICROSECONDS_IN_MILLISECOND;
}

inline double millisToSeconds(double millis) {
    return millis / MILLISECONDS_IN_SECOND;
}

inline double secondsToMicros(double seconds) {
    return seconds * MICROSECONDS_IN_SECOND;
}

inline double secondsToMillis(double seconds) {
    return seconds * MILLISECONDS_IN_SECOND;
}

}

} // namespace quasar

#endif // TIME_UTILS_H
