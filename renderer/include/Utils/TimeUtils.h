#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <chrono>

#define MILLISECONDS_IN_SECOND 1e3f
#define MICROSECONDS_IN_SECOND 1e6f
#define MICROSECONDS_IN_MILLISECOND 1e3f

#define MBPS_TO_BPS 1e6f

namespace timeutils {

inline int getTimeMillis() {
    // get unix timestamp in ms
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return ms.count();
}

inline int getTimeMicros() {
    // get unix timestamp in us
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return us.count();
}

inline int getTimeNanos() {
    // get unix timestamp in ns
    std::chrono::nanoseconds ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return ns.count();
}

inline float microsToMillis(float micros) {
    return micros / MICROSECONDS_IN_MILLISECOND;
}

inline float microsToSeconds(float micros) {
    return micros / MICROSECONDS_IN_SECOND;
}

inline float millisToMicros(float millis) {
    return millis * MICROSECONDS_IN_MILLISECOND;
}

inline float millisToSeconds(float millis) {
    return millis / MILLISECONDS_IN_SECOND;
}

inline float secondsToMicros(float seconds) {
    return seconds * MICROSECONDS_IN_SECOND;
}

inline float secondsToMillis(float seconds) {
    return seconds * MILLISECONDS_IN_SECOND;
}

}

#endif // TIME_UTILS_H
