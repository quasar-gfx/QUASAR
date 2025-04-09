#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <chrono>

namespace quasar {

#define MILLISECONDS_IN_SECOND (1e3f)
#define MICROSECONDS_IN_SECOND (1e6f)
#define MICROSECONDS_IN_MILLISECOND (1e3f)
#define NANOSECONDS_IN_SECOND (1e9f)
#define NANOSECONDS_IN_MILLISECOND (1e6f)

#define BYTES_IN_MB (1024 * 1024)

namespace timeutils {

inline uint64_t getTimeSeconds() {
    // get unix timestamp in seconds
    std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return s.count();
}

inline uint64_t getTimeMillis() {
    // get unix timestamp in ms
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return ms.count();
}

inline uint64_t getTimeMicros() {
    // get unix timestamp in us
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    );
    return us.count();
}

inline uint64_t getTimeNanos() {
    // get unix timestamp in ns
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
