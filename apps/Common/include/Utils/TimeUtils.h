#ifndef TIME_UTILS_H
#define TIME_UTILS_H

#include <chrono>

#define MILLISECONDS_IN_SECOND 1e3f
#define MICROSECONDS_IN_SECOND 1e6f
#define MICROSECONDS_IN_MILLISECOND 1e3f

#define MBPS_TO_BPS 1e6f

namespace timeutils {
    inline int getCurrTimeMs() {
        // get unix timestamp in ms
        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        );
        return ms.count();
    }
}

#endif // TIME_UTILS_H
