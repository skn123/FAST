#pragma once

#include <FAST/Streamers/Streamer.hpp>
#include <thread>

struct _ClariusProcessedImageInfo;
struct _ClariusPosInfo;


namespace fast {

/**
 * @brief Stream ultrasound images from a Clarius scanner
 *
 * Uses the [Clarius cast API](https://github.com/clariusdev/cast) library to stream images
 * from a Clarius ultrasound scanner in real-time.
 *
 * <h3>Output ports</h3>
 * - 0: Image - Ultrasound image
 *
 * @ingroup streamers
 */
class FAST_EXPORT ClariusStreamer : public Streamer {
    FAST_OBJECT(ClariusStreamer)
    public:
        void setConnectionAddress(std::string ipAddress);
        void setConnectionPort(int port);
        void stop();
        ~ClariusStreamer();
        uint getNrOfFrames();
        void newImageFn(const void* newImage, const _ClariusProcessedImageInfo* nfo, int npos, const _ClariusPosInfo* pos);
		void toggleFreeze();
		void increaseDepth();
		void decreaseDepth();
        /**
         * Set depth in centimeters
         */
        void setDepth(float depth);
        /**
         * Set gain in percentage
         */
        void setGain(float gain);
        void loadAttributes() override;
	private:
        ClariusStreamer();
        void execute();
        void generateStream() override {};

        bool mStreamIsStarted;
        bool mFirstFrameIsInserted;
        bool mStop;
        bool mGrayscale;
        std::string mIPAddress = "192.168.1.1";
        int mPort = 5828;
        uint mNrOfFrames;

        std::mutex mFirstFrameMutex;
        std::condition_variable mFirstFrameCondition;
};

}