#pragma once

#include <thread>
#include <unordered_map>
#include "FAST/Streamers/Streamer.hpp"
#include "FAST/ProcessObject.hpp"
#include <set>
#include "FASTExport.hpp"
#include <deque>

// Forward declare

namespace fast {

class Image;
class IGTLSocketWrapper;

/**
 * @brief Stream image or transforms from an OpenIGTLink server
 *
 * This streamer uses the OpenIGTLink protocol and library to stream data such as images and transforms from a server
 *
 * <h3>Output ports</h3>
 * Multiple ports possible dependeing on number of streams from OpenIGTLink server
 *
 * @ingroup streamers
 */
class FAST_EXPORT OpenIGTLinkStreamer : public Streamer {
    FAST_OBJECT(OpenIGTLinkStreamer)
    public:
		std::set<std::string> getImageStreamNames();
		std::set<std::string> getTransformStreamNames();
		std::vector<std::string> getActiveImageStreamNames();
		std::vector<std::string> getActiveTransformStreamNames();
		std::string getStreamDescription(std::string streamName);
        void setConnectionAddress(std::string address);
        void setConnectionPort(uint port);
        uint getNrOfFrames() const;

		/**
		 * Will select first image stream
		 * @return
		 */
		DataChannel::pointer getOutputPort(uint portID = 0) override;

        template<class T>
        DataChannel::pointer getOutputPort(std::string deviceName);

        /**
         * This method runs in a separate thread and adds frames to the
         * output object
         */
        void generateStream() override;

        ~OpenIGTLinkStreamer();
        void loadAttributes() override;

        float getCurrentFramerate();
    private:
        OpenIGTLinkStreamer();

        // Update the streamer if any parameters have changed
        void execute();

        void addTimestamp(uint64_t timestamp);

        uint mNrOfFrames;
        uint mMaximumNrOfFrames;
        bool mMaximumNrOfFramesSet;
        std::deque<uint64_t> m_timestamps;

        bool mInFreezeMode;

        std::string mAddress;
        uint mPort;

		IGTLSocketWrapper* mSocketWrapper;
        //igtl::ClientSocket::Pointer mSocket;

		std::set<std::string> mImageStreamNames;
		std::set<std::string> mTransformStreamNames;
		std::unordered_map<std::string, std::string> mStreamDescriptions;
        std::unordered_map<std::string, uint> mOutputPortDeviceNames;

        void updateFirstFrameSetFlag();
};



template<class T>
DataChannel::pointer OpenIGTLinkStreamer::getOutputPort(std::string deviceName) {
	uint portID;
	if(mOutputPortDeviceNames.count(deviceName) == 0) {
		portID = getNrOfOutputPorts();
		createOutputPort<T>(portID);
		mOutputPortDeviceNames[deviceName] = portID;
	} else {
		portID = mOutputPortDeviceNames[deviceName];
	}
    return ProcessObject::getOutputPort(portID);
}



} // end namespace

