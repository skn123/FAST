#pragma once

#include <QWidget>
#include <FAST/Streamers/RandomAccessStreamer.hpp>

class QSlider;
class QPushButton;

namespace fast {

// The destructor is causing seg faults in python after is has been used in a window
#ifdef SWIG
%nodefaultdtor PlaybackWidget;
%extend PlaybackWidget {
    ~PlaybackWidget() {
    }
};
#endif

/**
 * @brief A widget to control playback of a RandomAccessStreamer
 * @ingroup widgets
 */
class FAST_EXPORT PlaybackWidget : public QWidget {
    public:
        PlaybackWidget(std::shared_ptr<RandomAccessStreamer> streamer, QWidget* parent = nullptr);
        void show() { QWidget::show(); };
    private:
        std::shared_ptr<RandomAccessStreamer> m_streamer;
        QSlider* m_slider;
        QPushButton* m_playButton;
};

}