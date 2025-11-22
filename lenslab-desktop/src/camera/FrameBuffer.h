#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <mutex>

namespace lenslab {

/**
 * Thread-safe ring buffer for camera frames
 */
class FrameBuffer
{
public:
    explicit FrameBuffer(size_t capacity = 10);

    void push(const cv::Mat& frame);
    bool pop(cv::Mat& frame);
    bool peek(cv::Mat& frame) const;

    size_t size() const;
    bool empty() const;
    void clear();

private:
    std::vector<cv::Mat> m_buffer;
    size_t m_capacity;
    size_t m_head = 0;
    size_t m_tail = 0;
    size_t m_count = 0;
    mutable std::mutex m_mutex;
};

} // namespace lenslab
