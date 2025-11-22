#include "FrameBuffer.h"

namespace lenslab {

FrameBuffer::FrameBuffer(size_t capacity)
    : m_capacity(capacity)
{
    m_buffer.resize(capacity);
}

void FrameBuffer::push(const cv::Mat& frame)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    frame.copyTo(m_buffer[m_head]);
    m_head = (m_head + 1) % m_capacity;

    if (m_count < m_capacity) {
        m_count++;
    } else {
        // Overwrite oldest frame
        m_tail = (m_tail + 1) % m_capacity;
    }
}

bool FrameBuffer::pop(cv::Mat& frame)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_count == 0) return false;

    m_buffer[m_tail].copyTo(frame);
    m_tail = (m_tail + 1) % m_capacity;
    m_count--;

    return true;
}

bool FrameBuffer::peek(cv::Mat& frame) const
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_count == 0) return false;

    size_t latest = (m_head + m_capacity - 1) % m_capacity;
    m_buffer[latest].copyTo(frame);

    return true;
}

size_t FrameBuffer::size() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_count;
}

bool FrameBuffer::empty() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_count == 0;
}

void FrameBuffer::clear()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_head = 0;
    m_tail = 0;
    m_count = 0;
}

} // namespace lenslab
