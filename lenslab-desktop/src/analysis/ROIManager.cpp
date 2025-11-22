#include "ROIManager.h"

namespace lenslab {

ROIManager::ROIManager() = default;

void ROIManager::initializeDefaults(int imageWidth, int imageHeight, int roiSize)
{
    m_rois.clear();

    int margin = roiSize;
    int centerX = imageWidth / 2;
    int centerY = imageHeight / 2;

    // Center ROI
    m_rois.push_back({centerX - roiSize / 2, centerY - roiSize / 2, roiSize, roiSize, "C"});

    // Corner ROIs
    m_rois.push_back({margin, margin, roiSize, roiSize, "UL"});
    m_rois.push_back({imageWidth - margin - roiSize, margin, roiSize, roiSize, "UR"});
    m_rois.push_back({margin, imageHeight - margin - roiSize, roiSize, roiSize, "LL"});
    m_rois.push_back({imageWidth - margin - roiSize, imageHeight - margin - roiSize, roiSize, roiSize, "LR"});
}

void ROIManager::addROI(const ROI& roi)
{
    m_rois.push_back(roi);
}

void ROIManager::removeROI(size_t index)
{
    if (index < m_rois.size()) {
        m_rois.erase(m_rois.begin() + index);
    }
}

void ROIManager::clear()
{
    m_rois.clear();
}

void ROIManager::setPosition(size_t index, int x, int y)
{
    if (index < m_rois.size()) {
        m_rois[index].x = x;
        m_rois[index].y = y;
    }
}

void ROIManager::setSize(size_t index, int width, int height)
{
    if (index < m_rois.size()) {
        m_rois[index].width = width;
        m_rois[index].height = height;
    }
}

int ROIManager::hitTest(int x, int y) const
{
    for (size_t i = 0; i < m_rois.size(); i++) {
        const ROI& roi = m_rois[i];
        if (x >= roi.x && x < roi.x + roi.width &&
            y >= roi.y && y < roi.y + roi.height) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

} // namespace lenslab
