#pragma once

#include "MTFAnalyzer.h"
#include <vector>

namespace lenslab {

/**
 * Manages ROIs for analysis
 */
class ROIManager
{
public:
    ROIManager();

    /**
     * Initialize default ROI positions for an image size
     */
    void initializeDefaults(int imageWidth, int imageHeight, int roiSize = 100);

    /**
     * Get all ROIs
     */
    const std::vector<ROI>& getROIs() const { return m_rois; }

    /**
     * Get mutable ROI by index
     */
    ROI& getROI(size_t index) { return m_rois[index]; }

    /**
     * Get number of ROIs
     */
    size_t count() const { return m_rois.size(); }

    /**
     * Add a new ROI
     */
    void addROI(const ROI& roi);

    /**
     * Remove ROI by index
     */
    void removeROI(size_t index);

    /**
     * Clear all ROIs
     */
    void clear();

    /**
     * Update ROI position
     */
    void setPosition(size_t index, int x, int y);

    /**
     * Update ROI size
     */
    void setSize(size_t index, int width, int height);

    /**
     * Check if point is inside any ROI, return index or -1
     */
    int hitTest(int x, int y) const;

private:
    std::vector<ROI> m_rois;
};

} // namespace lenslab
