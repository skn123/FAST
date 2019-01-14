#ifndef ITERATIVE_CLOSEST_POINT_HPP
#define ITERATIVE_CLOSEST_POINT_HPP

#include "FAST/AffineTransformation.hpp"
#include "FAST/ProcessObject.hpp"
#include "FAST/Data/Mesh.hpp"

namespace fast {

class FAST_EXPORT  IterativeClosestPoint : public ProcessObject {
    FAST_OBJECT(IterativeClosestPoint)
    public:
        typedef enum { RIGID, TRANSLATION } TransformationType;
        void setFixedMeshPort(DataPort::pointer port);
        void setFixedMesh(Mesh::pointer data);
        void setMovingMeshPort(DataPort::pointer port);
        void setMovingMesh(Mesh::pointer data);
        void setTransformationType(const IterativeClosestPoint::TransformationType type);
        AffineTransformation::pointer getOutputTransformation();
        float getError() const;
        void setMinimumErrorChange(float errorChange);
        void setMaximumNrOfIterations(uint iterations);
        void setRandomPointSampling(uint nrOfPointsToSample);
        void setDistanceThreshold(float distance);

        double getTime() const;
        uint getIterations() const;

        void setLandmarkIndices(std::vector<int> landmarksFixed, std::vector<int> landmarksMoving);
        void setFilename(std::string filename);


    private:
        IterativeClosestPoint();
        void execute();

        float mMinErrorChange;
        uint mMaxIterations;
        int mRandomSamplingPoints;
        float mDistanceThreshold;
        float mError;
        AffineTransformation::pointer mTransformation;
        IterativeClosestPoint::TransformationType mTransformationType;

        double mTimeTot;
        uint mIterations;

        std::string mFilename;
        std::vector<int> mLandmarkIndicesFixed;
        std::vector<int> mLandmarkIndicesMoving;
        bool mLandmarks;
};

} // end namespace fast

#endif
