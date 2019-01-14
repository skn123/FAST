#ifndef FAST_NONRIGID_H
#define FAST_NONRIGID_H

#include "FAST/AffineTransformation.hpp"
#include "FAST/ProcessObject.hpp"
#include "FAST/Data/Mesh.hpp"
#include "FAST/Algorithms/CoherentPointDrift/CoherentPointDrift.hpp"


namespace fast {

    class FAST_EXPORT CoherentPointDriftNonRigid: public CoherentPointDrift {
    FAST_OBJECT(CoherentPointDriftNonRigid);

    public:
        CoherentPointDriftNonRigid();
        void setBeta(float beta);
        void setLambda(float lambda);
        void setLowRankApproxRank(unsigned int rank);
        void setDeformedIndices(std::vector<int> deformedPointIndices);

        void initializeVarianceAndMore() override;
        void maximization(MatrixXf& fixedPoints, MatrixXf& movingPoints) override;
        void saveResultsToTextFile() override;

        void createBuffers();

    private:
        void lowRankApproximation(MatrixXf& matrix, unsigned int rank,
                                  MatrixXf& eigenvectorMatrix, MatrixXf& eigenvalueMatrix);
        void calculateRHSKernel(MatrixXf& RHS, unsigned int rows, unsigned int cols);

        // Parameters
        float mBeta;
        float mLambda;
        unsigned int mLowRankApproxRank;

        // Data for calculations
        MatrixXf mEigenvaluesOfG;
        MatrixXf mEigenvectorsOfG;
        MatrixXf mG;
        MatrixXf mNonRigidDisplacement;

        // Buffers for OpenCL kernels
        cl::Buffer mP1Buffer;
        cl::Buffer mQBuffer;
        cl::Buffer mMovingPointsBuffer;
        cl::Buffer mInitialMovingPointsBuffer;
        cl::Buffer mPXBuffer;
        cl::Buffer mP1QBuffer;
        cl::Buffer mRHSBuffer;

        // Timing
        double mTimeAffinity;
        double mTimeLowRank;
    };

}
#endif //FAST_NONRIGID_H
