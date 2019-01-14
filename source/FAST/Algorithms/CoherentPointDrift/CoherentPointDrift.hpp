#ifndef COHERENT_POINT_DRIFT_HPP
#define COHERENT_POINT_DRIFT_HPP

#include "FAST/AffineTransformation.hpp"
#include "FAST/ProcessObject.hpp"
#include "FAST/Data/Mesh.hpp"

#include "FAST/SmartPointers.hpp"

namespace fast {

    class FAST_EXPORT  CoherentPointDrift: public ProcessObject {
//    FAST_OBJECT(CoherentPointDrift)
    public:
        typedef enum { RIGID, AFFINE, NONRIGID } TransformationType;
        void setFixedMeshPort(DataPort::pointer port);
        void setMovingMeshPort(DataPort::pointer port);
        void setFixedMesh(Mesh::pointer data);
        void setMovingMesh(Mesh::pointer data);
        void setFixedPoints(const MatrixXf& fixed);
        void setMovingPoints(const MatrixXf& moving);
        void setMaximumIterations(unsigned int maxIterations);
        void setUniformWeight(float uniformWeight);
        void setTolerance(float tolerance);
        void setResultTextFileName(std::string filename);
        void setNormalization(bool normalize);
        void setUpdateOutputMesh(bool updateOutputMesh);
        void setLandmarkIndices(std::vector<int> landmarksFixed, std::vector<int> landmarksMoving);
        AffineTransformation::pointer getOutputTransformation();
        Mesh::pointer getOutputMesh();

        virtual void initializeVarianceAndMore() = 0;
        void expectation(MatrixXf& fixedPoints, MatrixXf& movingPoints);
        virtual void maximization(MatrixXf& fixedPoints, MatrixXf& movingPoints) = 0;
        virtual void saveResultsToTextFile() = 0;

        void execute();

    protected:
        CoherentPointDrift();

        // Registration parameters
        float mUniformWeight;                   // Weight of the uniform distribution
        float mTolerance;                       // Convergence criteria for EM iterations
        unsigned int mMaxIterations;            // Maximum number of EM iterations
        std::string mResultFilename;
        CoherentPointDrift::TransformationType mTransformationType;

        // Variables
        MatrixXf mFixedPoints;                  // X
        MatrixXf mMovingPoints;                 // Y
        MatrixXf mMovingPointsInitial;
        MatrixXf mPX;                           // P * fixedPoints
        VectorXf mPt1;                          // Colwise sum of P, then transpose
        VectorXf mP1;                           // Rowwise sum of P
        VectorXf mFixedMeanInitial;             // Mean of fixed points
        VectorXf mMovingMeanInitial;            // Mean of moving points
        unsigned int mNumFixedPoints;           // N
        unsigned int mNumMovingPoints;          // M
        unsigned int mNumDimensions;            // D
        float mNp;                              // Sum of all elements in P
        float mScale;                           // s
        float mVariance;                        // sigma^2
        double mIterationError;
        double mFixedNormalizationScale;
        double mMovingNormalizationScale;
        AffineTransformation::pointer mTransformation;
        unsigned int mIteration;
        bool mRegistrationConverged;

        // Error metrics
        double mErrorPoints;
        double mErrorMax;
        double mErrorTransformation;
        double mErrorRotation;

        double mErrorDeformed;
        double mErrorNonDeformed;
        double mErrorLandmarksMSD;
        double mErrorLandmarksMeanDistPostreg;
        double mErrorLandmarksMeanDistPrereg;
        double mErrorLandmarksMax;
        std::vector<int> mDeformedPointIndices;
        std::vector<int> mLandmarkIndicesFixed;
        std::vector<int> mLandmarkIndicesMoving;
        bool mLandmarks;
        bool mUpdateOutputMesh;

        // OpenCL
        OpenCLDevice::pointer mDevice;
        cl::Program mProgram;
        cl::CommandQueue mQueue;
        cl::Kernel mResponsibilityReductionKernel1;
        cl::Kernel mResponsibilityReductionKernel2;
        cl::Kernel mCalculateRHSKernel;
        cl::Kernel mCalculateP1QKernel;
        cl::Kernel mCalculateGWKernel;
        cl::Kernel mQRDecompositionKernel;

        // Timing variables
        double timeE;
        double timeEBuffers;
        double timeEResponsibility;
        double timeEAffinity;
        double timeEKernel1;
        double timeEKernel2;
        double timeEUseful;
        double timeM;
        double timeMCenter;
        double timeMSolve;
        double timeMParameters;
        double timeMBuffers;
        double timeMUpdate;
        double timeMGW;
        double timeEM;
        double timeTot;

    private:
        void initializePointSets();
        void initializeVariables();
        void initializeOpenCLKernels();
        void printCloudDimensions();
        void normalizePointSets();
        void createExpectationBuffers();
        void printOutputMatrices (Affine3f existingTransform,
                Affine3f registration, Affine3f registrationTransformTotal);
        void printComputationTimes(double timeStart, double timeStartEM, double timeTotalEM);

        std::shared_ptr<Mesh> mFixedMesh;
        std::shared_ptr<Mesh> mMovingMesh;
        bool mNormalizePointSets;
        bool mPointsInitialized;

        // OpenCL buffers
        cl::Buffer mFixedBuffer;
        cl::Buffer mMovingBuffer;
        cl::Buffer mPt1BufferWrite;
        cl::Buffer mP1BufferWrite;
        cl::Buffer mPxBufferWrite;
        cl::Buffer mDenominatorBuffer;
        cl::Buffer mABuffer;
    };

} // end namespace fast

#endif
