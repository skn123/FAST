#include "FAST/Algorithms/CoherentPointDrift/CoherentPointDrift.hpp"
#include "FAST/Visualization/ImageRenderer/ImageRenderer.hpp"
#include "FAST/SceneGraph.hpp"

#include <limits>
#include <iostream>
#include "math.h"

namespace fast {

    CoherentPointDrift::CoherentPointDrift() {
        createInputPort<Mesh>(0);
        createInputPort<Mesh>(1);
        createOutputPort<Mesh>(0);
        mTransformation = AffineTransformation::New();

        // Set default values
        mMaxIterations = 100;
        mIteration = 0;
        mTolerance = 1e-6;
        mUniformWeight = 0.5;
        mRegistrationConverged = false;
        mScale = 1.0;
        mIterationError = 100*mTolerance;
        mFixedNormalizationScale = 1.0;
        mMovingNormalizationScale = 1.0;
        mResultFilename = "CPDResults" + currentDateTime() + ".txt";
        mNormalizePointSets = true;
        mLandmarks = false;
        mUpdateOutputMesh = false;
        mPointsInitialized = false;

        timeE = 0.0;
        timeEBuffers = 0.0;
        timeEResponsibility = 0.0;
        timeEAffinity = 0.0;
        timeEKernel1 = 0.0f;
        timeEKernel2 = 0.0f;
        timeEUseful = 0.0;
        timeM = 0.0;
        timeMCenter = 0.0;
        timeMSolve = 0.0;
        timeMParameters = 0.0;
        timeMBuffers = 0.0;
        timeMUpdate = 0.0;
        timeMGW = 0.0;
        timeEM = 0.0;
        timeTot = 0.0;
        mErrorPoints = 0.0;
        mErrorTransformation = 0.0;
        mErrorRotation = 0.0;
        mErrorDeformed = 0.0;
        mErrorNonDeformed = 0.0;

        // OpenCL
        createOpenCLProgram(Config::getKernelSourcePath() + "Algorithms/CoherentPointDrift/CoherentPointDrift3D.cl", "3D");
        mDevice = std::dynamic_pointer_cast<OpenCLDevice>(getMainDevice());
        mQueue = mDevice->getCommandQueue();
    }

    void CoherentPointDrift::expectation(MatrixXf& fixedPoints, MatrixXf& movingPoints) {

        double timeStartE = omp_get_wtime();

        auto uniformWeightConstant = (float) (pow(2.0*(double)EIGEN_PI*mVariance, (float)mNumDimensions/2.0)
                * (mUniformWeight/(1-mUniformWeight))
                * (float)mNumMovingPoints/mNumFixedPoints);

        /* ******************************************************************
         * Calculate the matrix reductions and products PT1, P1 and PX on GPU
         * *****************************************************************/

        // Create OpenCL buffers
        createExpectationBuffers();
        double timeEndBuffers = omp_get_wtime();

        // Calculate Pt1
        mResponsibilityReductionKernel1.setArg(0, mFixedBuffer);
        mResponsibilityReductionKernel1.setArg(1, mMovingBuffer);
        mResponsibilityReductionKernel1.setArg(2, mPt1BufferWrite);
        mResponsibilityReductionKernel1.setArg(3, mDenominatorBuffer);
        mResponsibilityReductionKernel1.setArg(4, uniformWeightConstant);
        mResponsibilityReductionKernel1.setArg(5, mVariance);
        mResponsibilityReductionKernel1.setArg(6, mNumMovingPoints);

        mQueue.enqueueNDRangeKernel(
                mResponsibilityReductionKernel1,
                cl::NullRange,
                cl::NDRange(mNumFixedPoints),
                cl::NullRange
        );
        mQueue.enqueueReadBuffer(mPt1BufferWrite,
                                 CL_TRUE,
                                 0,
                                 mNumFixedPoints*sizeof(float),
                                 mPt1.data()
        );
        double timeEndKernel1 = omp_get_wtime();

        // Calculate P1 and PX
        mResponsibilityReductionKernel2.setArg(0, mFixedBuffer);
        mResponsibilityReductionKernel2.setArg(1, mMovingBuffer);
        mResponsibilityReductionKernel2.setArg(2, mP1BufferWrite);
        mResponsibilityReductionKernel2.setArg(3, mDenominatorBuffer);
        mResponsibilityReductionKernel2.setArg(4, mPxBufferWrite);
        mResponsibilityReductionKernel2.setArg(5, uniformWeightConstant);
        mResponsibilityReductionKernel2.setArg(6, mVariance);
        mResponsibilityReductionKernel2.setArg(7, mNumFixedPoints);

        mQueue.enqueueNDRangeKernel(
                mResponsibilityReductionKernel2,
                cl::NullRange,
                cl::NDRange(mNumMovingPoints),
                cl::NullRange
        );
        mQueue.enqueueReadBuffer(mP1BufferWrite,
                                 CL_TRUE,
                                 0,
                                 mNumMovingPoints*sizeof(float),
                                 mP1.data()
        );
        mQueue.enqueueReadBuffer(mPxBufferWrite,
                                 CL_TRUE,
                                 0,
                                 mNumMovingPoints*mNumDimensions*sizeof(float),
                                 mPX.data()
        );
        double timeEndKernel2 = omp_get_wtime();

        mNp = mPt1.sum();
        double timeEndEUseful = omp_get_wtime();


        // Update computation times
        double timeEndE = omp_get_wtime();
        timeEBuffers = timeEndBuffers - timeStartE;
//        timeEResponsibility += timeEndResponsibility - timeEndBuffers;
//        timeEAffinity += timeEndAffinity - timeEndResponsibility;
        timeEKernel1 += timeEndKernel1 - timeEndBuffers;
        timeEKernel2 += timeEndKernel2 - timeEndKernel1;
        timeEUseful += timeEndEUseful - timeEndKernel2;
        timeE += timeEndE - timeStartE;

    }

    void CoherentPointDrift::execute() {

        auto existingTransform = Affine3f::Identity();
        // Store the point sets in matrices and store their dimensions
        if (not mPointsInitialized) {

            // Read point from meshes to matrices
            initializePointSets();

            // Apply the existing transform, if any, to moving point cloud
            existingTransform = SceneGraph::getEigenAffineTransformationFromData(mMovingMesh);
            mMovingPoints = mMovingPoints.rowwise().homogeneous() * existingTransform.affine().transpose();
        }
        if (mLandmarks) {
            mMovingPointsInitial = mMovingPoints;
        }

        double timeStart = omp_get_wtime();

        // Normalize the point sets, i.e. zero mean and unit variance
        if (mNormalizePointSets) {
            normalizePointSets();
        }

        // Initialization
        initializeVariables();
        initializeOpenCLKernels();
        initializeVarianceAndMore();


        /* *************************
         * Get some points drifting!
         * ************************/
        double timeStartEM = omp_get_wtime();
        while (mIteration < mMaxIterations && !mRegistrationConverged) {
            expectation(mFixedPoints, mMovingPoints);
            maximization(mFixedPoints, mMovingPoints);
            mIteration++;
        }
        double timeEndEM = omp_get_wtime();
        timeEM = timeEndEM - timeStartEM;

        /* ***********************************************
         * Denormalize and set total transformation matrix
         * **********************************************/
        Affine3f registrationTransformTotal;
        Affine3f registration = mTransformation->getTransform();
        registration.scale(mScale);

        if (mNormalizePointSets) {
            // Set normalization
            Affine3f normalization = Affine3f::Identity();
            normalization.translate((Vector3f) -(mMovingMeanInitial));

            // Denormalize moving point set
            mScale *= mFixedNormalizationScale / mMovingNormalizationScale;
            registration.scale(mFixedNormalizationScale / mMovingNormalizationScale);
            registration.translation() *= mFixedNormalizationScale;

            Affine3f denormalization = Affine3f::Identity();
            denormalization.translate((Vector3f) (mFixedMeanInitial));
            registrationTransformTotal = denormalization * registration * normalization;
        } else {
            registrationTransformTotal = registration;
        }

        timeTot = omp_get_wtime() - timeStart;

        // Set total transformation
        auto transform = AffineTransformation::New();
        transform->setTransform(registrationTransformTotal * existingTransform);

        // Update moving mesh for FAST visualization
        // NB! This may affect registration if multiple runs of same data are performed
        if (mUpdateOutputMesh) {
            if (mTransformationType == NONRIGID) {
                auto movingaccess = mMovingMesh->getMeshAccess(ACCESS_READ);
                auto vertices = movingaccess->getVertices();
                auto lines = movingaccess->getLines();
                auto triangles = movingaccess->getTriangles();
#pragma omp parallel for
                for(unsigned long i = 0; i < mNumMovingPoints; ++i) {
                    vertices.at(i).setPosition(mMovingPoints.row(i));
                }
                auto newMesh = Mesh::New();
                newMesh->create(vertices, lines, triangles);
                addOutputData(0, newMesh);
            }
            else {
                mMovingMesh->getSceneGraphNode()->setTransformation(transform);
                addOutputData(0, mMovingMesh);
            }
        }


        /* *****************************
        * Registration error and results
        * *****************************/
        unsigned int numPoints = min(mNumFixedPoints, mNumMovingPoints);
        double errorPoints = 0.0;
        double errorMax = 0.0;
//#pragma omp parallel for reduction(+: errorPoints)
        for (int i = 0; i < numPoints; i++) {
            double norm = (mFixedPoints.row(i)-mMovingPoints.row(i)).squaredNorm();
            errorPoints += norm;
            if (norm > errorMax) {
                errorMax = norm;
            }
        }
        mErrorPoints = errorPoints / numPoints;
        mErrorMax = errorMax;

        MatrixXf rotationDiff = registration.linear().inverse() - existingTransform.linear();
        MatrixXf affineDiff = registration.inverse().affine() - existingTransform.affine();
        mErrorRotation = rotationDiff.squaredNorm();
        mErrorTransformation = affineDiff.squaredNorm();

        double errorDeformed = 0.0;
        for (int i : mDeformedPointIndices) {
            errorDeformed += (mFixedPoints.row(i) - mMovingPoints.row(i)).squaredNorm();
        }
        mErrorDeformed = errorDeformed / mDeformedPointIndices.size();
        mErrorNonDeformed = (errorPoints-errorDeformed) / (numPoints - mDeformedPointIndices.size());

        if(mLandmarks) {
            double landmarksErrorMSD = 0.0;
            double landmarksMeanDistPre = 0.0;
            double landmarksMeanDistPost = 0.0;
            double landmarksMax = 0.0;
            int numLandmarks = mLandmarkIndicesFixed.size();
            assert (numLandmarks == mLandmarkIndicesMoving.size());
            for (int i = 0; i < numLandmarks; ++i) {
                int n = mLandmarkIndicesFixed[i];
                int m = mLandmarkIndicesMoving[i];
                landmarksMeanDistPre += (mFixedPoints.row(n) - mMovingPointsInitial.row(m)).norm();
                double norm = (mFixedPoints.row(n) - mMovingPoints.row(m)).norm();
                landmarksMeanDistPost += norm;
                if (norm > landmarksMax) {
                    landmarksMax = norm;
                }
                landmarksErrorMSD += (mFixedPoints.row(n) - mMovingPoints.row(m)).squaredNorm();
            }
            mErrorLandmarksMeanDistPrereg = landmarksMeanDistPre / numLandmarks;
            mErrorLandmarksMeanDistPostreg = landmarksMeanDistPost / numLandmarks;
            mErrorLandmarksMSD = landmarksErrorMSD / numLandmarks;
            mErrorLandmarksMax = landmarksMax;
        }

        if(isnan(mErrorPoints)) {
            std::cout << "--------------NAN issues-------------\n";
            exit(0);
        }


        // Save results to file
        saveResultsToTextFile();

//        printComputationTimes(timeStart, timeStartEM, timeEM);
//        printOutputMatrices(existingTransform, registration, registrationTransformTotal);
    }

    void CoherentPointDrift::setFixedMesh(Mesh::pointer data) {
        setInputData(0, data);
    }

    void CoherentPointDrift::setFixedPoints(const MatrixXf& fixed) {
        mFixedPoints = fixed;
        mNumFixedPoints = fixed.rows();
        mNumDimensions = fixed.cols();
        mPointsInitialized = true;

    }

    void CoherentPointDrift::setMovingMesh(Mesh::pointer data) {
        setInputData(1, data);
    }

    void CoherentPointDrift::setMovingPoints(const MatrixXf& moving) {
        mMovingPoints = moving;
        mNumMovingPoints = moving.rows();
        mPointsInitialized = true;
    }

    void CoherentPointDrift::setFixedMeshPort(DataPort::pointer port) {
        setInputConnection(0, port);
    }

    void CoherentPointDrift::setMovingMeshPort(DataPort::pointer port) {
        setInputConnection(1, port);
    }

    void CoherentPointDrift::setMaximumIterations(unsigned int maxIterations) {
        mMaxIterations = maxIterations;
    }

    void CoherentPointDrift::setUniformWeight(float uniformWeight) {
        mUniformWeight = uniformWeight;
    }

    void CoherentPointDrift::setTolerance(float tolerance) {
        mTolerance = tolerance;
    }

    void CoherentPointDrift::setResultTextFileName(std::string filename) {
        mResultFilename = filename;
    }

    void CoherentPointDrift::setNormalization(bool normalize) {
        mNormalizePointSets = normalize;
    }

    void CoherentPointDrift::setUpdateOutputMesh(bool updateOutputMesh) {
        mUpdateOutputMesh = updateOutputMesh;
    }

    void CoherentPointDrift::initializePointSets() {

        // Load point meshes
        mFixedMesh = getInputData<Mesh>(0);
        mMovingMesh = getInputData<Mesh>(1);

        // Get access to the two point sets
        MeshAccess::pointer accessFixedSet = mFixedMesh->getMeshAccess(ACCESS_READ);
        MeshAccess::pointer accessMovingSet = mMovingMesh->getMeshAccess(ACCESS_READ);

        // Get the points from the meshes
        std::vector<MeshVertex> fixedVertices = accessFixedSet->getVertices();
        std::vector<MeshVertex> movingVertices = accessMovingSet->getVertices();

        // Set dimensions of point sets
        unsigned int numDimensionsFixed = (unsigned int)fixedVertices[0].getPosition().size();
        unsigned int numDimensionsMoving = (unsigned int)movingVertices[0].getPosition().size();
        assert(numDimensionsFixed == numDimensionsMoving);
        mNumDimensions = numDimensionsFixed;
        mNumFixedPoints = (unsigned int)fixedVertices.size();
        mNumMovingPoints = (unsigned int)movingVertices.size();

        // Store point sets in matrices
        mFixedPoints = MatrixXf::Zero(mNumFixedPoints, mNumDimensions);
        mMovingPoints = MatrixXf::Zero(mNumMovingPoints, mNumDimensions);
        for(int i = 0; i < mNumFixedPoints; ++i) {
            mFixedPoints.row(i) = fixedVertices[i].getPosition();
        }
        for(int i = 0; i < mNumMovingPoints; ++i) {
            mMovingPoints.row(i) = movingVertices[i].getPosition();
        }
    }

    void CoherentPointDrift::initializeVariables() {
        mPX = MatrixXf::Zero(mNumMovingPoints, mNumDimensions);
        mPt1 = VectorXf::Zero(mNumFixedPoints);
        mP1 = VectorXf::Zero(mNumMovingPoints);
        mFixedMeanInitial = VectorXf::Zero(mNumDimensions);
        mMovingMeanInitial = VectorXf::Zero(mNumDimensions);

        mVariance = (   (double)mNumMovingPoints * (mFixedPoints.transpose() * mFixedPoints).trace() +
                        (double)mNumFixedPoints * (mMovingPoints.transpose() * mMovingPoints).trace() -
                        2.0 * mFixedPoints.colwise().sum() * mMovingPoints.colwise().sum().transpose()  ) /
                    (double)(mNumFixedPoints * mNumMovingPoints * mNumDimensions);
        mIterationError = 10.0*mTolerance;
    }

    void CoherentPointDrift::initializeOpenCLKernels() {
        if (mNumDimensions != 3) {
            std::cout << "Kernels for other dimensions than 3 is not yet implemented" << std::endl;
            exit(0);
        }

        mProgram = getOpenCLProgram(mDevice, "3D");
        mResponsibilityReductionKernel1 = cl::Kernel(mProgram, "calculatePT1AndDenominator");
        mResponsibilityReductionKernel2 = cl::Kernel(mProgram, "calculateP1AndPX");
        mCalculateP1QKernel = cl::Kernel(mProgram, "calculateP1Q");
        mCalculateRHSKernel = cl::Kernel(mProgram, "calculateRHS");
//        mCalculateGWKernel = cl::Kernel(mProgram, "calculateGW");
        mQRDecompositionKernel = cl::Kernel(mProgram, "qr");
    }

    void CoherentPointDrift::normalizePointSets() {

        // Center point clouds around origin, i.e. zero mean
        mFixedMeanInitial = (mFixedPoints.colwise().sum() / mNumFixedPoints).transpose();
        mMovingMeanInitial = (mMovingPoints.colwise().sum() / mNumMovingPoints).transpose();

        mFixedPoints -= mFixedMeanInitial.transpose().replicate(mNumFixedPoints, 1);
        mMovingPoints -= mMovingMeanInitial.transpose().replicate(mNumMovingPoints, 1);

        // Scale point clouds to have unit variance
        mFixedNormalizationScale = sqrt(mFixedPoints.cwiseProduct(mFixedPoints).sum() / (double)mNumFixedPoints);
        mMovingNormalizationScale = sqrt(mMovingPoints.cwiseProduct(mMovingPoints).sum() / (double)mNumMovingPoints);
        mFixedPoints /= mFixedNormalizationScale;
        mMovingPoints /= mMovingNormalizationScale;
    }

    void CoherentPointDrift::createExpectationBuffers() {
        // Set up initial read buffers
        mFixedBuffer = cl::Buffer(mDevice->getContext(),
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  mNumFixedPoints*mNumDimensions*sizeof(float),
                                  mFixedPoints.data());
        mMovingBuffer = cl::Buffer(mDevice->getContext(),
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   mNumMovingPoints*mNumDimensions*sizeof(float),
                                   mMovingPoints.data());

        // Set up write buffers
        mPt1BufferWrite = cl::Buffer(mDevice->getContext(),
                                     CL_MEM_WRITE_ONLY,
                                     mNumFixedPoints*sizeof(float));
        mP1BufferWrite = cl::Buffer(mDevice->getContext(),
                                    CL_MEM_WRITE_ONLY,
                                    mNumMovingPoints*sizeof(float));
        mPxBufferWrite = cl::Buffer(mDevice->getContext(),
                                    CL_MEM_WRITE_ONLY,
                                    mNumMovingPoints*mNumDimensions*sizeof(float));

        // Set up read-write buffers (temporary calculations)
        mDenominatorBuffer = cl::Buffer(mDevice->getContext(),
                                        CL_MEM_READ_WRITE,
                                        mNumFixedPoints*sizeof(float));
        mABuffer = cl::Buffer(mDevice->getContext(),
                              CL_MEM_READ_WRITE,
                              mNumFixedPoints*sizeof(float));
    }

    void CoherentPointDrift::setLandmarkIndices(std::vector<int> landmarksFixed, std::vector<int> landmarksMoving) {
        mLandmarkIndicesFixed = landmarksFixed;
        mLandmarkIndicesMoving = landmarksMoving;
        mLandmarks = true;
    }

    AffineTransformation::pointer CoherentPointDrift::getOutputTransformation() {
        return mTransformation;
    }

    Mesh::pointer CoherentPointDrift::getOutputMesh() {
        return mMovingMesh;
    }

    void CoherentPointDrift::printCloudDimensions() {
        std::cout << "\n****************************************\n";
        std::cout << "Fixed point cloud: " << mNumFixedPoints << " x " << mNumDimensions << std::endl;
        std::cout << "Moving point cloud: " << mNumMovingPoints << " x " << mNumDimensions << std::endl;
    }

    void CoherentPointDrift::printOutputMatrices(Affine3f existingTransform, Affine3f registration,
                                                 Affine3f registrationTransformTotal) {
        std::cout << "\n*****************************************\n";
        std::cout << "Existing transform: \n" << existingTransform.matrix() << std::endl;
        std::cout << "Registration matrix: \n" << registration.matrix() << std::endl;
        std::cout << "Registration matrix inverse: \n" << registration.matrix().inverse() << std::endl;
        std::cout << "Final registration matrix: \n" << registrationTransformTotal.matrix() << std::endl;
        std::cout << "Registered transform * existingTransform (should be identity): \n"
                  << registrationTransformTotal * existingTransform.matrix() << std::endl;
    }

    void CoherentPointDrift::printComputationTimes(double timeStart, double timeStartEM, double timeTotalEM) {
        std::cout << "\nCOMPUTATION TIMES:\n";
        std::cout << "Initialization and normalization: " << timeStartEM-timeStart << " s.\n";
        std::cout << "EM converged in " << mIteration-1 << " iterations in " << timeTotalEM << " s.\n";
        std::cout << "Time spent on expectation: " << timeE << " s\n";
        std::cout << "      - Setting up buffers: " << timeEBuffers << " s.\n";
        std::cout << "      - Calculating P: " << timeEResponsibility << " s.\n";
        std::cout << "      - Calculating K: " << timeEAffinity << " s.\n";
        std::cout << "      - Calculating Pt1 and a (kernel 1): " << timeEKernel1 << " s.\n";
        std::cout << "      - Calculating Pt and PX (kernel 2): " << timeEKernel2 << " s.\n";
        std::cout << "      - Calculating P1, Pt1, Np (OpenMP): " << timeEUseful << " s.\n";
        std::cout << "      - TOT GPU expectation: " << timeEAffinity+timeEKernel1+timeEKernel2 << " s.\n";
        std::cout << "      - TOT CPU expectation: " << timeEResponsibility+timeEUseful << " s.\n";

        std::cout << "Time spent on maximization: " << timeM << " s\n";
        if (mTransformationType == NONRIGID) {
            std::cout << "      - Calc. k: " << timeMParameters << " s.\n";
            std::cout << "      - Setting up buffers: " << timeMBuffers << " s.\n";
            std::cout << "      - Solving for W: " << timeMSolve << " s.\n";
            std::cout << "      - Calculating GW: " << timeMGW << " s.\n";
            std::cout << "      - Calc. new points, updating transformation and error: " << timeMUpdate << " s.\n";
        } else {
            std::cout << "      - Centering point clouds: " << timeMCenter << " s.\n";
            std::cout << "      - SVD (rigid): " << timeMSolve << " s.\n";
            std::cout << "      - Calculation transformation parameters: " << timeMParameters << " s.\n";
            std::cout << "      - Updating transformation and error: " << timeMUpdate << " s.\n";
        }
    }


}
