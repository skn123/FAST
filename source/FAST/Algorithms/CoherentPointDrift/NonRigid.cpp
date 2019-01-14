#include "CoherentPointDrift.hpp"
#include "NonRigid.hpp"

#include <Eigen/Eigenvalues>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>

#include "math.h"

namespace fast {

    CoherentPointDriftNonRigid::CoherentPointDriftNonRigid() {
        mScale = 1.0;
        mTransformationType = TransformationType::NONRIGID;
        mBeta = 2.0f;
        mLambda = 2.0f;
        mLowRankApproxRank = 10;
        mTimeAffinity = 0.0;
        mTimeLowRank = 0.0;
    }

    void CoherentPointDriftNonRigid::setBeta(float beta) {
        mBeta = beta;
    }

    void CoherentPointDriftNonRigid::setLambda(float lambda) {
        mLambda = lambda;
    }

    void CoherentPointDriftNonRigid::setLowRankApproxRank(unsigned int rank) {
        mLowRankApproxRank = rank;
    }

    void CoherentPointDriftNonRigid::setDeformedIndices(std::vector<int> deformedPointIndices) {
        mDeformedPointIndices = deformedPointIndices;
    }

    void CoherentPointDriftNonRigid::initializeVarianceAndMore() {
        double timeInitStart = omp_get_wtime();
        mNonRigidDisplacement = MatrixXf::Zero(mNumMovingPoints, mNumDimensions);
        mG = MatrixXf::Zero(mNumMovingPoints, mNumMovingPoints);        // Kernel matrix
#pragma omp parallel for
        for (int i = 0; i < mNumMovingPoints; ++i) {
            for (int j = 0; j < i; ++j) {
                float norm = (mMovingPoints.row(i) - mMovingPoints.row(j)).squaredNorm();
                mG(i, j) = expf(norm / (-2.0f * mBeta * mBeta));
                mG(j, i) = mG(i, j);
            }
            mG(i, i) = 1.0f;
        }
        mTimeAffinity = omp_get_wtime() - timeInitStart;

        mInitialMovingPointsBuffer = cl::Buffer(mDevice->getContext(),
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         mNumMovingPoints*mNumDimensions*sizeof(float),
                                         mMovingPoints.data());
        mEigenvaluesOfG = MatrixXf::Zero(mLowRankApproxRank, mLowRankApproxRank);
        mEigenvectorsOfG = MatrixXf::Zero(mNumMovingPoints, mLowRankApproxRank);

        double tStartLowRank = omp_get_wtime();
        lowRankApproximation(mG, mLowRankApproxRank, mEigenvectorsOfG, mEigenvaluesOfG);
        mTimeLowRank = omp_get_wtime() - tStartLowRank;

        mQBuffer = cl::Buffer(mDevice->getContext(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              mNumMovingPoints*mLowRankApproxRank*sizeof(float),
                              mEigenvectorsOfG.data());
    }

    void CoherentPointDriftNonRigid::maximization(Eigen::MatrixXf &fixedPoints, Eigen::MatrixXf &movingPoints) {
        double startM = omp_get_wtime();

        auto k = (float) 1.0f / (mLambda * mVariance);
        double timeEndParameters = omp_get_wtime();

        // Create buffers for OpenCL kernels
        createBuffers();
        double timeEndBuffers = omp_get_wtime();


        /* ********************************************************************
         * Solve the matrix equation LHS * W = PX - dP1 * movingPoints = RHS
         * for W by pre-multiplying by the inverse of LHS, where
         * LHS = d(P1) * G + lambda*variance*I,
         * and G is approximated by low rank approximation with the
         * lowRankApproxRank largest eigenvalues:
         * G = eigenvectors*eigenvalues.inv*eigenvectors.transp = Q*Lambda*Q^T.
         * The Woodbury identity is used to simplify the inverse of LHS.
         * *******************************************************************/

        MatrixXf dP1Q(mNumMovingPoints, mLowRankApproxRank);
#pragma omp parallel for
        for (int i = 0; i < mNumMovingPoints; ++i) {
            for (int j = 0; j < mLowRankApproxRank; ++j) {
                dP1Q(i, j) = mP1(i) * mEigenvectorsOfG(i, j);
            }
        }

        // Calculate the inverse from the Woodbury identity E multiplied by Q^T
        MatrixXf eigenInverse = (mEigenvaluesOfG.inverse() + k * mEigenvectorsOfG.transpose() * dP1Q).inverse();
        MatrixXf EQt = eigenInverse * mEigenvectorsOfG.transpose();

        // Calculate RHS
        MatrixXf RHS(mNumMovingPoints, mNumDimensions);
        calculateRHSKernel(RHS, mNumMovingPoints, mNumDimensions);

        // Calculate W
        MatrixXf WPart1 = k * k * (dP1Q * (EQt * RHS));
        MatrixXf W = k * RHS - WPart1;
        double timeEndSolve = omp_get_wtime();

        /* *************************
         * Transform the point cloud
         * ************************/

        // Calculate GW using low rank approximation of G
        MatrixXf GW = mEigenvectorsOfG * (mEigenvaluesOfG * (mEigenvectorsOfG.transpose() * W));
        double timeEndGW = omp_get_wtime();

        // Update the moving points
        movingPoints += GW;

        // Update variance
        double varianceOld = mVariance;
        float trXPX = 0.0f;
        float trYPY = 0.0f;
        float trPXY = 0.0f;
        for (int n = 0; n < mNumFixedPoints; ++n) {
            trXPX += mPt1(n) * (fixedPoints(n, 0) * fixedPoints(n, 0)
                            + fixedPoints(n, 1) * fixedPoints(n, 1)
                            + fixedPoints(n, 2) * fixedPoints(n, 2));
        }
        for (int m = 0; m < mNumMovingPoints; ++m) {
            trYPY += mP1(m) * (movingPoints(m, 0) * movingPoints(m, 0)
                           + movingPoints(m, 1) * movingPoints(m, 1)
                           + movingPoints(m, 2) * movingPoints(m, 2));
            trPXY += mPX(m, 0) * movingPoints(m, 0)
                 + mPX(m, 1) * movingPoints(m, 1)
                 + mPX(m, 2) * movingPoints(m, 2);
        }

        mVariance = ( trXPX - 2*trPXY + trYPY) / (mNp * mNumDimensions);
        if (mVariance < 0) {
            mVariance = std::fabs(mVariance);
        } else if (mVariance == 0){
            mVariance = 10.0f * std::numeric_limits<float>::epsilon();
            mRegistrationConverged = true;
        }

        // Update displacement matrix
        mNonRigidDisplacement += GW;

        // Calculate iteration error and check for convergence
        mIterationError = std::fabs(varianceOld - mVariance);
        mRegistrationConverged =  mIterationError <= mTolerance;


        double endM = omp_get_wtime();
        timeM           += endM - startM;
        timeMParameters += timeEndParameters - startM;
        timeMBuffers    += timeEndBuffers - timeEndParameters;
        timeMSolve        += timeEndSolve - timeEndBuffers;
        timeMGW         += timeEndGW - timeEndSolve;
        timeMUpdate     += endM - timeEndGW;
    }

    void CoherentPointDriftNonRigid::lowRankApproximation(
            MatrixXf& matrix, unsigned int rank, MatrixXf& eigenvectorMatrix, MatrixXf& eigenvalueMatrix) {

        /* ******************************************************
         * Fast randomized range finder (alg #4.5 in Halko et.al)
         * *****************************************************/

        unsigned int numRows = matrix.rows();
        unsigned int numCols = matrix.cols();

        // Set random distribution
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine distributionEngine(seed);
        std::uniform_real_distribution<float> rand(0, 1);

        // Subsampled Random Fourier Transform (SRFT)
        auto randomColIdx = std::vector<int>(numCols);
        auto randomUniformSample = std::vector<float>(numCols);
        for (int i = 0; i < numCols; ++i) {
            randomColIdx[i] = i;
            float r = sqrtf(rand(distributionEngine));
            float theta = rand(distributionEngine) * 2.0f * (float) EIGEN_PI;
            randomUniformSample[i] = r*cosf(theta);
        }

        MatrixXf SRFT = MatrixXf::Zero(numCols, rank);
        std::shuffle(randomColIdx.begin(), randomColIdx.end(), distributionEngine);
        for (int col = 0; col < rank; col++) {
            int colIdx = randomColIdx[col];
            for (int row = 0; row < numCols; ++row) {
                SRFT(row, col) =    randomUniformSample[row]
                                    * cosf(2.0f * (float) EIGEN_PI * row * colIdx / (float) numCols)
                                    / sqrtf((float)numCols);
            }
        }

        // Find orthonormal basis matrix Q for input matrix sampled with SRFT
        double timeQ = omp_get_wtime();
        MatrixXf sampleMatrix = matrix * SRFT;

        // QR-decomposition - Eigen implementaiton
//        auto qr = sampleMatrix.colPivHouseholderQr();
//        MatrixXf Q2 = qr.matrixQ();
//        auto Q = Q2.block(0, 0, numRows, rank);
        double timeEndQREigen = omp_get_wtime();

         // QR-decomposition by Modified Gram-Schmidt with reorthogonalization
        MatrixXf Q = sampleMatrix;
        for (int k = 0; k < rank; ++k) {
            float tt = 0.0f;
            float t = Q.col(k).norm();
            bool reorthogonalize = true;
            while (reorthogonalize) {
                for (int i = 0; i < k; ++i) {
                    float s = Q.col(i).transpose()*Q.col(k);
                    Q.col(k) -= s * Q.col(i);
                }
                tt = Q.col(k).norm();
                reorthogonalize = false;
                if (tt < t/10.0f) {
                    t = tt;
                    reorthogonalize = true;
                }
            }
            Q.col(k) /= tt;
        }
        double timeEndQGS = omp_get_wtime();

        /* ***************************************************
         * Direct eigenvalue decomposition. Alg. #5.3 in Halko
         * **************************************************/
        double timeEigen = omp_get_wtime();
        MatrixXf B = Q.adjoint() * matrix * Q;
        Eigen::SelfAdjointEigenSolver<MatrixXf> es(B);
        eigenvalueMatrix = es.eigenvalues().asDiagonal();
        eigenvectorMatrix = Q * es.eigenvectors();

        double timeStop = omp_get_wtime();
//    std::cout << "COMP TIMES\n"
//              << "SRFT: " << timeQ - timeStart << std::endl
//              << "Orthogonal basis Q: " << timeEndQGS - timeQ << std::endl
//              << "Eigen: " << timeStop - timeEigen << std::endl;
    }

    void CoherentPointDriftNonRigid::createBuffers() {
        mP1Buffer = cl::Buffer(mDevice->getContext(),
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               mNumMovingPoints*sizeof(float),
                               mP1.data());
        mMovingPointsBuffer = cl::Buffer(mDevice->getContext(),
                                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         mNumMovingPoints*mNumDimensions*sizeof(float),
                                         mMovingPoints.data());
        mPXBuffer = cl::Buffer(mDevice->getContext(),
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               mNumMovingPoints*mNumDimensions*sizeof(float),
                               mPX.data());
        mRHSBuffer = cl::Buffer(mDevice->getContext(),
                                CL_MEM_WRITE_ONLY,
                                mNumMovingPoints*mNumDimensions*sizeof(float));
    }

    void CoherentPointDriftNonRigid::calculateRHSKernel(MatrixXf& RHS, unsigned int rows, unsigned int cols) {
        mCalculateRHSKernel.setArg(0, mP1Buffer);
        mCalculateRHSKernel.setArg(1, mPXBuffer);
        mCalculateRHSKernel.setArg(2, mMovingPointsBuffer);
        mCalculateRHSKernel.setArg(3, mRHSBuffer);

        mQueue.enqueueNDRangeKernel(
                mCalculateRHSKernel,
                cl::NullRange,
                cl::NDRange(rows, cols),
                cl::NullRange
        );

        mQueue.enqueueReadBuffer(mRHSBuffer,
                                 CL_TRUE,
                                 0,
                                 rows * cols * sizeof(float),
                                 RHS.data()
        );
    }

    void CoherentPointDriftNonRigid::saveResultsToTextFile() {
        std::ofstream results;
        results.open (mResultFilename, std::ios::out | std::ios::app);
        results     << mUniformWeight << " " << mLowRankApproxRank << " "
                    << mBeta << " " << mLambda << " " << mIteration-1 << " "
                    << mErrorPoints  << " " << mErrorDeformed << " "
                    << mErrorNonDeformed << " " << mErrorMax << " "
                    << timeTot << " " << mTimeAffinity << " " << mTimeLowRank << " " << timeEM << " "
                    << timeE << " " << timeM << " "
                    << timeEBuffers << " "<< timeEKernel1 << " " << timeEKernel2 << " " << timeEUseful << " "
                    << timeMBuffers << " "<< timeMSolve << " " << timeMGW << " " << timeMUpdate;
        if (mLandmarks) {
            results << " " << mErrorLandmarksMeanDistPrereg << " " << mErrorLandmarksMeanDistPostreg
                    << " " << mErrorLandmarksMax
                    << std::endl;
        } else {
            results << std::endl;
        }
        results.close();
    }

}