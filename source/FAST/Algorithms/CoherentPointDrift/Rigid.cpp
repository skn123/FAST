#include "CoherentPointDrift.hpp"
#include "Rigid.hpp"

#include <limits>
#include <iostream>
#include <fstream>

namespace fast {

    CoherentPointDriftRigid::CoherentPointDriftRigid() {
        mScale = 1.0;
        mTransformationType = TransformationType::RIGID;
    }

    void CoherentPointDriftRigid::initializeVarianceAndMore() {
    }

    void CoherentPointDriftRigid::maximization(MatrixXf& fixedPoints, MatrixXf& movingPoints) {
        double startM = omp_get_wtime();

        // Estimate new mean vectors
        MatrixXf fixedMean = fixedPoints.transpose() * mPt1 / mNp;
        MatrixXf movingMean = movingPoints.transpose() * mP1 / mNp;
        double timeEndMCenter = omp_get_wtime();

        // Single value decomposition (SVD)
        const MatrixXf A = mPX.transpose() * movingPoints - mNp * fixedMean * movingMean.transpose();
        auto svdU =  A.bdcSvd(Eigen::ComputeThinU);
        auto svdV =  A.bdcSvd(Eigen::ComputeThinV);
        const MatrixXf* U = &svdU.matrixU();
        const MatrixXf* V = &svdV.matrixV();
        VectorXf singularValues = svdU.singularValues();

        MatrixXf UVt = *U * V->transpose();
        Eigen::RowVectorXf C = Eigen::RowVectorXf::Ones(mNumDimensions);
        C[mNumDimensions-1] = UVt.determinant();

        double timeEndMSVD = omp_get_wtime();


        /* ************************************************************
         * Find transformation parameters: rotation, scale, translation
         * ***********************************************************/
        mRotation = *U * C.asDiagonal() * V->transpose();
        MatrixXf AtR = A.transpose() * mRotation;

        float traceAtR = 0.0f;
        float traceYPY = 0.0f;
        float traceXPX = 0.0f;
        for (int d = 0; d < mNumDimensions - 1; ++d) {
            traceAtR += singularValues(d);
        }
        traceAtR += singularValues(mNumDimensions-1) * C[mNumDimensions-1];

        for (int m = 0; m < mNumMovingPoints; ++m) {
            traceYPY += mP1(m) * (movingPoints.row(m).squaredNorm());
        }
        traceYPY -= mNp * movingMean.squaredNorm();

        for (int n = 0; n < mNumFixedPoints; ++n) {
            traceXPX += mPt1(n) * (fixedPoints.row(n).squaredNorm());
        }
        traceXPX -= mNp * fixedMean.squaredNorm();

        mScale = traceAtR / traceYPY;
        mTranslation = fixedMean - mScale * mRotation * movingMean;

        // Update variance
        double varianceOld = mVariance;
        mVariance = ( traceXPX - mScale * traceAtR ) / (mNp * mNumDimensions);
        if (mVariance < 0) {
            mVariance = std::fabs(mVariance);
        } else if (mVariance == 0){
            mVariance = 10.0f * std::numeric_limits<float>::epsilon();
            mRegistrationConverged = true;
        }
        double timeEndMParameters = omp_get_wtime();


        /* ****************
         * Update transform
         * ***************/
        Affine3f iterationTransform = Affine3f::Identity();
        iterationTransform.translation() = Vector3f(mTranslation);
        iterationTransform.linear() = mRotation;
        iterationTransform.scale(float(mScale));

        Affine3f currentRegistrationTransform;
        MatrixXf registrationMatrix = iterationTransform.matrix() * mTransformation->getTransform().matrix();
        currentRegistrationTransform.matrix() = registrationMatrix;
        mTransformation->setTransform(currentRegistrationTransform);


        /* *************************
         * Transform the point cloud
         * ************************/
        MatrixXf movingPointsTransformed =
                mScale * movingPoints * mRotation.transpose() + mTranslation.transpose().replicate(mNumMovingPoints, 1);
        movingPoints = movingPointsTransformed;


        /* **************************************************
         * Calculate change in error and check for convergence
         * **************************************************/
        mIterationError = std::fabs(mVariance - varianceOld);
        mRegistrationConverged =  mIterationError <= mTolerance;


        double endM = omp_get_wtime();
        timeM += endM - startM;
        timeMCenter += timeEndMCenter - startM;
        timeMSolve += timeEndMSVD - timeEndMCenter;
        timeMParameters += timeEndMParameters - timeEndMSVD;
        timeMUpdate += endM - timeEndMParameters;
    }

    void CoherentPointDriftRigid::saveResultsToTextFile() {
        std::ofstream results;
        results.open (mResultFilename, std::ios::out | std::ios::app);
        results     << mUniformWeight << " " << mIteration-1 << " "
                    << mErrorPoints << " "<< mErrorTransformation << " " << mErrorRotation << " "
                    << timeTot << " "<< timeEM << " " << timeE << " " << timeM << " "
                    << timeEBuffers << " "<< timeEKernel1 << " " << timeEKernel2 << " " << timeEUseful << " "
                    << timeMCenter << " " << timeMSolve << " "<< timeMParameters << " " << timeMUpdate
                    << std::endl;
        results.close();
    }
}