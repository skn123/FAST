#include "CoherentPointDrift.hpp"
#include "Affine.hpp"

#include <limits>
#include <iostream>
#include <fstream>

namespace fast {

    CoherentPointDriftAffine::CoherentPointDriftAffine() {
        mScale = 1.0;
        mTransformationType = TransformationType::AFFINE;
    }

    void CoherentPointDriftAffine::initializeVarianceAndMore() {}

    void CoherentPointDriftAffine::maximization(Eigen::MatrixXf &fixedPoints, Eigen::MatrixXf &movingPoints) {

        double startM = omp_get_wtime();

        // Estimate new mean vectors
        MatrixXf fixedMean = fixedPoints.transpose() * mPt1 / mNp;
        MatrixXf movingMean = movingPoints.transpose() * mP1 / mNp;
        double timeEndMCenter = omp_get_wtime();


        /* **********************************************************
         * Find transformation parameters: affine matrix, translation
         * *********************************************************/
        MatrixXf A = mPX.transpose() * movingPoints - mNp * fixedMean * movingMean.transpose();
        MatrixXf YPY = movingPoints.transpose() * mP1.asDiagonal() * movingPoints
                     - mNp * movingMean * movingMean.transpose();

        float traceXPX = 0.0f;
        for (int n = 0; n < mNumFixedPoints; ++n) {
            traceXPX += mPt1(n) * (fixedPoints.row(n).squaredNorm());
        }
        traceXPX -= mNp * fixedMean.squaredNorm();

        mAffineMatrix = A * YPY.inverse();
        mTranslation = fixedMean - mAffineMatrix * movingMean;

        // Update variance
        double varianceOld = mVariance;
        MatrixXf ABt = A * mAffineMatrix.transpose();
        mVariance = ( traceXPX - ABt.trace() ) / (mNp * mNumDimensions);
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
        iterationTransform.linear() = mAffineMatrix;

        Affine3f currentRegistrationTransform;
        MatrixXf registrationMatrix = iterationTransform.matrix() * mTransformation->getTransform().matrix();
        currentRegistrationTransform.matrix() = registrationMatrix;
        mTransformation->setTransform(currentRegistrationTransform);


        /* *************************
         * Transform the point cloud
         * ************************/
        MatrixXf movingPointsTransformed =
                movingPoints * mAffineMatrix.transpose() + mTranslation.transpose().replicate(mNumMovingPoints, 1);
        movingPoints = movingPointsTransformed;


        /* ***************************************************
         * Calculate iteration error and check for convergence
         * **************************************************/
        mIterationError = std::fabs(varianceOld - mVariance);
        mRegistrationConverged =  mIterationError <= mTolerance;

        double endM = omp_get_wtime();
        timeM += endM - startM;
        timeMCenter += timeEndMCenter - startM;
        timeMParameters += timeEndMParameters - timeEndMCenter;
        timeMUpdate += endM - timeEndMParameters;
    }

    void CoherentPointDriftAffine::saveResultsToTextFile() {
        std::ofstream results;
        results.open (mResultFilename, std::ios::out | std::ios::app);
        results     << mUniformWeight << " " << mIteration-1 << " "
                    << mErrorPoints << " "<< mErrorTransformation << " " << mErrorRotation << " "
                    << timeTot << " "<< timeEM << " " << timeE << " " << timeM << " "
                    << timeEBuffers << " "<< timeEKernel1 << " " << timeEKernel2 << " " << timeEUseful << " "
                    << timeMCenter << " "<< timeMParameters << " " << timeMUpdate
                    << std::endl;
        results.close();
    }

}
