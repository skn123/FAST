#include "FAST/Algorithms/CoherentPointDrift/CoherentPointDrift.hpp"
#include "FAST/SceneGraph.hpp"
#include "CoherentPointDrift.hpp"

#include "FAST/Algorithms/CoherentPointDrift/Rigid.hpp"

#undef min
#undef max
#include <limits>

#include <iostream>
#include <FAST/Visualization/ImageRenderer/ImageRenderer.hpp>

namespace fast {

    CoherentPointDrift::CoherentPointDrift() {
        createInputPort<Mesh>(0);
        createInputPort<Mesh>(1);
        createOutputPort<Mesh>(0);
        mMaxIterations = 100;
        mIteration = 0;
        mTolerance = 1e-6;
        mUniformWeight = 0.5;
        mTransformation = AffineTransformation::New();
        mRegistrationConverged = false;
        mScale = 1.0;
        mIterationError = 100*mTolerance;
        mFixedNormalizationScale = 1.0;
        mMovingNormalizationScale = 1.0;
        mFixedMeanInitial = MatrixXf::Zero(3, 1);
        mMovingMeanInitial = MatrixXf::Zero(3, 1);

        timeE = 0.0;
        timeENormal = 0.0;
        timeEPosteriorDivision = 0.0;
        timeM = 0.0;
        timeMUseful = 0.0;
        timeMCenter = 0.0;
        timeMSVD = 0.0;
        timeMParameters = 0.0;
        timeMUpdate = 0.0;
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

    void CoherentPointDrift::normalizePointSets() {

        // Center point clouds around origin, i.e. zero mean
        mFixedMeanInitial = mFixedPoints.colwise().sum() / mNumFixedPoints;
        mMovingMeanInitial = mMovingPoints.colwise().sum() / mNumMovingPoints;
        mFixedPoints -= mFixedMeanInitial.replicate(mNumFixedPoints, 1);
        mMovingPoints -= mMovingMeanInitial.replicate(mNumMovingPoints, 1);

        // Scale point clouds to have unit variance
        mFixedNormalizationScale = sqrt(mFixedPoints.cwiseProduct(mFixedPoints).sum() / (double)mNumFixedPoints);
        mMovingNormalizationScale = sqrt(mMovingPoints.cwiseProduct(mMovingPoints).sum() / (double)mNumMovingPoints);
        mFixedPoints /= mFixedNormalizationScale;
        mMovingPoints /= mMovingNormalizationScale;
    }

    void CoherentPointDrift::printCloudDimensions() {
        std::cout << "\n****************************************\n";
        std::cout << "Fixed point cloud: " << mNumFixedPoints << " x " << mNumDimensions << std::endl;
        std::cout << "Moving point cloud: " << mNumMovingPoints << " x " << mNumDimensions << std::endl;
    }

    void CoherentPointDrift::expectation(MatrixXf& fixedPoints, MatrixXf& movingPoints) {

        double timeStartE = omp_get_wtime();

        /* **********************************************************************************
         * Calculate distances between the points in the two point sets
         * Let row i in P equal the squared distances from all fixed points to moving point i
         * *********************************************************************************/

        /* Implementation without OpenMP */
        MatrixXf movingPointMatrix = MatrixXf::Zero(mNumMovingPoints, mNumFixedPoints);
        MatrixXf distances = MatrixXf::Zero(mNumMovingPoints, mNumFixedPoints);
        for (int i = 0; i < mNumMovingPoints; ++i) {
            movingPointMatrix = movingPoints.row(i).replicate(mNumFixedPoints, 1);
            distances = fixedPoints - movingPoints.row(i).replicate(mNumFixedPoints, 1);
            distances = fixedPoints - movingPointMatrix;            // Distance between all fixed points and moving point i
            distances = distances.cwiseAbs2();                            // Square distance components (3xN)
            mResponsibilityMatrix.row(i) = distances.rowwise().sum();   // Sum x, y, z components (1xN)
        }


        auto c = (float) (pow(2*(double)EIGEN_PI*mVariance, (double)mNumDimensions/2.0)
                          * (mUniformWeight/(1-mUniformWeight)) * (float)mNumMovingPoints/mNumFixedPoints);

//#pragma omp parallel for //collapse(2)
//        for (int col = 0; col < mNumFixedPoints; ++col) {
//            for (int row = 0; row < mNumMovingPoints; ++row) {
//                double norm = (fixedPoints.row(col) - movingPoints.row(row)).squaredNorm();
//                mResponsibilityMatrix(row, col) = exp(norm / (-2.0 * mVariance));
//            }
//        }
        double timeEndFirstLoop = omp_get_wtime();
//

        mResponsibilityMatrix *= -1.0/(2.0 * mVariance);
        mResponsibilityMatrix = mResponsibilityMatrix.array().exp();

        MatrixXf denominatorRow = mResponsibilityMatrix.colwise().sum();
        denominatorRow =  denominatorRow.array() + c;

        // Ensure that one does not divide by zero
        MatrixXf shouldBeLargerThanEpsilon = Eigen::NumTraits<float>::epsilon() * MatrixXf::Ones(1, mNumFixedPoints);
        denominatorRow = denominatorRow.cwiseMax(shouldBeLargerThanEpsilon);
        MatrixXf denominator = denominatorRow.replicate(mNumMovingPoints, 1);

        mResponsibilityMatrix = mResponsibilityMatrix.cwiseQuotient(denominator);


//#pragma omp parallel for
//        for (int col = 0; col < mNumFixedPoints; ++col) {
//            float denom = mResponsibilityMatrix.col(col).sum() + c;
//            mResponsibilityMatrix.col(col) /= max(denom, Eigen::NumTraits<float>::epsilon() );
//        }

        // Update computation times
        double timeEndE = omp_get_wtime();
//        timeENormal += timeEndFirstLoop - timeStartE;
//        timeEPosteriorDivision += timeEndE - timeEndFirstLoop;
        timeE += timeEndE - timeStartE;
    }

    void CoherentPointDrift::execute() {

        double timeStart = omp_get_wtime();

        // Store the point sets in matrices and store their dimensions
        initializePointSets();
        printCloudDimensions();

        // Apply the existing transform, if any, to moving point cloud
        auto existingTransform = Affine3f::Identity();
        existingTransform = SceneGraph::getEigenAffineTransformationFromData(mMovingMesh);
        mMovingPoints = mMovingPoints.rowwise().homogeneous() * existingTransform.affine().transpose();

        // Normalize the point sets, i.e. zero mean and unit variance
//        normalizePointSets();

        // Initialize variance and error
        initializeVarianceAndMore();


        /* *************************
         * Get some points drifting!
         * ************************/
        double timeStartEM = omp_get_wtime();

        while (mIteration < mMaxIterations && !mRegistrationConverged) {
//            std::cout << "ITERATION " << (int) mIteration << std::endl;
            expectation(mFixedPoints, mMovingPoints);
            maximization(mFixedPoints, mMovingPoints);
            mIteration++;
        }


        /* *****************
         * Computation times
         * ****************/
        double timeEndEM = omp_get_wtime();
        double timeTotalEM = timeEndEM - timeStartEM;

//        std::cout << "\nCOMPUTATION TIMES:\n";
//        std::cout << "Initialization of point sets and normalization: " << timeStartEM-timeStart << " s.\n";
//        std::cout << "EM converged in " << mIteration-1 << " iterations in " << timeTotalEM << " s.\n";
//        std::cout << "Time spent on expectation: " << timeE << " s\n";
//        std::cout << "      - Normal distribution: " << timeENormal << " s.\n";
//        std::cout << "      - Posterior GMM probabilities, division: " << timeEPosteriorDivision << " s.\n";
//        std::cout << "Time spent on maximization: " << timeM << " s\n";
//        std::cout << "      - Calculating P1, Pt1, Np: " << timeMUseful << " s.\n";
//        std::cout << "      - Centering point clouds: " << timeMCenter << " s.\n";
//        std::cout << "      - SVD (rigid): " << timeMSVD << " s.\n";
//        std::cout << "      - Calculation transformation parameters: " << timeMParameters << " s.\n";
//        std::cout << "      - Updating transformation and error: " << timeMUpdate << " s.\n";

        std::cout << mIteration-1 << std::endl;
        std::cout <<timeTotalEM << std::endl;
        std::cout <<timeE << std::endl;
        std::cout <<timeENormal << std::endl;
        std::cout <<timeEPosteriorDivision << std::endl;
        std::cout <<timeM << std::endl;
        std::cout <<timeMUseful << std::endl;
        std::cout <<timeMCenter<< std::endl;
        std::cout <<timeMSVD<< std::endl;
        std::cout <<timeMParameters << std::endl;
        std::cout <<timeMUpdate << std::endl;


        /* ***********************************************
         * Denormalize and set total transformation matrix
         * **********************************************/
        // Set normalization
        Affine3f normalization = Affine3f::Identity();
        normalization.translate((Vector3f) -(mMovingMeanInitial).transpose());

        // Denormalize moving point set

        mScale *= mFixedNormalizationScale / mMovingNormalizationScale;
        Affine3f registration = mTransformation->getTransform();
        registration.scale((float) mScale);
        registration.translation() *= mFixedNormalizationScale;

        Affine3f denormalization = Affine3f::Identity();
        denormalization.translate((Vector3f) (mFixedMeanInitial).transpose());

        // Set total transformation
        auto transform = AffineTransformation::New();
        Affine3f registrationTransformTotal = denormalization * registration * normalization;
        transform->setTransform(registrationTransformTotal * existingTransform);

        mMovingMesh->getSceneGraphNode()->setTransformation(transform);
        addOutputData(0, mMovingMesh);


        /* ******************
        * Registration error
        * *****************/
        double error = 0.0;
        double errorRotationDiffNorm = 0.0;
        double errorAffineDiffNorm = 0.0;
        unsigned int lastPoint = min(mNumFixedPoints, mNumMovingPoints);
#pragma omp parallel for
        for (int i = 0; i < lastPoint; i++) {
            error += (mFixedPoints.row(i)-mMovingPoints.row(i)).squaredNorm();
        }

        MatrixXf rotationDiff = registration.linear().inverse() - existingTransform.linear();
        MatrixXf affineDiff = registration.inverse().affine() - existingTransform.affine();
        errorRotationDiffNorm = rotationDiff.norm();
        errorAffineDiffNorm = affineDiff.norm();

        std::cout << "\nCOMPUTATION ERROR\n";
//        std::cout << "Squared norm error: " << error << std::endl;
//        std::cout << "Norm of rotation matrix difference: " << errorRotationDiffNorm << std::endl;
        std::cout << "Norm of total matrix difference: " << errorAffineDiffNorm << std::endl;

        mResults[0] = errorRotationDiffNorm;
        mResults[4] = errorAffineDiffNorm;
        mResults[1] = mIteration-1;
        mResults[2] = timeTotalEM;

        // Print some matrices
//        printOutputMatrices(existingTransform, registration, registrationTransformTotal);
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

    void CoherentPointDrift::setFixedMesh(Mesh::pointer data) {
        setInputData(0, data);
    }

    void CoherentPointDrift::setMovingMesh(Mesh::pointer data) {
        setInputData(1, data);
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

    void CoherentPointDrift::setTolerance(double tolerance) {
        mTolerance = tolerance;
    }

    AffineTransformation::pointer CoherentPointDrift::getOutputTransformation() {
        return mTransformation;
    }

}
