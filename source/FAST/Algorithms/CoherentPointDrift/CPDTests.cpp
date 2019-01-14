#include <FAST/Visualization/SimpleWindow.hpp>
#include <FAST/Importers/VTKMeshFileImporter.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include "FAST/Exporters/VTKMeshFileExporter.hpp"
#include "FAST/Exporters/VTKMeshFileExporter.hpp"
#include <FAST/Visualization/ImageRenderer/ImageRenderer.hpp>
#include "FAST/Visualization/VertexRenderer/VertexRenderer.hpp"
#include "FAST/Visualization/TriangleRenderer/TriangleRenderer.hpp"
#include "FAST/Algorithms/SurfaceExtraction/SurfaceExtraction.hpp"
#include <FAST/Algorithms/IterativeClosestPoint/IterativeClosestPoint.hpp>
#include "FAST/Testing.hpp"

#include "CoherentPointDrift.hpp"
#include "Rigid.hpp"
#include "Affine.hpp"
#include "NonRigid.hpp"
#include "CPDTestsUtils.hpp"

#include <random>
#include <iostream>
#include <fstream>

using namespace fast;

// Set registration settings
bool VISUALIZE_REGISTRATION = false;
bool NORMALIZE_POINTS_BEFORE_REG = true;
bool NORMALIZE_POITNS_IN_CPD = false;
bool APPLY_TRANSFORMATION = true;
bool ADD_NOISE = false;

// Set additional non-rigid registration settings
bool APPLY_DEFORMATION = false;
bool INCLUDE_LANDMARKS = false;

// Set registration parameters
float TOLERANCE = 1e-6;
unsigned int MAX_ITERATIONS = 100;
int POINT_SIZE = 5000;
int NUM_RUNS = 5;
float BETA = 2.0;
float LAMBDA = 2.0;

// Set deformation/transformation parameters
float NOISE_STDEV = 0.10;
float UNIFORM_WEIGHT = 0.5f;
float ROTATION_DEG = 50.0f;
float DEFORMATION_RADIUS = 0.5;

int BRAIN_SHIFT_PATIENT = 1;

TEST_CASE("cpd rigid", "[fast][coherentpointdrift][visual][cpd]") {

    // Set registration settings
    float uniformWeight = 0.0;
    unsigned int numRuns = NUM_RUNS;
    int pointSize = POINT_SIZE;

    // Set transformation settings
    float scale = 1.0;
    float rotationDeg = ROTATION_DEG;
    Vector3f translation(0.0f, 0.0f, 0.0f);
    Vector3f rotationAxis = Vector3f::UnitY();

//    auto dataset1 = "Surface_LV.vtk";
//    auto dataset2 = "Surface_LV.vtk";
//    auto fixedDataset = "Bunny/bunny.vtk";
//    auto movingDataset = "Bunny/bunny.vtk";
    auto fixedDataset = "ROMO/CT_Abdomen_surface_front.vtk";
    auto movingDataset = "ROMO/CT_Abdomen_surface_front.vtk";
//    auto dataset2 = "ROMO/GM_hands_down.vtk";


    // Load point clouds
    auto fixedPointMesh = getPointCloud(fixedDataset);
    auto movingPointMesh= getPointCloud(movingDataset);
    auto refPointMesh = getPointCloud(movingDataset);

    // Modify point clouds

    downsample(fixedPointMesh, pointSize);
    downsample(movingPointMesh, pointSize);
    downsample(refPointMesh, pointSize);

    if (NORMALIZE_POINTS_BEFORE_REG) {
        normalizePointCloud(fixedPointMesh);
        normalizePointCloud(movingPointMesh);
        normalizePointCloud(refPointMesh);
    }

//    addGaussianNoise(fixedPointMesh, 0.12f*0.12f);
//    addGaussianNoise(movingPointMesh, powf(0.2f,2));
//    modifyPointCloud(fixedPointMesh, 1.0, 0.3);
//    modifyPointCloud(movingPointMesh, 1.0, 0.3);
//    modifyPointCloud(movingPointMesh, 1.0, 0.3);
//    keepPartOfPointCloud(fixedPointMesh, 0, 1000);
//    keepPartOfPointCloud(movingPointMesh, 200, 1000);

    // Find point set sizes
    unsigned int numFixedPoints =  fixedPointMesh.get()->getNrOfVertices();
    unsigned int numMovingPoints =  movingPointMesh.get()->getNrOfVertices();
    unsigned int numDimensions = getPointCloudFromMesh(fixedPointMesh).cols();

    // Transform one of the point clouds
    Affine3f affine = Affine3f::Identity();
    if (APPLY_TRANSFORMATION) {
        auto transform = AffineTransformation::New();
        affine.rotate(Eigen::AngleAxisf(3.141592f / 180.0f * rotationDeg, rotationAxis));
        affine.scale(scale);
//    affine.translate(translation);
        transform->setTransform(affine);

        movingPointMesh->getSceneGraphNode()->setTransformation(transform);
        refPointMesh->getSceneGraphNode()->setTransformation(transform);
    }

    if (NORMALIZE_POINTS_BEFORE_REG) {
        normalizePointCloud(fixedPointMesh);
        normalizePointCloud(movingPointMesh);
        normalizePointCloud(refPointMesh);
    }

    // Prepare result text file
    std::string filename = "CPD-Rigid-Bunny" + std::to_string(pointSize) + '-'
                            + currentDateTime() + ".txt";
    std::ofstream results;
    results.open (filename, std::ios::out | std::ios::app);
    results << "CPD-Rigid\n";
    results << "Fixed " << fixedDataset << " " << numFixedPoints << "x" << numDimensions << std::endl;
    results << "Moving " << movingDataset << " " << numMovingPoints << "x" << numDimensions << std::endl;
    results << "MaxIterations " << MAX_ITERATIONS << std::endl;
    results << "Tolerance " << TOLERANCE << std::endl;
    if (NORMALIZE_POINTS_BEFORE_REG) {
        results << "Points normalized before registration." << std::endl;
    } else {
        results << "Point NOT normalizes before registration." << std::endl;
    }
    if (APPLY_TRANSFORMATION) {
        results << "Transform_applied rotationDeg " << rotationDeg
                << " rotationAxis (" << rotationAxis.x() << "," << rotationAxis.y() << "," << rotationAxis.z()
                << ") scale " << scale
                << " translation (" << translation.x() << "," << translation.y() << "," << translation.z() << ")\n";
        results << affine.affine() << std::endl;
    } else {
        results << "No_transformation_applied\n";
    }
    results << "uniformWeight iterations errorPoints errorTransformation errorRot ";
    results << "timeTot timeEM timeE timeM timeEBuffers timeK1 timeK2 timeNp timeMean timeSVD timeParam timeUpdate\n";
    results.close();

    unsigned int run = 0;
    while (run++ < numRuns) {

        std::cout << "CPD Rigid: Iteration " << run << " of " << numRuns << std::endl;

        // Run Coherent Point Drift
        auto cpd = CoherentPointDriftRigid::New();
        cpd->setFixedMesh(fixedPointMesh);
        cpd->setMovingMesh(movingPointMesh);
        cpd->setMaximumIterations(MAX_ITERATIONS);
        cpd->setTolerance(TOLERANCE);
        cpd->setUniformWeight(uniformWeight);
        cpd->setResultTextFileName(filename);
        cpd->setNormalization(NORMALIZE_POITNS_IN_CPD);
        cpd->setUpdateOutputMesh(VISUALIZE_REGISTRATION);

        if (VISUALIZE_REGISTRATION) {
            auto renderer = VertexRenderer::New();
            renderer->addInputData(fixedPointMesh, Color::Red(), 1.5);                        // Fixed points
            renderer->addInputData(refPointMesh, Color::Black(), 1.5);                         // Moving points
            renderer->addInputConnection(cpd->getOutputPort(), Color::Blue(), 2.0);      // Moving points registered

            auto window = SimpleWindow::New();
            window->addRenderer(renderer);
//        window->setTimeout(1000);
            window->start();
        } else {
            cpd->update(0);
        }
    }
}

TEST_CASE("cpd affine", "[fast][coherentpointdrift][visual][cpd]") {

    // Set registration settings
    float uniformWeight = 0.0;
    unsigned int numRuns = 2;

    // Set transformation parameters
    float scale = 0.5;
    float rotationDeg = 50.0f;
    Vector3f translation(-0.052f, 0.005f, -0.001f);
    Vector3f rotationAxis = Vector3f::UnitY();

//    auto dataset1 = "Surface_LV.vtk";
//    auto dataset2 = "Surface_LV.vtk";
    auto fixedDataset = "Bunny/bunny.vtk";
    auto movingDataset = "Bunny/bunny.vtk";
//    auto dataset1 = "ROMO/A_hands_down.vtk";
//    auto dataset2 = "ROMO/GM_hands_down.vtk";

    // Load point clouds
    auto fixedPointMesh = getPointCloud(fixedDataset);
    auto movingPointMesh= getPointCloud(movingDataset);
    auto refPointMesh = getPointCloud(movingDataset);

    // Modify point clouds
    downsample(fixedPointMesh, 3000);
    downsample(movingPointMesh, 3050);
    downsample(movingPointMesh, 3000);

    if (NORMALIZE_POINTS_BEFORE_REG) {
        normalizePointCloud(fixedPointMesh);
        normalizePointCloud(movingPointMesh);
        normalizePointCloud(refPointMesh);
    }

    // Find point set sizes
    unsigned int numFixedPoints =  fixedPointMesh.get()->getNrOfVertices();
    unsigned int numMovingPoints =  movingPointMesh.get()->getNrOfVertices();
    unsigned int numDimensions = getPointCloudFromMesh(fixedPointMesh).cols();

    // Transform one of the point clouds
    Affine3f affine = Affine3f::Identity();
    if (APPLY_TRANSFORMATION) {
        auto transform = AffineTransformation::New();
        MatrixXf shearing = Matrix3f::Identity();
        shearing(0, 0) = 0.3;
        shearing(0, 1) = 1.2;
        shearing(1, 0) = 0.0;
        affine.rotate(Eigen::AngleAxisf(3.141592f / 180.0f * rotationDeg, rotationAxis));
        affine.scale(scale);
//    affine.translate(translation);
        affine.linear() += shearing;
        transform->setTransform(affine);

        movingPointMesh->getSceneGraphNode()->setTransformation(transform);
        refPointMesh->getSceneGraphNode()->setTransformation(transform);
    }

    // Prepare result text file
    std::string filename = "CPD-Affine" + currentDateTime() + ".txt";
    std::ofstream results;
    results.open (filename, std::ios::out | std::ios::app);
    results << "CPD-Affine\n";
    results << "Fixed " << fixedDataset << " " << numFixedPoints << "x" << numDimensions << std::endl;
    results << "Moving " << movingDataset << " " << numMovingPoints << "x" << numDimensions << std::endl;
    results << "MaxIterations " << MAX_ITERATIONS << std::endl;
    results << "Tolerance " << TOLERANCE << std::endl;
    if (APPLY_TRANSFORMATION) {
        results << "Transform_applied rotationDeg " << rotationDeg
                << " rotationAxis (" << rotationAxis.x() << "," << rotationAxis.y() << "," << rotationAxis.z()
                << ") scale " << scale
                << " translation (" << translation.x() << "," << translation.y() << "," << translation.z() << ")\n";
        results << affine.affine() << std::endl;
    } else {
        results << "No_transformation_applied\n";
    }
    results << "uniformWeight iterations errorPoints errorTransformation errorRot ";
    results << "timeTot timeEM timeE timeM timeBuffers timeK1 timeK2 timeNp timeMean timeParam timeUpdate\n";
    results.close();

    unsigned int run = 0;
    while (run++ < numRuns) {

        std::cout << "Iteration " << run << " of " << numRuns << std::endl;

        // Run Coherent Point Drift
        auto cpd = CoherentPointDriftAffine::New();
        cpd->setFixedMesh(fixedPointMesh);
        cpd->setMovingMesh(movingPointMesh);
        cpd->setMaximumIterations(MAX_ITERATIONS);
        cpd->setTolerance(TOLERANCE);
        cpd->setUniformWeight(uniformWeight);
        cpd->setResultTextFileName(filename);
        cpd->setNormalization(NORMALIZE_POITNS_IN_CPD);
        cpd->setUpdateOutputMesh(VISUALIZE_REGISTRATION);

        if (VISUALIZE_REGISTRATION) {
            auto renderer = VertexRenderer::New();
            renderer->addInputData(fixedPointMesh, Color::Red(), 1.5);                        // Fixed points
//        renderer->addInputData(movingPointMesh, Color::Blue(), 2.0);                        // Moving points initial
//        renderer->addInputData(refPointMesh, Color::Black(), 2.0);                         // Moving points
            renderer->addInputConnection(cpd->getOutputPort(), Color::Blue(), 2.0);      // Moving points registered

            auto window = SimpleWindow::New();
            window->addRenderer(renderer);
//        window->setTimeout(1000);
            window->start();
        } else {
            cpd->update(0);
        }
    }
}

TEST_CASE("cpd nonrigid", "[fast][coherentpointdrift][visual][cpd]") {

    std::string fixedDataset = "Bunny/bunny.vtk";
    std::string movingDataset = "Bunny/bunny.vtk";
//    auto fixedDataset = "ROMO/A_hands_down.vtk";
//    auto movingDataset = "ROMO/GM_hands_down.vtk";
//    std::string fixedDataset = "BrainShift/1/MRA_TOF_FOV_cl1.vtk";
//    std::string movingDataset = "BrainShift/1/US1_cl1.vtk";

//    std::vector<float> noiseStdevs = {0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.24};
//    std::vector<float> noiseStdevs = {0.08, 0.1};
//    for (auto noiseStdev : noiseStdevs) {
//        std::cout << "Noise stdev: " << noiseStdev << std::endl;


    // Set registration settings
    float beta = BETA;
    float lambda = LAMBDA;
    float uniformWeight = UNIFORM_WEIGHT;
    unsigned int numRuns = NUM_RUNS;
    int pointSetSize = POINT_SIZE;
    float noiseStdev = NOISE_STDEV;

    // Set deformation settings
    float deformationRadius = DEFORMATION_RADIUS;
    Vector3f deformationCenter = Vector3f(0.0, 0.0, 0.0);

    // Set transformation settings
    float scale = 1.0;
    float rotationDeg = ROTATION_DEG;
    Vector3f translation(0.0f, 0.0f, 0.0f);
    Vector3f rotationAxis = Vector3f::UnitY();

//    std::vector<int> pointSetSizes = {500, 800, 1600, 3200, 6400, 12800};
//    for (auto size: pointSetSizes) {
//    int pointSetSize = size;
//    std::cout << "Point set size: " << pointSetSize << std::endl;

    // Load point clouds
    auto fixedPointMesh = getPointCloud(fixedDataset);
    auto movingPointMesh = getPointCloud(movingDataset);
    auto refPointMesh = getPointCloud(movingDataset);

    // Modify point clouds
    downsample(fixedPointMesh, pointSetSize);
    downsample(movingPointMesh, pointSetSize);
    downsample(refPointMesh, pointSetSize);

    if (NORMALIZE_POINTS_BEFORE_REG) {
        normalizePointCloud(fixedPointMesh);
        normalizePointCloud(movingPointMesh);
        normalizePointCloud(refPointMesh);
    }

    // Find point set sizes
    unsigned int numFixedPoints =  fixedPointMesh.get()->getNrOfVertices();
    unsigned int numMovingPoints =  movingPointMesh.get()->getNrOfVertices();
    unsigned int numDimensions = getPointCloudFromMesh(fixedPointMesh).cols();

    // Set low rank approx rank
    auto rank = (unsigned int) ceil(sqrt(numMovingPoints));

    // Deform point set
    std::vector<int> deformedPointsIndices = {};
    std::vector<int> deformedRefPointsIndices = {};
    if (APPLY_DEFORMATION) {
        deformPointCloud(movingPointMesh, deformationRadius, deformationCenter, deformedPointsIndices);
        deformPointCloud(refPointMesh, deformationRadius, deformationCenter, deformedRefPointsIndices);
    }

    // Transform one of the point clouds
    Affine3f affine = Affine3f::Identity();
    if (APPLY_TRANSFORMATION) {
        auto transform = AffineTransformation::New();
        affine.rotate(Eigen::AngleAxisf(M_PI / 180.0f * rotationDeg, rotationAxis));
        affine.scale(scale);
        transform->setTransform(affine);
        movingPointMesh->getSceneGraphNode()->setTransformation(transform);
        refPointMesh->getSceneGraphNode()->setTransformation(transform);
    }

    // Add noise
    if (ADD_NOISE) {
        addGaussianNoise(movingPointMesh, noiseStdev);
        addGaussianNoise(refPointMesh, noiseStdev);
    }

    if (NORMALIZE_POINTS_BEFORE_REG) {
        normalizePointCloud(fixedPointMesh);
        normalizePointCloud(movingPointMesh);
        normalizePointCloud(refPointMesh);
    }

    MatrixXf fixed  = getPointCloudFromMesh(fixedPointMesh);
    MatrixXf moving  = getPointCloudFromMesh(movingPointMesh);

    // Prepare result text file
    std::string filename = "CPD-Nonrigid-Compare-"
                           + std::to_string(pointSetSize) + '-'
//                           + std::to_string(noiseStdev) + '-'
                           + currentDateTime() + ".txt";
//    std::string filename = "CPD-Nonrigid-Betas-Lambda" + std::to_string(lambda)
//                        + '-' + currentDateTime() + ".txt";
//    std::string filename = "CPD-Nonrigid-ErrorLambdas-" + currentDateTime() + ".txt";
    std::ofstream results;
    results.open (filename, std::ios::out | std::ios::app);
    results << "CPD-Nonrigid\n";
    results << "Fixed " << fixedDataset << " " << numFixedPoints << "x" << numDimensions << std::endl;
    results << "Moving " << movingDataset << " " << numMovingPoints << "x" << numDimensions << std::endl;
    results << "MaxIterations " << MAX_ITERATIONS << std::endl;
    results << "Tolerance " << TOLERANCE << std::endl;
    if (APPLY_DEFORMATION) {
        results << "DeformationRadius " << deformationRadius << " deformationCenter ("
                << deformationCenter.x() << "," << deformationCenter.y() << "," << deformationCenter.z() << ")\n";
    } else {
        results << "No_deformation_applied\n";
    }
    results << "uniformWeight rank beta lambda iterations errorPoints errorDeformed errorNonDeformed errorMax ";
    results << "timeTot timeAffinity timeLowRank timeEM timeE timeM timeEBuffers timeK1 timeK2 timeNp ";
    results << "timeMBuffers timeSolveW timeGW timeUpdate\n";
    results.close();


    std::vector<float> lambdas = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0};
    std::vector<float> betas = {1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0};
//    for (auto beta : betas) {

        unsigned int run = 0;
//        std::cout << "Beta " << beta << std::endl;
        while (run++ < numRuns) {

            std::cout << "Run " << run << " of " << numRuns << std::endl;

            // Run Coherent Point Drift
            auto cpd = CoherentPointDriftNonRigid::New();
            cpd->setBeta(beta);
            cpd->setLambda(lambda);
            cpd->setLowRankApproxRank(rank);
            cpd->setUniformWeight(uniformWeight);
            cpd->setResultTextFileName(filename);
            cpd->setTolerance(TOLERANCE);
            cpd->setMaximumIterations(MAX_ITERATIONS);
            cpd->setNormalization(NORMALIZE_POITNS_IN_CPD);
            cpd->setUpdateOutputMesh(VISUALIZE_REGISTRATION);
            if (APPLY_DEFORMATION) {
                cpd->setDeformedIndices(deformedPointsIndices);
            }
            if (VISUALIZE_REGISTRATION) {
                cpd->setFixedMesh(fixedPointMesh);
                cpd->setMovingMesh(movingPointMesh);

                auto renderer = VertexRenderer::New();
                renderer->setDefaultColor(Color(0.0, 1.0, 0.0));
                renderer->setDefaultSize(1.5);
                renderer->addInputData(fixedPointMesh, Color::Red(), 2.0);                // Fixed points
//                renderer->addInputData(movingPointMesh, Color::Black(), 2.0);             // Moving points before transf/deformation
                renderer->addInputConnection(cpd->getOutputPort(), Color::Blue(), 2.0);   // Moving points registered

                auto window = SimpleWindow::New();
                window->addRenderer(renderer);
                window->start();
            } else {
                cpd->setFixedPoints(fixed);
                cpd->setMovingPoints(moving);
                cpd->execute();
            }
        }
//    } // for betas/lambdas
//    } // End noiseStdevs loop
//    } // End point set size loop
}

TEST_CASE("cpd lowrank accuracy", "[fast][coherentpointdrift][cpd]") {

    auto movingDataset = "Bunny/bunny.vtk";
    auto movingPointMesh = getPointCloud(movingDataset);

    // Set registration settings
    float beta = BETA;
    unsigned int numRuns = NUM_RUNS;
    unsigned int numMoving = POINT_SIZE;
    unsigned int rank = (int) ceilf(sqrt(numMoving));
//    unsigned int rank =  (unsigned int) ceil(sqrt(numMoving));

    downsample(movingPointMesh, numMoving);
    normalizePointCloud(movingPointMesh);

    MatrixXf movingPoints = getPointCloudFromMesh(movingPointMesh);
    unsigned int numMovingPoints =  movingPoints.rows();
    unsigned int numDimensions = movingPoints.cols();

    // Prepare result text file
    std::string filename = "LowRankAccuracyEigenOnly-Ranks-" + std::to_string(numMoving) + "-" + currentDateTime() + ".txt";
    std::ofstream results;
    results.open (filename, std::ios::out | std::ios::app);
    results << "LowRankAccuracyTest\n";
    results << "Moving " << movingDataset << " " << numMovingPoints << "x" << numDimensions << std::endl;
    results << "rank beta timeLowRankEigen timeProduct errorNormEigen timeLowRankGS errorGS\n";
    results.close();

//    std::vector<unsigned int> ranks = {10, 12, 16, 20, 25, 60, 80};
    std::vector<unsigned int> ranks = {5, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 80, 100, 150};
//    std::vector<float> betas= {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0};
//    for(auto beta : betas) {
    for(auto rank : ranks) {


            MatrixXf G = MatrixXf::Zero(numMovingPoints, numMovingPoints);
        #pragma omp parallel for
            for (int i = 0; i < numMovingPoints; ++i) {
                for (int j = 0; j < i; ++j) {
                    float norm = (movingPoints.row(i) - movingPoints.row(j)).squaredNorm();
                    G(i, j) = expf(norm / (-2.0f * beta * beta));
                    G(j, i) = G(i, j);
                }
                G(i, i) = 1.0f;
            }

        std::cout << "Rank: " << rank  << std::endl;
//        std::cout << "Beta: " << beta << std::endl;

        unsigned int run = 0;
        while (run++ < numRuns) {

            MatrixXf Lambda = MatrixXf::Zero(rank, rank);
            MatrixXf Q = MatrixXf::Zero(numMoving, rank);
            MatrixXf S = MatrixXf::Zero(rank, rank);
            MatrixXf U = MatrixXf::Zero(numMoving, rank);
            MatrixXf S2 = MatrixXf::Zero(rank, rank);
            MatrixXf U2 = MatrixXf::Zero(numMoving, rank);

            double timeLowRankStart = omp_get_wtime();
            lowRankApproximation(G, rank, Q, Lambda);
            double timeLowRank = omp_get_wtime() - timeLowRankStart;

//            double timeLowRankGSStart = omp_get_wtime();
//            lowRankApproximation_GramSchmidt(G, rank, U, S);
//            double timeLowRankGS = omp_get_wtime() - timeLowRankGSStart;

            double timeProductStart = omp_get_wtime();
            MatrixXf GApprox = Q * Lambda * Q.transpose();
            double timeProduct = omp_get_wtime() - timeProductStart;

//            double timeProductGSStart = omp_get_wtime();
//            MatrixXf GApproxGS = U * S * U.transpose();
//            double timeProductGS = omp_get_wtime() - timeProductGSStart;

            double errorNorm = (G - GApprox).squaredNorm();
//            double errorNormH = (G - GApproxH).squaredNorm();
//            double errorNormGS = (G - GApproxGS).squaredNorm();

            results.open(filename, std::ios::out | std::ios::app);
//            results << rank << " " << beta << " "
//                    << timeLowRankGS << " " << timeProductGS << " " << errorNormGS << " "
////                    << timeLowRank << " " << timeProduct << " " << errorNorm << " "
//                    << timeLowRankGS << " " << errorNormGS
//                    << std::endl;
//            results.close();
            results << rank << " " << beta << " "
                    << timeLowRank << " " << timeProduct << " " << errorNorm << " "
                    //                    << timeLowRank << " " << timeProduct << " " << errorNorm << " "
                    << timeLowRank << " " << errorNorm
                    << std::endl;
            results.close();
        }
    } // For ranks/betas
}

TEST_CASE("cpd brainshift", "[fast][coherentpointdrift][visual][cpd]") {

    int patient = BRAIN_SHIFT_PATIENT;

    std::string fixedDataset;
    std::string movingDataset;
    std::vector<int> landmarksFixed;
    std::vector<int> landmarksMoving;
    switch (patient) {
        case 1:
//            fixedDataset = "BrainShift/1/MRA_TOF_FOV_cl1.vtk";
//            fixedDataset = "BrainShift/1/MRA_TOF_FOV_cl1.vtk";
            fixedDataset = "BrainShift/1/US1_cl1.vtk";
            movingDataset = "BrainShift/1/US2_cl1.vtk";
            break;
        case 2:
            fixedDataset = "BrainShift/2/MRA_TOF_FOV_cl1.vtk";
            movingDataset = "BrainShift/2/US1_cl1.vtk";
            break;
        case 3:
            fixedDataset = "BrainShift/3/MRA_TOF_FOV_cl1.vtk";
            movingDataset = "BrainShift/3/US1_cl1.vtk";
            break;
        default:
            exit(0);
    }

    // Load point clouds
    auto fixedPointMesh = getPointCloud(fixedDataset);
    auto movingPointMesh = getPointCloud(movingDataset);
    auto refPointMesh = getPointCloud(movingDataset);

    // Set registration settings
    float uniformWeight = UNIFORM_WEIGHT;
    float beta = BETA;
    float lambda = LAMBDA;
    unsigned int numRuns = NUM_RUNS;

    if (NORMALIZE_POINTS_BEFORE_REG) {
        normalizePointCloud(fixedPointMesh);
        normalizePointCloud(movingPointMesh);
        normalizePointCloud(refPointMesh);
    }

    // Find point set sizes
    unsigned int numFixedPoints =  fixedPointMesh.get()->getNrOfVertices();
    unsigned int numMovingPoints =  movingPointMesh.get()->getNrOfVertices();
    unsigned int numDimensions = getPointCloudFromMesh(fixedPointMesh).cols();

    // Set low rank approx rank
    auto rank = (unsigned int) ceil(sqrt(numMovingPoints));


    MatrixXf fixed = getPointCloudFromMesh(fixedPointMesh);
    MatrixXf moving = getPointCloudFromMesh(movingPointMesh);

    if (INCLUDE_LANDMARKS) {
        switch (patient) {
            case 1:
                landmarksFixed = {1150, 846, 1705, 1963};
                landmarksMoving = {838, 979, 433, 210};
                break;
            case 2:
                landmarksFixed = {2082, 2097, 1573, 2686, 2253};
                landmarksMoving = {784, 893, 1337, 259, 663};
                break;
            case 3:
                landmarksFixed = {601, 289, 186};
                landmarksMoving = {42, 145, 223};
                break;
            default:
                INCLUDE_LANDMARKS = false;
                std::cout << "No landmarks registered. Landmark error switched off.\n";
        }
    }

//    std::vector<float> lambdas = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0};
//    for (auto lambda : lambdas) {
//        std::cout << "Lambda: " << lambda << std::endl;

    // Prepare result text file
    std::string filename = "Brainshift-CPDNonRigid-Patient"
            + std::to_string(patient) + '-'
//            + std::to_string(lambda) + '-'
            + currentDateTime() + ".txt";
    std::ofstream results;
    results.open (filename, std::ios::out | std::ios::app);
    results << "CPD-NonRigid\n";
    results << "Fixed " << fixedDataset << " " << numFixedPoints << "x" << numDimensions << std::endl;
    results << "Moving " << movingDataset << " " << numMovingPoints << "x" << numDimensions << std::endl;
    results << "MaxIterations " << MAX_ITERATIONS << std::endl;
    results << "Tolerance " << TOLERANCE << std::endl;
    results << "No_transformation_applied" << std::endl;
    results << "uniformWeight rank beta lambda iterations errorPoints errorDeformed errorNonDeformed ";
    results << "timeTot timeAffinity timeLowRank timeEM timeE timeM timeEBuffers timeK1 timeK2 timeNp ";
    results << "timeMBuffers timeSolveW timeGW timeUpdate";
    if (INCLUDE_LANDMARKS) {
        results << " lmMeanDistPre lmMeanDistPost lmMaxDist\n";
    } else {
        results << std::endl;
    }
    results.close();


//    std::vector<float> betas = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0};
//    for (auto beta : betas) {
//        std::cout << "Beta " << beta << std::endl;

    unsigned int run = 0;
    while (run++ < numRuns) {

//        IterativeClosestPoint::pointer icp = IterativeClosestPoint::New();
//        icp->setMovingMesh(movingPointMesh);
//        icp->setFixedMesh(fixedPointMesh);

        // Run Coherent Point Drift
        auto cpd = CoherentPointDriftNonRigid::New();
        cpd->setBeta(beta);
        cpd->setLambda(lambda);
        cpd->setLowRankApproxRank(rank);
        cpd->setUniformWeight(uniformWeight);
        cpd->setResultTextFileName(filename);
        cpd->setTolerance(TOLERANCE);
        cpd->setMaximumIterations(MAX_ITERATIONS);
        cpd->setNormalization(NORMALIZE_POITNS_IN_CPD);
        cpd->setUpdateOutputMesh(VISUALIZE_REGISTRATION);
        if (INCLUDE_LANDMARKS) {
            cpd->setLandmarkIndices(landmarksFixed, landmarksMoving);
        }
        if (VISUALIZE_REGISTRATION) {

            cpd->setFixedMesh(fixedPointMesh);
            cpd->setMovingMesh(movingPointMesh);

            auto renderer = VertexRenderer::New();
            renderer->setDefaultColor(Color(0.0, 1.0, 0.0));
            renderer->setDefaultSize(1.5);
            renderer->addInputData(fixedPointMesh, Color::Red(), 2.0);                // Fixed points
            renderer->addInputData(movingPointMesh, Color::Black(), 2.0);             // Moving points before transf/deformation
            renderer->addInputConnection(cpd->getOutputPort(), Color::Blue(), 2.0);   // Moving points registered
//                renderer->addInputConnection(icp->getOutputPort(), Color::Cyan(), 2.0);   // Moving points registered

            auto window = SimpleWindow::New();
            window->addRenderer(renderer);
            window->start();
        } else {
            cpd->setFixedPoints(fixed);
            cpd->setMovingPoints(moving);
            cpd->execute();
        }
    }
//    } // End loop over betas
//    } // End loop over lambdas

}

TEST_CASE("icp brainshift", "[fast][coherentpointdrift][visual][cpd]") {

    int patient = BRAIN_SHIFT_PATIENT;

    std::string fixedDataset;
    std::string movingDataset;
    std::vector<int> landmarksFixed;
    std::vector<int> landmarksMoving;
    switch (patient) {
        case 1:
//            fixedDataset = "BrainShift/1/MRA_TOF_FOV_cl1.vtk";
//            fixedDataset = "BrainShift/1/MRA_TOF_FOV_cl1.vtk";
            fixedDataset = "BrainShift/1/US1_cl1.vtk";
            movingDataset = "BrainShift/1/US2_cl1.vtk";
            break;
        case 2:
            fixedDataset = "BrainShift/2/MRA_TOF_FOV_cl1.vtk";
            movingDataset = "BrainShift/2/US1_cl1.vtk";
            break;
        case 3:
            fixedDataset = "BrainShift/3/MRA_TOF_FOV_cl1.vtk";
            movingDataset = "BrainShift/3/US1_cl1.vtk";
            break;
        default:
            exit(0);
    }

    // Load point clouds
    auto fixedPointMesh = getPointCloud(fixedDataset);
    auto movingPointMesh = getPointCloud(movingDataset);
    auto refPointMesh = getPointCloud(movingDataset);

    // Set registration settings
    float uniformWeight = UNIFORM_WEIGHT;
    unsigned int numRuns = NUM_RUNS;

    MatrixXf fixed = getPointCloudFromMesh(fixedPointMesh);
    MatrixXf moving = getPointCloudFromMesh(movingPointMesh);

    // Find point set sizes
    unsigned int numFixedPoints =  fixed.rows();
    unsigned int numMovingPoints =  moving.rows();
    unsigned int numDimensions = moving.cols();

    if (INCLUDE_LANDMARKS) {
        switch (patient) {
            case 1:
                landmarksFixed = {1150, 846, 1705, 1963};
                landmarksMoving = {838, 979, 433, 210};
                break;
            case 2:
                landmarksFixed = {2082, 2097, 1573, 2686, 2253};
                landmarksMoving = {784, 893, 1337, 259, 663};
                break;
            case 3:
                landmarksFixed = {601, 289, 186};
                landmarksMoving = {42, 145, 223};
                break;
            default:
                INCLUDE_LANDMARKS = false;
                std::cout << "No landmarks registered. Landmark error switched off.\n";
        }
    }

    // Prepare result text file
    std::string filename = "Brainshift-ICP-Patient"
                           + std::to_string(patient) + '-'
                           + currentDateTime() + ".txt";
    std::ofstream results;
    results.open (filename, std::ios::out | std::ios::app);
    results << "ICP-Brainshift\n";
    results << "Fixed " << fixedDataset << " " << numFixedPoints << "x" << numDimensions << std::endl;
    results << "Moving " << movingDataset << " " << numMovingPoints << "x" << numDimensions << std::endl;
    results << "MaxIterations " << MAX_ITERATIONS << std::endl;
    results << "Tolerance default" << std::endl;
    results << "Point NOT normalizes before registration." << std::endl;
    results << "No_transformation_applied\n";
    results << "iterations timeTot LMMeanDistPrereg LMMeanDistICP\n";
    results.close();


    unsigned int run = 0;
    while (run++ < numRuns) {

        // Do ICP registration
        IterativeClosestPoint::pointer icp = IterativeClosestPoint::New();
        icp->setMovingMesh(movingPointMesh);
        icp->setFixedMesh(fixedPointMesh);
        icp->setMaximumNrOfIterations(MAX_ITERATIONS);
        icp->setFilename(filename);
        icp->setLandmarkIndices(landmarksFixed, landmarksMoving);
        if (VISUALIZE_REGISTRATION) {
            auto renderer = VertexRenderer::New();
            renderer->setDefaultColor(Color(0.0, 1.0, 0.0));
            renderer->setDefaultSize(1.5);
            renderer->addInputData(fixedPointMesh, Color::Red(), 2.0);                // Fixed points
            renderer->addInputData(movingPointMesh, Color::Green(), 1.5);                // Moving points before registration
            renderer->addInputConnection(icp->getOutputPort(), Color::Blue(), 2.0);
            auto window = SimpleWindow::New();
            window->addRenderer(renderer);
            window->start();
        } else {
            icp->update(0);
        }

    }
}

TEST_CASE("cpd rigid icp", "[fast][coherentpointdrift][visual][cpd]") {

    std::string fixedDataset = "Bunny/bunny.vtk";
    std::string movingDataset = "Bunny/bunny.vtk";
//    std::string fixedDataset = "BrainShift/1/MRA_TOF_FOV_cl1.vtk";
//    std::string movingDataset = "BrainShift/1/US1_cl1.vtk";

    int pointSize = POINT_SIZE;
    int numRuns = NUM_RUNS;
    float scale = 1.0f;
    float rotationDeg = 50.0f;
    Vector3f translation = Vector3f(0.0f, 0.0f, 0.0f);
    Vector3f rotationAxis = Vector3f(0.0f, 1.0f, 0.0f);

    // Load point clouds
    auto fixedPointMesh = getPointCloud(fixedDataset);
    auto movingPointMesh = getPointCloud(movingDataset);
    auto refPointMesh = getPointCloud(movingDataset);

    downsample(fixedPointMesh, pointSize);
    downsample(movingPointMesh, pointSize);

    // Transform one of the point clouds
    Affine3f affine = Affine3f::Identity();
    if (APPLY_TRANSFORMATION) {
        auto transform = AffineTransformation::New();
        affine.rotate(Eigen::AngleAxisf(3.141592f / 180.0f * rotationDeg, rotationAxis));
        affine.scale(scale);
        affine.translate(translation);
        transform->setTransform(affine);

        movingPointMesh->getSceneGraphNode()->setTransformation(transform);
        refPointMesh->getSceneGraphNode()->setTransformation(transform);
    }

    if (NORMALIZE_POINTS_BEFORE_REG) {
        normalizePointCloud(fixedPointMesh);
        normalizePointCloud(movingPointMesh);
        normalizePointCloud(refPointMesh);
    }

    int numFixedPoints = fixedPointMesh->getNrOfVertices();
    int numDimensions = 3;
    int numMovingPoints = movingPointMesh->getNrOfVertices();

    // Prepare result text file
    std::string filename = "ICP-Rigid-Bunny" + std::to_string(pointSize) + '-'
                            + currentDateTime() + ".txt";
    std::ofstream results;
    results.open (filename, std::ios::out | std::ios::app);
    results << "ICP-Rigid\n";
    results << "Fixed " << fixedDataset << " " << numFixedPoints << "x" << numDimensions << std::endl;
    results << "Moving " << movingDataset << " " << numMovingPoints << "x" << numDimensions << std::endl;
    results << "MaxIterations " << MAX_ITERATIONS << std::endl;
    results << "Tolerance default" << std::endl;
    if (NORMALIZE_POINTS_BEFORE_REG) {
        results << "Points normalized before registration." << std::endl;
    } else {
        results << "Point NOT normalizes before registration." << std::endl;
    }
    if (APPLY_TRANSFORMATION) {
        results << "Transform_applied rotationDeg " << rotationDeg
                << " rotationAxis (" << rotationAxis.x() << "," << rotationAxis.y() << "," << rotationAxis.z()
                << ") scale " << scale
                << " translation (" << translation.x() << "," << translation.y() << "," << translation.z() << ")\n";
        results << affine.affine() << std::endl;
    } else {
        results << "No_transformation_applied\n";
    }
    results << "iterations timeTot errorTransformation errorRotation \n";
    results.close();

    int run = 0;
    while(run++ < numRuns) {

        std::cout << "ICP Rigid: Iteration " << run << " of " << numRuns << std::endl;

        // Do ICP registration
        IterativeClosestPoint::pointer icp = IterativeClosestPoint::New();
        icp->setMovingMesh(movingPointMesh);
        icp->setFixedMesh(fixedPointMesh);
        icp->setMaximumNrOfIterations(MAX_ITERATIONS);

        if (VISUALIZE_REGISTRATION) {
            auto renderer = VertexRenderer::New();
            renderer->setDefaultColor(Color(0.0, 1.0, 0.0));
            renderer->setDefaultSize(1.5);
            renderer->addInputData(fixedPointMesh, Color::Red(), 2.0);                // Fixed points
            renderer->addInputData(movingPointMesh, Color::Green(), 1.5);                // Moving points before registration
            renderer->addInputConnection(icp->getOutputPort(), Color::Blue(), 2.0);
            auto window = SimpleWindow::New();
            window->addRenderer(renderer);
            window->start();
        } else {
            icp->update(0);
        }

        // Validate result
        Vector3f detectedTranslation = icp->getOutputTransformation()->getTransform().translation();
        auto detectedTransformation = icp->getOutputTransformation()->getTransform().matrix();
        auto detectedRotationMatrix = icp->getOutputTransformation()->getTransform().rotation();
    Vector3f detectedRotation = icp->getOutputTransformation()->getEulerAngles();
//    std::cout << "Rotation deg: " << ((180.0f/3.14159f) * detectedRotation).transpose() << std::endl;
//    std::cout << "Transformation: \n" << detectedTransformation << std::endl;
//    std::cout << "Rotation: \n" << detectedRotationMatrix << std::endl;

        double errorTransformation = (detectedTransformation.inverse() - affine.matrix()).squaredNorm();
        double errorRotation = (detectedRotationMatrix.inverse() - affine.rotation()).squaredNorm();

        std::ofstream resultsICP;
        resultsICP.open (filename, std::ios::out | std::ios::app);
        resultsICP  << icp->getIterations()-1  << " " << icp->getTime() << " "
                    << errorTransformation << " " << errorRotation
                    << std::endl;
        resultsICP.close();
    }
}

TEST_CASE("cpd rigid gadomski", "[fast][coherentpointdrift][visual][cpd]") {

////    std::string fixedDataset = "BrainShift/1/MRA_TOF_FOV_cl1.vtk";
////    std::string movingDataset = "BrainShift/1/US1_cl1.vtk";
//
//    std::string fixedDataset = "Bunny/bunny.vtk";
//    std::string movingDataset = "Bunny/bunny.vtk";
//
//    int pointSize = POINT_SIZE;
//    int numRuns = NUM_RUNS;
//    float outliers = 0.0;
//    float scale = 1.0f;
//    float rotationDeg = 50.0f;
//    Vector3f translation = Vector3f(0.0f, 0.0f, 0.0f);
//    Vector3f rotationAxis = Vector3f::UnitY();
//
//    // Load point clouds
//    auto fixedPointMesh = getPointCloud(fixedDataset);
//    downsample(fixedPointMesh, pointSize);
//
//    // Transform one of the point clouds
//    Affine3f affine = Affine3f::Identity();
//    if (APPLY_TRANSFORMATION) {
//        affine.rotate(Eigen::AngleAxisf(3.141592f / 180.0f * rotationDeg, rotationAxis));
//        affine.scale(scale);
//        affine.translate(translation);
//    }
//
//    MatrixXf fixed = getPointCloudFromMesh(fixedPointMesh);
//    MatrixXf moving = fixed.rowwise().homogeneous() * affine.affine().transpose();
//    int numFixedPoints = fixed.rows();
//    int numMovingPoints = moving.rows();
//    int numDimensions = moving.cols();
//
//    // Prepare result text file
//    std::string filename    = "Gadomski-Rigid-Bunny" + std::to_string(pointSize) + '-'
//                            + currentDateTime() + ".txt";
//    std::ofstream results;
//    results.open (filename, std::ios::out | std::ios::app);
//    results << "Gadomksi-Rigid\n";
//    results << "Fixed " << fixedDataset << " " << numFixedPoints << "x" << numDimensions << std::endl;
//    results << "Moving " << movingDataset << " " << numMovingPoints << "x" << numDimensions << std::endl;
//    results << "MaxIterations " << MAX_ITERATIONS << std::endl;
//    results << "Tolerance " << TOLERANCE << std::endl;
//    if (NORMALIZE_POINTS_BEFORE_REG) {
//        results << "Points normalized before registration." << std::endl;
//    } else if (NORMALIZE_POITNS_IN_CPD) {
//        results << "Points normalizes in CPD" << std::endl;
//    }
//    if (APPLY_TRANSFORMATION) {
//        results << "Transform_applied rotationDeg " << rotationDeg
//                << " rotationAxis (" << rotationAxis.x() << "," << rotationAxis.y() << "," << rotationAxis.z()
//                << ") scale " << scale
//                << " translation (" << translation.x() << "," << translation.y() << "," << translation.z() << ")\n";
//        results << affine.affine() << std::endl;
//    } else {
//        results << "No_transformation_applied\n";
//    }
//    results << "uniformWeight iterations errorPoints errorTransformation errorRot ";
//    results << "timeTot timeEM timeE timeM\n";
//    results.close();
//
//    int run = 0;
//    while (run++ < numRuns) {
//
//        std::cout << "CPD Gadomski: Iteration " << run << " of " << numRuns << std::endl;
//
//        // Run Coherent Point Drift - Gadomski
//        cpd::Rigid registration;
//        cpd::RigidResult result;
//        registration.normalize(NORMALIZE_POITNS_IN_CPD);
//        registration.outliers(outliers);
//        registration.tolerance(TOLERANCE);
//
//        result = registration.run(fixed, moving);
//
//        double errorPoints = (fixed - result.points).squaredNorm();
//        double errorTransformation = (affine.inverse().matrix() - result.matrix()).squaredNorm();
//        double errorRot = (affine.inverse().rotation() - result.rotation).squaredNorm();
//
//        AffineTransformation affT;
//        Affine3f aff = Affine3f::Identity();
//        aff = aff.linear() = result.rotation;
//        affT.setTransform(aff);
//        auto eulerAngles = affT.getEulerAngles();
//
//        std::ofstream resultsGadomski;
//        resultsGadomski.open (filename, std::ios::out | std::ios::app);
//        resultsGadomski << outliers << " " << result.iterations  << " "
//                        << errorPoints << " " << errorTransformation << " " << errorRot << " "
//                        << result.timeTotRes << " " << result.timeEMRes << " "
//                        << result.timeERes << " " << result.timeMRes
//                        << std::endl;
//        resultsGadomski.close();
//    }
}

TEST_CASE("cpd nonrigid gadomski", "[fast][coherentpointdrift][visual][cpd]") {
//
////    std::string fixedDataset = "BrainShift/1/MRA_TOF_FOV_cl1.vtk";
////    std::string movingDataset = "BrainShift/1/US1_cl1.vtk";
//    std::string fixedDataset = "Bunny/bunny.vtk";
//    std::string movingDataset = "Bunny/bunny.vtk";
//
//    // Set nonrigid registration parameters
//    double beta = BETA;
//    double lambda = LAMBDA;
//    bool applyLowRank = true;
//
//    // Set deformation settings
//    float deformationRadius = DEFORMATION_RADIUS;
//    Vector3f deformationCenter = Vector3f(0.0, 0.0, 0.0);
//
//    // Set parameters for noise and outliers
//    float noiseStdev = NOISE_STDEV;
//
//    // Set transformation parameters
//    int numRuns = NUM_RUNS;
//    float outliers = UNIFORM_WEIGHT;
//    float scale = 1.0f;
//    float rotationDeg = ROTATION_DEG;
//    Vector3f translation = Vector3f(0.0f, 0.0f, 0.0f);
//    Vector3f rotationAxis = Vector3f::UnitY();
//
//    std::vector<int> pointSetSizes = {12800};
//    for (auto size: pointSetSizes) {
//
//        std::cout << "Point set size: " << size << std::endl;
//
//    int pointSetSize = size;
//
//    // Load point meshes
//    auto fixedPointMesh = getPointCloud(fixedDataset);
//    auto movingPointMesh = getPointCloud(movingDataset);
//
//    downsample(fixedPointMesh, pointSetSize);
//    downsample(movingPointMesh, pointSetSize);
//
//    if (NORMALIZE_POINTS_BEFORE_REG) {
//        normalizePointCloud(fixedPointMesh);
//        normalizePointCloud(movingPointMesh);
//    }
//
//    // Deform point set
//    std::vector<int> deformedPointsIndices = {};
//    std::vector<int> deformedRefPointsIndices = {};
//    if (APPLY_DEFORMATION) {
//        deformPointCloud(movingPointMesh, deformationRadius, deformationCenter, deformedPointsIndices);
//    }
//
//    // Transform one of the point clouds
//    Affine3f affine = Affine3f::Identity();
//    if (APPLY_TRANSFORMATION) {
//        auto transform = AffineTransformation::New();
//        affine.rotate(Eigen::AngleAxisf(M_PI / 180.0f * rotationDeg, rotationAxis));
//        affine.scale(scale);
//        transform->setTransform(affine);
//        movingPointMesh->getSceneGraphNode()->setTransformation(transform);
//    }
//
//    // Add noise
//    if (ADD_NOISE) {
//        addGaussianNoise(movingPointMesh, noiseStdev);
//    }
//
//    if (NORMALIZE_POINTS_BEFORE_REG) {
//        normalizePointCloud(fixedPointMesh);
//        normalizePointCloud(movingPointMesh);
//    }
//
//    // Load point matrices from meshes
//    MatrixXf fixed = getPointCloudFromMesh(fixedPointMesh);
//    MatrixXf moving = getPointCloudFromMesh(movingPointMesh);
//
//    int numFixed = fixed.rows();
//    int numMoving = moving.rows();
//    int numDimensions = moving.cols();
//
//    // Set low-rank approx parameters
//    size_t numPoints = fixed.rows();
//    auto rank = (size_t) ceilf( sqrtf(numPoints) );
//
//    // Prepare result text file
//    std::string filename    = "Gadomski-Nonrigid-Compare-"
//                            + std::to_string(pointSetSize) + '-'
////                            + std::to_string(noiseStdev) + '-'
//                            + currentDateTime() + ".txt";
//    std::ofstream results;
//    results.open (filename, std::ios::out | std::ios::app);
//    results << "Gadomksi-Nonrigid-Bunny-\n";
//    results << "Fixed " << fixedDataset << " " << numFixed << "x" << numDimensions << std::endl;
//    results << "Moving " << movingDataset << " " << numMoving << "x" << numDimensions << std::endl;
//    results << "MaxIterations " << MAX_ITERATIONS << std::endl;
//    results << "Tolerance " << TOLERANCE << std::endl;
//    if (NORMALIZE_POINTS_BEFORE_REG) {
//        results << "Points normalized before registration." << std::endl;
//    } else if (NORMALIZE_POITNS_IN_CPD) {
//        results << "Points normalizes in CPD" << std::endl;
//    }
//    if (APPLY_TRANSFORMATION) {
//        results << "Transform_applied rotationDeg " << rotationDeg
//                << " rotationAxis (" << rotationAxis.x() << "," << rotationAxis.y() << "," << rotationAxis.z()
//                << ") scale " << scale
//                << " translation (" << translation.x() << "," << translation.y() << "," << translation.z() << ")\n";
//        results << affine.affine() << std::endl;
//    } else {
//        results << "No_transformation_applied\n";
//    }
//    results << "uniformWeight rank beta lambda iterations errorPoints errorDeformed errorNonDeformed errorMax ";
//    results << "timeNorm timeLowRank timeTot timeEM timeE timeM\n";
//    results.close();
//
//    std::vector<float> betas = {1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0};
////    for (auto beta : betas) {
//
////        std::cout << "Beta: " << beta << std::endl;
//
//        int run = 0;
//        while (run++ < numRuns) {
//
//            std::cout << "Run " << run << " of " << numRuns << std::endl;
//
//            // Run Coherent Point Drift - Gadomski
//            cpd::Nonrigid registration;
//            cpd::NonrigidResult result;
//            registration.normalize(NORMALIZE_POITNS_IN_CPD);
//            registration.outliers(outliers);
//            registration.beta(beta);
//            registration.lambda(lambda);
//            registration.applyLowRankApprox(applyLowRank);
//            registration.rank(rank);
//            registration.tolerance(TOLERANCE);
//            registration.max_iterations(MAX_ITERATIONS);
//
//            result = registration.run(fixed, moving);
//
//            float regErrorPoints = (fixed - result.points).squaredNorm();
//            float errorMax = 0.0;
//            float errorPoints = 0.0;
//            float errorDeformed = 0.0;
//            for (int i = 0; i < fixed.rows(); ++i) {
//                float errorNorm2 = (fixed.row(i) - result.points.row(i)).squaredNorm();
//                errorPoints += errorNorm2;
//                if (errorNorm2 > errorMax) {
//                    errorMax = errorNorm2;
//                }
//            }
//            for (int i : deformedPointsIndices) {
//                errorDeformed += (fixed.row(i) - result.points.row(i)).squaredNorm();
//            }
//            float numDeformedPoints = deformedPointsIndices.size();
//            float errorNonDeformed = (errorPoints - errorDeformed) / (numMoving - numDeformedPoints);
//            errorDeformed /= numDeformedPoints;
//            errorPoints /= numMoving;
//
//
//            std::ofstream resultsNRGadomski;
//            resultsNRGadomski.open(filename, std::ios::out | std::ios::app);
//            resultsNRGadomski << outliers << " " << rank << " " << beta << " " << lambda << " "
//                              << result.iterations << " "
//                              << errorPoints << " " << errorDeformed << " "
//                              << errorNonDeformed << " " << errorMax << " "
//                              << result.timeNormalizeRes << " " << result.timeLowRankApproxRes << " "
//                              << result.timeTotRes << " " << result.timeEMRes << " "
//                              << result.timeERes << " " << result.timeMRes
//                              << std::endl;
//            resultsNRGadomski.close();
//
//        } // loop numRuns
////    } // loop betas
//    } // loop point set sizes
}