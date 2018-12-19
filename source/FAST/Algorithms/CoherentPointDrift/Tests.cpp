#include <FAST/Visualization/SimpleWindow.hpp>
#include <FAST/Importers/VTKMeshFileImporter.hpp>
#include <FAST/Importers/ImageFileImporter.hpp>
#include "FAST/Exporters/VTKMeshFileExporter.hpp"
#include <FAST/Visualization/ImageRenderer/ImageRenderer.hpp>
#include "FAST/Visualization/VertexRenderer/VertexRenderer.hpp"
#include "FAST/Visualization/TriangleRenderer/TriangleRenderer.hpp"
#include "FAST/Algorithms/SurfaceExtraction/SurfaceExtraction.hpp"
#include "FAST/Testing.hpp"
#include "CoherentPointDrift.hpp"
#include "Rigid.hpp"
#include "Affine.hpp"

#include <random>
#include <iostream>
using namespace fast;

Mesh::pointer getPointCloud(std::string filename=std::string("Surface_LV.vtk")) {
    auto importer = VTKMeshFileImporter::New();
//    importer->setFilename(Config::getTestDataPath() + "Surface_LV.vtk");
    importer->setFilename(Config::getTestDataPath() + filename);
    auto port = importer->getOutputPort();
    importer->update(0);
    return port->getNextFrame<Mesh>();
}

MatrixXf getPointCloudFromMesh(Mesh::pointer &pointCloud) {

    MeshAccess::pointer accessPointCloud = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessPointCloud->getVertices();

    unsigned int numDimensions = (unsigned int)vertices[0].getPosition().size();
    auto numPoints = (unsigned int)vertices.size();
    MatrixXf points = MatrixXf::Zero(numPoints, numDimensions);
    for(int i = 0; i < numPoints; ++i) {
        points.row(i) = vertices[i].getPosition();
    }
    return points;
}

void normalizePointCloud(Mesh::pointer &pointCloud) {

    MeshAccess::pointer accessCloud = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessCloud->getVertices();

    // Set dimensions of point sets
    unsigned int numDimensions = (unsigned int)vertices[0].getPosition().size();
    auto numPoints = (unsigned int)vertices.size();

    // Store point sets in matrices
    MatrixXf points = MatrixXf::Zero(numPoints, numDimensions);
    for(int i = 0; i < numPoints; ++i) {
        points.row(i) = vertices[i].getPosition();
    }

    // Center point clouds around origin, i.e. zero mean
    MatrixXf mean = points.colwise().sum() / numPoints;
    points -= mean.replicate(numPoints, 1);

    // Scale point clouds to have unit variance
    double scale = sqrt(points.cwiseProduct(points).sum() / (double)numPoints);
    points /= scale;

    // Create new vertices
    std::vector<MeshVertex> newVertices;
    for(int i = 0; i < numPoints; ++i) {
//        newVertices[i].setPosition(Vector3f(points.row(i)[0], points.row(i)[1], points.row(i)[2]));
        newVertices.push_back(Vector3f(points.row(i)));
    }

    pointCloud->create(newVertices);
}

void addGaussianNoise(Mesh::pointer &pointCloud, float variance) {
    MeshAccess::pointer accessFixedSet = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessFixedSet->getVertices();
    std::vector<MeshVertex> newVertices;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    auto numVertices = (unsigned int) vertices.size();
    std::default_random_engine distributionEngine(seed);

    // Add random gaussian noise to each point
    std::normal_distribution<float> distributionNoiseX(0, variance);
    std::normal_distribution<float> distributionNoiseY(0, variance);
    std::normal_distribution<float> distributionNoiseZ(0, variance);

    for (unsigned int i = 0; i < numVertices; i++) {
        float noiseX = distributionNoiseX (distributionEngine);
        float noiseY = distributionNoiseY (distributionEngine);
        float noiseZ = distributionNoiseZ (distributionEngine);
        Vector3f noisePosition = Vector3f(noiseX, noiseY, noiseZ);
        Vector3f position = vertices.at(i).getPosition();
        MeshVertex noise = MeshVertex(position+noisePosition, Vector3f(1, 0, 0), Color::Black());
        newVertices.push_back(noise);
    }

    pointCloud->create(newVertices);
}

void modifyPointCloud(Mesh::pointer &pointCloud, float fractionOfPointsToKeep,
        float outlierLevel=0.0) {

    MeshAccess::pointer accessFixedSet = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessFixedSet->getVertices();
    std::vector<MeshVertex> newVertices;

    auto numVertices = (unsigned int) vertices.size();
    auto numSamplePoints = numVertices;
    std::default_random_engine distributionEngine((unsigned long) omp_get_wtime());

    // Sample the preferred amount of points from the point cloud
    if (fractionOfPointsToKeep <= 1) {
        numSamplePoints = (unsigned int) ceilf(fractionOfPointsToKeep * numVertices);
        std::unordered_set<int> movingIndices;
        unsigned int sampledPoints = 0;
        std::uniform_int_distribution<unsigned int> distribution(0, numVertices - 1);
        while (sampledPoints < numSamplePoints) {
            unsigned int index = distribution(distributionEngine);
            if (movingIndices.count(index) < 1 && vertices.at(index).getPosition().array().isNaN().sum() == 0) {
                newVertices.push_back(vertices.at(index));
                movingIndices.insert(index);
                ++sampledPoints;
            }
        }
    } else {
        newVertices = vertices;
    }

    // Add uniformly distributed outliers
    auto numOutliers = (unsigned int) ceilf(outlierLevel * numSamplePoints);
    float minX, minY, minZ;
    Vector3f position0 = vertices[0].getPosition();
    minX = position0[0];
    minY = position0[1];
    minZ = position0[2];
    float maxX = minX, maxY = minY, maxZ = minZ;
    for (auto &vertex : vertices) {
        Vector3f position = vertex.getPosition();
        if (position[0] < minX) {minX = position[0]; }
        if (position[0] > maxX) {maxX = position[0]; }
        if (position[1] < minY) {minY = position[1]; }
        if (position[1] > maxY) {maxY = position[1]; }
        if (position[2] < minZ) {minZ = position[2]; }
        if (position[2] > maxZ) {maxZ = position[2]; }
    }

    std::uniform_real_distribution<float> distributionOutliersX(minX, maxX);
    std::uniform_real_distribution<float> distributionOutliersY(minY, maxY);
    std::uniform_real_distribution<float> distributionOutliersZ(minZ, maxZ);

    for (int outliersAdded = 0; outliersAdded < numOutliers; outliersAdded++) {
        float outlierX = distributionOutliersX (distributionEngine);
        float outlierY = distributionOutliersY (distributionEngine);
        float outlierZ = distributionOutliersZ (distributionEngine);
        Vector3f outlierPosition = Vector3f(outlierX, outlierY, outlierZ);
        MeshVertex outlier = MeshVertex(outlierPosition, Vector3f(1, 0, 0), Color::Black());
        newVertices.push_back(outlier);
    }

    // Update point cloud to include the removed points and added noise
    pointCloud->create(newVertices);
}


void keepPartOfPointCloud(Mesh::pointer &pointCloud, unsigned int startPoint, unsigned int endPoint) {
    MeshAccess::pointer accessFixedSet = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessFixedSet->getVertices();

    // Sample the preferred amount of points from the point cloud
    auto numVertices = (unsigned int) vertices.size();
    endPoint = min<unsigned int>(endPoint, numVertices);
    std::vector<MeshVertex> newVertices;

    assert(endPoint > startPoint);
    for (unsigned long index = startPoint; index < endPoint; index++) {
        if (vertices.at(index).getPosition().array().isNaN().sum() == 0) {
            newVertices.push_back(vertices.at(index));
        }
    }
    pointCloud->create(newVertices);
}

void filterPointCloud(Mesh::pointer &pointCloud, float depthFar, float depthNear, float angle, float shift=0.0f) {
    MeshAccess::pointer accessFixedSet = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessFixedSet->getVertices();

    auto numVertices = (unsigned int) vertices.size();
    std::vector<MeshVertex> newVertices;

    for (unsigned long index = 0; index < numVertices; index++) {
        Vector3f position = vertices.at(index).getPosition();
        if (position[2] <= depthFar && position[2] >= depthNear
               && std::fabs(std::atan((position[0]+shift)/position[2])) <= angle) {
            newVertices.push_back(vertices.at(index));
        }
    }
    pointCloud->create(newVertices);
}

void flip(Mesh::pointer &pointCloud) {
    MeshAccess::pointer accessFixedSet = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessFixedSet->getVertices();

    auto numVertices = (unsigned int) vertices.size();
    std::vector<MeshVertex> newVertices;
    newVertices = vertices;

    for (unsigned long index = 0; index < numVertices; index++) {
        Vector3f position = vertices.at(index).getPosition();
        Vector3f newPosition = position;
        newPosition[1] *= -1;
        newPosition[2] *= -1;
        newVertices.at(index).setPosition(newPosition);
    }
    pointCloud->create(newVertices);
}


void downsample(Mesh::pointer &pointCloud, unsigned int desiredNumberOfPoints) {

    MeshAccess::pointer accessFixedSet = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessFixedSet->getVertices();

    // Sample the preferred amount of points from the point cloud
    auto numVertices = (unsigned int) vertices.size();
    auto numSamplePoints = min<unsigned int>(numVertices, desiredNumberOfPoints);
    std::vector<MeshVertex> newVertices;

    auto step = (float) numVertices / (float)numSamplePoints;
    unsigned int sampledPoints = 0;
    for (unsigned long i = 0; i < desiredNumberOfPoints; i++) {
        auto newIndex = (unsigned long) floorf((float)i*step);
        if (vertices.at(newIndex).getPosition().array().isNaN().sum() == 0 ) {
            newVertices.push_back(vertices.at(newIndex));
        }
        ++sampledPoints;
    }
    pointCloud->create(newVertices);
}

void savePointCloud(Mesh::pointer &pointCloud, std::string filename) {

    auto exporter = VTKMeshFileExporter::New();
    exporter->setFilename(Config::getTestDataPath() + filename + ".vtk");
    exporter->setInputData(pointCloud);
    exporter->update(0);
}

void saveAbdominalSurfaceExtraction(int threshold=-500) {
    // Import CT image
    ImageFileImporter::pointer importer = ImageFileImporter::New();
    importer->setFilename(Config::getTestDataPath() + "CT/CT-Abdomen.mhd");

    // Extract surface mesh using a threshold value
    SurfaceExtraction::pointer extraction = SurfaceExtraction::New();
    extraction->setInputConnection(importer->getOutputPort());
    extraction->setThreshold(threshold);

    auto exporter = VTKMeshFileExporter::New();
    exporter->setFilename(Config::getTestDataPath() + "AbdominalModel.vtk");
    exporter->setInputConnection(extraction->getOutputPort());
    exporter->update(0);
}

void visualizeSurfaceExtraction(int threshold=-500) {

    // Import CT image
    ImageFileImporter::pointer importer = ImageFileImporter::New();
    importer->setFilename(Config::getTestDataPath() + "CT/CT-Abdomen.mhd");

    // Extract surface mesh using a threshold value
    SurfaceExtraction::pointer extraction = SurfaceExtraction::New();
    extraction->setInputConnection(importer->getOutputPort());
    extraction->setThreshold(threshold);

    TriangleRenderer::pointer surfaceRenderer = TriangleRenderer::New();
    surfaceRenderer->setInputConnection(extraction->getOutputPort());
    SimpleWindow::pointer window = SimpleWindow::New();
    window->addRenderer(surfaceRenderer);
    window->start();
}


TEST_CASE("cpd", "[fast][coherentpointdrift][visual][cpd]") {

    auto dataset1 = "bunny.vtk";
    auto dataset2 = "bunny.vtk";

    // Load point clouds
    auto cloud1 = getPointCloud(dataset1);
    auto cloud2 = getPointCloud(dataset2);
    auto cloud3 = getPointCloud(dataset2);

    // Normalize point clouds
    normalizePointCloud(cloud1);
    normalizePointCloud(cloud2);
    normalizePointCloud(cloud3);

    // Modify point clouds

    downsample(cloud1, 12000);
    downsample(cloud2, 12000);
//    downsample(cloud3, 12000);
//    addGaussianNoise(cloud1, 0.12f*0.12f);
//    addGaussianNoise(cloud2, powf(0.2f,2));

//    modifyPointCloud(cloud1, 1.0, 0.3);
//    modifyPointCloud(cloud2, 1.0, 0.3);
//    modifyPointCloud(cloud3, 1.0, 0.3);

//    keepPartOfPointCloud(cloud1, 0, 3000);
//    keepPartOfPointCloud(cloud2, 1000, 2500);

    // Set registration settings
    float uniformWeight = 0.0;
    double tolerance = 1e-8;
    bool applyTransform = true;

    // Transform one of the point clouds
    Vector3f translation(-0.052f, 0.005f, -0.001f);
    auto transform = AffineTransformation::New();
    MatrixXf shearing = Matrix3f::Identity();
    shearing(0, 0) = 0.3;
    shearing(0, 1) = 1.2;
    shearing(1, 0) = 0.0;
    Affine3f affine = Affine3f::Identity();
    affine.rotate(Eigen::AngleAxisf(3.141592f / 180.0f * 50.0f, Eigen::Vector3f::UnitZ()));
    affine.scale(0.5);
    affine.translate(translation);
//    affine.linear() += shearing;
    transform->setTransform(affine);

    if (applyTransform) {
        cloud2->getSceneGraphNode()->setTransformation(transform);
        cloud3->getSceneGraphNode()->setTransformation(transform);
    }

    // Run for different numbers of iterations
    MatrixXf fixedPoints = getPointCloudFromMesh(cloud1);
    std::vector<unsigned int> iterations = {100};
    for(auto maxIterations : iterations) {

        // Run Coherent Point Drift
        auto cpd = CoherentPointDriftRigid::New();
        cpd->setFixedMesh(cloud1);
        cpd->setMovingMesh(cloud2);
        cpd->setMaximumIterations(maxIterations);
        cpd->setTolerance(tolerance);
        cpd->setUniformWeight(uniformWeight);

        auto renderer = VertexRenderer::New();
        renderer->addInputData(cloud1, Color::Red(), 1.5);                        // Fixed points
//        renderer->addInputData(cloud2, Color::Blue(), 2.0);                        // Fixed points
//        renderer->addInputData(cloud3, Color::Black(), 2.0);                         // Moving points
        renderer->addInputConnection(cpd->getOutputPort(), Color::Blue(), 2.0);      // Moving points registered

        auto window = SimpleWindow::New();
        window->addRenderer(renderer);
//        window->setTimeout(1000);
        window->start();

//        std::cout << "Error: " << cpd->mResults[0] << std::endl;
//        std::cout << "Iterations: " << cpd->mResults[1] << std::endl;
//        std::cout << "Time EM: " << cpd->mResults[2] << std::endl;
    }

}


//TEST_CASE("cpd", "[fast][coherentpointdrift][visual][cpd]") {
//
//    auto dataset1 = "bunny1890.vtk";
//    auto dataset2 = "bunny1890.vtk";
//
//    // Set registration settings
//    float uniformWeight = 0.7;
//    double tolerance = 1e-8;
//    unsigned int maxIterations = 300;
//    bool applyTransform = true;
//
//    // Load point clouds
//    auto cloud1 = getPointCloud(dataset1);
//    auto cloud2 = getPointCloud(dataset2);
//    auto cloud3 = getPointCloud(dataset2);
//
//    // Normalize point clouds
//    normalizePointCloud(cloud1);
//    normalizePointCloud(cloud2);
//    normalizePointCloud(cloud3);
//
//    // Transform one of the point clouds
//    Vector3f translation(-0.052f, 0.005f, -0.001f);
//    auto transform = AffineTransformation::New();
//    MatrixXf shearing = Matrix3f::Identity();
//    shearing(0, 0) = 0.5;
//    shearing(0, 1) = 1.2;
//    shearing(1, 0) = 0.0;
//    Affine3f affine = Affine3f::Identity();
//    affine.rotate(Eigen::AngleAxisf(3.141592f / 180.0f * 50.0f, Eigen::Vector3f::UnitZ()));
////    affine.scale(0.5);
////    affine.translate(translation);
////    affine.linear() += shearing;
//    transform->setTransform(affine);
//
//    clock_t seed = clock();
//    int numRuns = 23;
//    double errorRotation[numRuns];
//    double numIterations[numRuns];
//    double times[numRuns];
//
//    float gaussianSTD = 0.12f;
//
//    for (int i = 0; i < numRuns; i++) {
//
//        // Modify point clouds
//        addGaussianNoise(cloud2, gaussianSTD * gaussianSTD);
//
//        if (applyTransform) {
//            cloud2->getSceneGraphNode()->setTransformation(transform);
//            cloud3->getSceneGraphNode()->setTransformation(transform);
//        }
//
//        // Run Coherent Point Drift
//        auto cpd = CoherentPointDriftRigid::New();
//        cpd->setFixedMesh(cloud1);
//        cpd->setMovingMesh(cloud2);
//        cpd->setMaximumIterations(maxIterations);
//        cpd->setTolerance(tolerance);
//        cpd->setUniformWeight(uniformWeight);
//
//        auto renderer = VertexRenderer::New();
//        renderer->addInputData(cloud1, Color::Green(), 3.0);                        // Fixed points
////        renderer->addInputData(cloud3, Color::Blue(), 2.0);                         // Moving points
//        renderer->addInputConnection(cpd->getOutputPort(), Color::Red(), 2.0);      // Moving points registered
//
//        auto window = SimpleWindow::New();
//        window->addRenderer(renderer);
//        window->setTimeout(1);
//        window->start();
//
//        errorRotation[i] = cpd->mResults[0];
//        numIterations[i] = cpd->mResults[1];
//        times[i] = cpd->mResults[2];
//    }
//
//    std::cout << "ERROR\n";
//    for (int i = 0; i < numRuns; i++) {
//        std::cout << errorRotation[i] << std::endl;
//    }
//
//    std::cout << "Iterations\n";
//    for (int i = 0; i < numRuns; i++) {
//        std::cout << numIterations[i] << std::endl;
//    }
//
//    std::cout << "TimeEM\n";
//    for (int i = 0; i < numRuns; i++) {
//        std::cout << times[i] << std::endl;
//    }
//}