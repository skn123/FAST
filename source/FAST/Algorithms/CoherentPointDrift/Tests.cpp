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

void modifyPointCloud(Mesh::pointer &pointCloud,
        int numbers[3], float fractionOfPointsToKeep,
        float outlierLevel=0.0, float noiseLevel=0.0,
        float noiseVariance=0.1, float noiseMean=0.0) {

    MeshAccess::pointer accessFixedSet = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessFixedSet->getVertices();

    // Sample the preferred amount of points from the point cloud
    auto numVertices = (unsigned int) vertices.size();
    auto numSamplePoints = (unsigned int) ceilf(fractionOfPointsToKeep * numVertices);
    std::vector<MeshVertex> newVertices;

    std::unordered_set<int> movingIndices;
    unsigned int sampledPoints = 0;
    std::default_random_engine distributionEngine((unsigned long)omp_get_wtime());
    std::uniform_int_distribution<unsigned int> distribution(0, numVertices-1);
    while (sampledPoints < numSamplePoints) {
        unsigned int index = distribution(distributionEngine);
        if (movingIndices.count(index) < 1 && vertices.at(index).getPosition().array().isNaN().sum() == 0 ) {
            newVertices.push_back(vertices.at(index));
            movingIndices.insert(index);
            ++sampledPoints;
        }
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

    // Add random gaussian noise
    auto numNoisePoints = (unsigned int) ceilf(noiseLevel * numSamplePoints);

    std::normal_distribution<float> distributionNoiseX(noiseMean, noiseVariance);
    std::normal_distribution<float> distributionNoiseY(noiseMean, noiseVariance);
    std::normal_distribution<float> distributionNoiseZ(noiseMean, noiseVariance);

    for (int noiseAdded = 0; noiseAdded < numNoisePoints; noiseAdded++) {
        float noiseX = distributionNoiseX (distributionEngine);
        float noiseY = distributionNoiseY (distributionEngine);
        float noiseZ = distributionNoiseZ (distributionEngine);
        Vector3f noisePosition = Vector3f(noiseX, noiseY, noiseZ);
//        Vector3f noisePosition = Vector3f((maxX-minX)*noiseX, (maxY-minY)*noiseY, (maxZ-minZ)*noiseZ);
        MeshVertex noise = MeshVertex(noisePosition, Vector3f(1, 0, 0), Color::Black());
        newVertices.push_back(noise);
    }

    // Update point cloud to include the removed points and added noise
    pointCloud->create(newVertices);

    // Return number of points, noise and outliers
    numbers[0] = numSamplePoints;
    numbers[1] = numOutliers;
    numbers[2] = numNoisePoints;
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

void downsample(Mesh::pointer &pointCloud, unsigned int desiredNumberOfPoints) {

    MeshAccess::pointer accessFixedSet = pointCloud->getMeshAccess(ACCESS_READ);
    std::vector<MeshVertex> vertices = accessFixedSet->getVertices();

    // Sample the preferred amount of points from the point cloud
    auto numVertices = (unsigned int) vertices.size();
    auto numSamplePoints = min<unsigned int>(numVertices, desiredNumberOfPoints);
    std::vector<MeshVertex> newVertices;

    auto step = (int) floor((double) numVertices / (double)numSamplePoints);

    unsigned int sampledPoints = 0;
    for (unsigned long i = 0; i < numVertices; i += step) {
        if (vertices.at(i).getPosition().array().isNaN().sum() == 0 ) {
            newVertices.push_back(vertices.at(i));
            ++sampledPoints;
        }
    }
    pointCloud->create(newVertices);
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

    auto dataset1 = "GM_test_1.vtk";
    auto dataset2 = "GM_test_1.vtk";

    // Load point clouds
    auto cloud1 = getPointCloud(dataset1);
    auto cloud2 = getPointCloud(dataset2);
    auto cloud3 = getPointCloud(dataset2);

    // Normalize point clouds
//    normalizePointCloud(cloud1);
//    normalizePointCloud(cloud2);
//    normalizePointCloud(cloud3);

    // Modify point clouds

    int numbersCloud1[3];
    int numbersCloud2[3];
//    modifyPointCloud(cloud1, numbersCloud1, 0.01, 0.0, 0.0, 0.4);
//    modifyPointCloud(cloud2, numbersCloud2, 0.01, 0.0, 0.0);
//    modifyPointCloud(cloud3, numbersCloud2, 0.75, 0.0, 0.0, 0.0);

    downsample(cloud1, 3000);
    downsample(cloud2, 3000);
    keepPartOfPointCloud(cloud1, 0, 3000);
    keepPartOfPointCloud(cloud2, 1000, 2500);

    // Set registration settings
    float uniformWeight = 0.5;
    double tolerance = 1e-6;
    bool applyTransform = true;

    // Transform one of the point clouds
    Vector3f translation(-0.052f, 0.005f, -0.001f);
    auto transform = AffineTransformation::New();
    MatrixXf shearing = Matrix3f::Identity();
    shearing(0, 0) = 0.5;
    shearing(0, 1) = 1.2;
    shearing(1, 0) = 0.0;
    Affine3f affine = Affine3f::Identity();
    affine.rotate(Eigen::AngleAxisf(3.141592f / 180.0f * 50.0f, Eigen::Vector3f::UnitY()));
//    affine.scale(0.5);
//    affine.translate(translation);
//    affine.linear() += shearing;
    transform->setTransform(affine);

    if (applyTransform) {
        // Apply transform to one point cloud
        cloud2->getSceneGraphNode()->setTransformation(transform);
        // Apply transform to a point cloud not registered (for reference)
        cloud3->getSceneGraphNode()->setTransformation(transform);
    }

    // Run for different numbers of iterations
    std::vector<unsigned char> iterations = {50};
    for(auto maxIterations : iterations) {

        // Run Coherent Point Drift
        auto cpd = CoherentPointDriftRigid::New();
        cpd->setFixedMesh(cloud1);
        cpd->setMovingMesh(cloud2);
        cpd->setMaximumIterations(maxIterations);
        cpd->setTolerance(tolerance);
        cpd->setUniformWeight(uniformWeight);

        auto renderer = VertexRenderer::New();
        renderer->addInputData(cloud1, Color::Green(), 3.0);                        // Fixed points
//        renderer->addInputData(cloud3, Color::Blue(), 2.0);                         // Moving points
        renderer->addInputConnection(cpd->getOutputPort(), Color::Red(), 2.0);      // Moving points registered

        auto window = SimpleWindow::New();
        window->addRenderer(renderer);
        //window->setTimeout(1000);
        window->start();
    }

}