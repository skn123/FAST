#include <FAST/Exporters/VTKMeshFileExporter.hpp>
#include "FAST/Testing.hpp"
#include "FAST/Importers/VTKMeshFileImporter.hpp"
#include "FAST/Data/Mesh.hpp"
#include <FAST/Data/Color.hpp>

namespace fast {

TEST_CASE("No filename given to VTKMeshFileImporter throws exception", "[fast][VTKMeshFileImporter]") {
    auto importer = VTKMeshFileImporter::create("");
    CHECK_THROWS(importer->run());
}

TEST_CASE("Wrong filename VTKMeshFileImporter throws exception", "[fast][VTKMeshFileImporter]") {
    auto importer = VTKMeshFileImporter::create("asdasd");
    CHECK_THROWS(importer->run());
}

TEST_CASE("Importer VTK surface from file", "[fast][VTKMeshFileImporter]") {
    auto importer = VTKMeshFileImporter::create(Config::getTestDataPath() + "Surface_LV.vtk");
    auto surface = importer->runAndGetOutputData<Mesh>();

    // Check the surface
    CHECK(surface->getNrOfTriangles() == 768);
    CHECK(surface->getNrOfVertices() == 386);
}

TEST_CASE("Import VTK mesh with colors, normals and labels", "[fast][VTKMeshFileImporter]") {
    std::vector<MeshVertex> vertices = {
            MeshVertex(Vector3f(1, 1, 1), Vector3f(1, 0, 0), Color::Red(), 1),
            MeshVertex(Vector3f(1, 25, 1), Vector3f(0,1,0), Color::Green(), 3),
            MeshVertex(Vector3f(25, 20, 3), Vector3f(0,0,1), Color::Blue(), 0),
            MeshVertex(Vector3f(20, 1, 4), Vector3f(0.5, 0.5, 0.25).normalized(), Color(0.5, 0.5, 0.1), 2),
            };
    std::vector<MeshLine> lines = {
            MeshLine(0, 1),
            MeshLine(1, 2),
            MeshLine(2, 3),
            MeshLine(3, 0)
    };

    auto mesh = Mesh::create(vertices, lines);

    auto exporter = VTKMeshFileExporter::create("VTKMeshFileExporterImporterNormalColorLabelTest.vtk")->connect(mesh);
    CHECK_NOTHROW(exporter->run());

    auto importer = VTKMeshFileImporter::create("VTKMeshFileExporterImporterNormalColorLabelTest.vtk");
    auto importedMesh = importer->runAndGetOutputData<Mesh>();

    auto access = importedMesh->getMeshAccess(ACCESS_READ);
    auto importedVertices = access->getVertices();
    REQUIRE(vertices.size() == importedVertices.size());
    for(int i = 0; i < vertices.size(); ++i) {
        auto a = vertices[i];
        auto b = importedVertices[i];
        for(int j = 0; j < 3; ++j) {
            CHECK(a.getPosition()[j] == Approx(b.getPosition()[j]));
            CHECK(a.getNormal()[j] == Approx(b.getNormal()[j]));
            CHECK(a.getColor().asVector()[j] == Approx(b.getColor().asVector()[j]));
        }
        CHECK(a.getLabel() == b.getLabel());
    }

}

} // end namespace fast
