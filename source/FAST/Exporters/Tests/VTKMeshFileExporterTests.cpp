#include "FAST/Testing.hpp"
#include "FAST/Exporters/VTKMeshFileExporter.hpp"
#include "FAST/Data/Mesh.hpp"

using namespace fast;

TEST_CASE("No filename given to VTKMeshFileExporter", "[fast][VTKMeshFileExporter]") {
	auto mesh = Mesh::create({});
	auto exporter = VTKMeshFileExporter::create("")->connect(mesh);
	CHECK_THROWS(exporter->run());
}

TEST_CASE("Export mesh with lines", "[fast][VTKMeshFileExporter]") {
    std::vector<MeshVertex> vertices = {
            MeshVertex(Vector3f(1, 1, 0)),
            MeshVertex(Vector3f(1, 25, 0)),
            MeshVertex(Vector3f(25, 20, 0)),
            MeshVertex(Vector3f(20, 1, 0)),
    };
    std::vector<MeshLine> lines = {
            MeshLine(0, 1),
            MeshLine(1, 2),
            MeshLine(2, 3),
            MeshLine(3, 0)
    };

    auto mesh = Mesh::create(vertices, lines);

    auto exporter = VTKMeshFileExporter::create("VTKMeshFileExporter2DTest.vtk")->connect(mesh);
	CHECK_NOTHROW(exporter->run());
}

TEST_CASE("Export mesh with triangles", "[fast][VTKMeshFileExporter]") {
    std::vector<MeshVertex> vertices = {
            MeshVertex(Vector3f(1, 1, 1)),
            MeshVertex(Vector3f(1, 1, 10)),
            MeshVertex(Vector3f(1, 10, 10)),

            MeshVertex(Vector3f(1, 1, 1)),
            MeshVertex(Vector3f(1, 1, 10)),
            MeshVertex(Vector3f(30, 15, 15)),

            MeshVertex(Vector3f(1, 1, 10)),
            MeshVertex(Vector3f(1, 10, 10)),
            MeshVertex(Vector3f(30, 15, 15)),

            MeshVertex(Vector3f(1, 1, 1)),
            MeshVertex(Vector3f(1, 10, 10)),
            MeshVertex(Vector3f(30, 15, 15))
    };
    std::vector<MeshTriangle> triangles = {
            MeshTriangle(0, 1, 2),
            MeshTriangle(3, 4, 5),
            MeshTriangle(6, 7, 8),
            MeshTriangle(9, 10, 11)
    };

    auto mesh = Mesh::create(vertices, {}, triangles);

    auto exporter = VTKMeshFileExporter::create("VTKMeshFileExporter3DTest.vtk")->connect(mesh);
	CHECK_NOTHROW(exporter->run());
}

TEST_CASE("Export mesh with colors, normals and labels", "[fast][VTKMeshFileExporter]") {
    std::vector<MeshVertex> vertices = {
            MeshVertex(Vector3f(1, 1, 1), Vector3f(1, 0, 0), Color::Red(), 1),
            MeshVertex(Vector3f(1, 25, 1), Vector3f(0,1,0), Color::Green(), 3),
            MeshVertex(Vector3f(25, 20, 3), Vector3f(0,0,1), Color::Blue(), 0),
            MeshVertex(Vector3f(20, 1, 4), Vector3f(0.333, 0.333, 0.333), Color(0.5, 0.5, 0.1), 2),
    };
    std::vector<MeshLine> lines = {
            MeshLine(0, 1),
            MeshLine(1, 2),
            MeshLine(2, 3),
            MeshLine(3, 0)
    };

    auto mesh = Mesh::create(vertices, lines);

    auto exporter = VTKMeshFileExporter::create("VTKMeshFileExporter2DNormalColorLabelTest.vtk", true, true, true)->connect(mesh);
    CHECK_NOTHROW(exporter->run());
}

