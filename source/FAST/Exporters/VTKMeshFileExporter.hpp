#pragma once

#include "FAST/Exporters/FileExporter.hpp"

namespace fast {

/**
 * @brief Write Mesh to file using the VTK polydata format
 *
 * <h3>Input ports</h3>
 * - 0: Mesh
 *
 * @sa VTKMeshFileImporter
 * @ingroup exporters
 */
class FAST_EXPORT VTKMeshFileExporter : public FileExporter {
    FAST_PROCESS_OBJECT(VTKMeshFileExporter);
    public:
        /**
         * @brief Create instance
         * @param filename File to export mesh data to
         * @param writeNormals Write normals if it exists in the input Mesh
         * @param writeColors  Write colors if it exists in the input Mesh
         * @param writeLabels Write labels if it exists in the input Mesh
         * @return instance
         */
        FAST_CONSTRUCTOR(VTKMeshFileExporter,
                         std::string, filename,,
                         bool, writeNormals, = true,
                         bool, writeColors, = true,
                         bool, writeLabels, = true
        )
        void setWriteNormals(bool writeNormals);
        void setWriteColors(bool writeColors);
        void setWriteLabels(bool writeLabels);
    private:
        VTKMeshFileExporter();
        void execute();

        bool m_writeNormals;
        bool m_writeColors;
        bool m_writeLabels;
};

}