#pragma once

#include "DataTypes.hpp"
#include "Color.hpp"
#include <vector>

namespace fast {

/**
 * @brief Vertex of a Mesh
 * This class represents a vertex in a Mesh data object.
 * It has a 3D position, and may also have a normal, color, and label (integer).
 * @sa Mesh
 */
class FAST_EXPORT MeshVertex {
    public:
		MeshVertex(Vector3f position, Vector3f normal = Vector3f(0, 0, 0), Color color = Color::Null(), uchar label = 0);
		Vector3f getPosition() const;
		Vector3f getNormal() const;
		void setPosition(Vector3f position);
		void setNormal(Vector3f normal);
		void setLabel(uchar label);
		uchar getLabel() const;
		void setColor(Color color);
		Color getColor() const;
    private:
        Vector3f mPosition;
        Vector3f mNormal;
		Color mColor;
        uchar mLabel;
};

/**
 * @brief A connection between two or more vertices in a Mesh
 * @sa Mesh
 */
class FAST_EXPORT  MeshConnection {
	public:
        int getEndpoint(uint index);
		int getEndpoint1();
		int getEndpoint2();
        Color getColor();
		void setEndpoint(int endpointIndex, int vertexIndex);
		void setEndpoint1(uint index);
		void setEndpoint2(uint index);
		void setColor(Color color);
	protected:
        VectorXui mEndpoints;
		Color mColor;
		MeshConnection() {};
};

/**
 * @brief Line of a Mesh.
 * @sa MeshConnection
 * @sa Mesh
 */
class FAST_EXPORT  MeshLine : public MeshConnection {
	public:
		MeshLine(uint endpoint1, uint endpoint2, Color color = Color::Red());
};

/**
 * @brief Triangle of a Mesh.
 * @sa MeshConnection
 * @sa Mesh
 */
class FAST_EXPORT  MeshTriangle : public MeshConnection {
	public:
		MeshTriangle(uint endpoint1, uint endpoint2, uint endpoint3, Color color = Color::Red());
		int getEndpoint3();
		void setEndpoint3(uint index);
};

} // end namespace fast
