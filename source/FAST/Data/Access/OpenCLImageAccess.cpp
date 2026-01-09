#include "OpenCLImageAccess.hpp"
#include "FAST/Data/Image.hpp"

namespace fast {

cl::Image* OpenCLImageAccess::get() const {
    return mImage;
}

cl::Image2D* OpenCLImageAccess::get2DImage() const {
    return (cl::Image2D*)mImage;
}

cl::Image3D* OpenCLImageAccess::get3DImage() const {
    return (cl::Image3D*)mImage;
}


OpenCLImageAccess::OpenCLImageAccess(cl::Image3D* image, std::shared_ptr<Image> object) {
    // Copy the image
    mImage = new cl::Image3D(*image);
    mIsDeleted = false;
    mImageObject = object;
    m_dims = 3;
}

OpenCLImageAccess::OpenCLImageAccess(cl::Image2D* image, std::shared_ptr<Image> object) {
    // Copy the image
    mImage = new cl::Image2D(*image);
    mIsDeleted = false;
    mImageObject = object;
    m_dims = 2;
}

void OpenCLImageAccess::release() {
	mImageObject->accessFinished();
    if(!mIsDeleted) {
        delete mImage;
        mImage = nullptr;
        mIsDeleted = true;
    }
}

OpenCLImageAccess::~OpenCLImageAccess() {
    if(!mIsDeleted)
        release();
}

int OpenCLImageAccess::getDimensions() const {
    return m_dims;
}

} // end namespace fast
