#ifndef FAST_AFFINE_H
#define FAST_AFFINE_H


#include "FAST/AffineTransformation.hpp"
#include "FAST/ProcessObject.hpp"
#include "FAST/Data/Mesh.hpp"
#include "FAST/Algorithms/CoherentPointDrift/CoherentPointDrift.hpp"

namespace fast {

    class FAST_EXPORT CoherentPointDriftAffine: public CoherentPointDrift {
    FAST_OBJECT(CoherentPointDriftAffine);
    public:
        CoherentPointDriftAffine();
        void initializeVarianceAndMore() override;
        void maximization(MatrixXf& fixedPoints, MatrixXf& movingPoints) override;
        void saveResultsToTextFile() override;

    private:
        MatrixXf mAffineMatrix;                 // B
        MatrixXf mTranslation;                  // t
    };

}


#endif //FAST_AFFINE_H
