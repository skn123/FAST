#ifndef FAST_RIGID_H
#define FAST_RIGID_H


#include "FAST/AffineTransformation.hpp"
#include "FAST/ProcessObject.hpp"
#include "FAST/Data/Mesh.hpp"
#include "FAST/Algorithms/CoherentPointDrift/CoherentPointDrift.hpp"

namespace fast {

    class FAST_EXPORT CoherentPointDriftRigid: public CoherentPointDrift {
    FAST_OBJECT(CoherentPointDriftRigid);
    public:
        CoherentPointDriftRigid();
        void initializeVarianceAndMore() override;
        void maximization(MatrixXf& fixedPoints, MatrixXf& movingPoints) override;
        void saveResultsToTextFile() override;

    private:
        MatrixXf mRotation;                     // R
        MatrixXf mTranslation;                  // t
    };

}


#endif //FAST_RIGID_H
