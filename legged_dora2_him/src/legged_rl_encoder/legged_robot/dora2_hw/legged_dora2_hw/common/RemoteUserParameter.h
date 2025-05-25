#ifndef BAER_ETHERCAT_REMOTEUSERPARAMETER_H
#define BAER_ETHERCAT_REMOTEUSERPARAMETER_H


#include "ControlParameters.h"


class RemoteUserParameter : public ControlParameters {
public:
    RemoteUserParameter():
            ControlParameters("dora2_user_parameters"),
            INIT_PARAMETER(MainThreadPeriod),
            INIT_PARAMETER(Kp),
            INIT_PARAMETER(Kd),
            INIT_PARAMETER(NonZeroAngle),
            INIT_PARAMETER(TorCmdValue),
            INIT_PARAMETER(InerpolateTime),
            INIT_PARAMETER(TorExecTime){}
   
    DECLARE_PARAMETER(double, MainThreadPeriod);
    DECLARE_PARAMETER(double, Kp);
    DECLARE_PARAMETER(double, Kd);
    DECLARE_PARAMETER(double, NonZeroAngle);
    DECLARE_PARAMETER(double, TorCmdValue);
    DECLARE_PARAMETER(double, InerpolateTime);
    DECLARE_PARAMETER(double, TorExecTime);
};

#endif //BAER_ETHERCAT_REMOTEUSERPARAMETER_H
