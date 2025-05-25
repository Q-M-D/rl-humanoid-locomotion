//
// Created by han on 23-2-20.
//

#ifndef BAER_ETHERCAT_TRANSITIONDATA_H
#define BAER_ETHERCAT_TRANSITIONDATA_H

#include "cppTypes.h"

struct TransitionData {
    TransitionData() { zero(); }

    void zero() {
        // Flag to mark when transition is done
        done = false;

    }

    // Flag to mark when transition is done
    bool done = false;

};

#endif //BAER_ETHERCAT_TRANSITIONDATA_H
