#ifndef LOSS_H
#define LOSS_H

#include "utils.h"
#include "vecops.h"

adouble CosineLoss(const ACol& a, const Col&b);
double CosineLoss(const Col& a, const Col&b);

adouble NoiseMarginLoss(const ACol&, const Col&,
                        const vector<Col>&, const int&);

#endif
