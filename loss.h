#ifndef LOSS_H
#define LOSS_H

#include "utils.h"
#include "vecops.h"

adouble CosineLoss(const ACol& a, const Col&b);
double CosineLoss(const Col& a, const Col&b);

adouble NoiseMarginLoss(const ACol&, const Col&,
                        const vector<Col>&, const int&);
adouble LossNCE(const ACol&, const Col&, const unsigned&, const vector<Col>&,
                const ACol&, const int&, vector<double>&, AliasSampler&);
pair<adouble, adouble> NegLogProb(const ACol&, const Col&, const unsigned&,
                  const vector<Col>&, const ACol&);
#endif
