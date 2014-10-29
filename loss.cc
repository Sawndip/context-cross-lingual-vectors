#include "utils.h"
#include "vecops.h"

adouble CosineLoss(const ACol& a, const Col&b) {
  return 1 - CosineSim(a, b);
}

double CosineLoss(const Col& a, const Col&b) {
  return 1 - CosineSim(a, b);
}

adouble NoiseMarginLoss(const ACol& pred, const Col& gold,
                        const vector<Col>& vecs, const int& num_noise) {
  adouble error = 0, zero = 0;
  adouble gold_sim = CosineSim(pred, gold);
  for (int i = 0; i < num_noise; ++i) {
    int index = rand() % vecs.size();
    adouble diff = 1 - gold_sim + CosineSim(pred, vecs[index]);
    error += max(diff, zero);
  }
  return error;
}

