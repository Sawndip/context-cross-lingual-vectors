#include "loss.h"
#include "alias_sampler.h"

adouble CosineLoss(const ACol& a, const Col& b) {
  return 1 - CosineSim(a, b);
}

double CosineLoss(const Col& a, const Col& b) {
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

adouble NegSamplingLoss(const ACol& hidden, const unsigned& tgt_word,
                        const vector<Col>& tgt_vecs,
                        const ACol& tgt_bias, const int& k,
                        vector<double>& noise_dist, AliasSampler& sampler) {
  adouble lp = 0.0;
  adouble score = DotProdCol(hidden, tgt_vecs[tgt_word]) +
                  tgt_bias[tgt_word];
  adouble real_given_w = exp(score) / (1 + exp(score));
  lp += log(real_given_w);

  for (int i = 0; i < k; ++i) {
    unsigned r_word = sampler.Draw();
    adouble score = DotProdCol(hidden, tgt_vecs[r_word]) + tgt_bias[r_word];
    adouble noise_given_w = 1.0 / (1 + exp(score));
    lp += k * noise_dist[r_word] * log(noise_given_w);
  }
  return -1.0 * lp;
}

double rand01() {
  return (double) rand() / RAND_MAX;
}

adouble NCELoss(const ACol& hidden, const unsigned& tgt_word,
                const vector<Col>& tgt_vecs, const ACol& tgt_bias, const int& k,
                const vector<double>& noise_dist, AliasSampler& sampler) {
  double p_real = 1.0 / (1 + k), p_noise = k / (1.0 + k);
  adouble lp = 0.0;
  double word_in_noise = 1.0 / tgt_vecs.size();

  adouble score = DotProdCol(hidden, tgt_vecs[tgt_word]) + tgt_bias[tgt_word];
  adouble joint_real_w = p_real * exp(score);
  //double word_in_noise = noise_dist[tgt_word];
  adouble joint_noise_w = p_noise * word_in_noise;
  adouble real_given_w = joint_real_w / (joint_real_w + joint_noise_w);
  lp += log(real_given_w);

  for (int i = 0; i < k; ++i) {
    unsigned r_word = rand01() * tgt_vecs.size();
    //unsigned r_word = sampler.Draw();
    adouble score = DotProdCol(hidden, tgt_vecs[r_word]) + tgt_bias[r_word];
    adouble joint_real_w = p_real * exp(score);
    //double word_in_noise = noise_dist[r_word];
    adouble joint_noise_w = p_noise * word_in_noise;
    adouble noise_given_w = joint_noise_w / (joint_real_w + joint_noise_w);
    lp += log(noise_given_w);
  }
  return -1.0 * lp;
}

adouble LogProbLoss(const ACol& hidden, const unsigned& tgt_word,
                    const vector<Col>& tgt_vecs, const ACol& tgt_bias) {
  adouble denom = -999999999999;
  adouble score = DotProdCol(hidden, tgt_vecs[tgt_word]) + tgt_bias[tgt_word];
  /* Sum over the vocab here */
  for (int i = 0; i < tgt_vecs.size(); ++i)
    denom = LogAdd(denom, DotProdCol(hidden, tgt_vecs[i]) + tgt_bias[i]);
  return -1 * (score - denom);
}
