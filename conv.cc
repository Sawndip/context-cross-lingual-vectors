/* 
  This program learns to compute the embedding of a word in context
  by using convolutional neural network. The word-in-context embedding 
  is them contrasted against the embedding of the word aligned in a
  different language from which the error is computed.

  The gradient of the error function is calculated using automatic
  differentiation from library adept.h
*/
#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <string>
#include <tr1/unordered_map>
#include <Eigen/Core>
#include <random>
#include <adept.h>

#include "utils.h"
#include "vecops.h"

#define RHO 0.95
#define EPSILON 0.000001

using namespace std;
using namespace Eigen;
using adept::adouble;

/* General parameters of the model */
template <typename AT, typename T>
class Param {

 public:
  AT var;

  void Init(const int& rows, const int& cols) {
    var = (0.6 / sqrt (rows * cols)) * AT::Random(rows, cols);
    del_var = T::Zero(rows, cols);
    del_grad_var = T::Zero(rows, cols);
  }

  void AdadeltaUpdate(const double& rho, const double& epsilon) {
    for (unsigned i = 0; i < var.rows(); ++i) {
      for (unsigned j = 0; j < var.cols(); ++j) {
        double g = var(i, j).get_gradient();
        double accum_g = rho * del_grad_var(i, j) + (1 - rho) * g * g;
        double del = g * sqrt(del_var(i, j) + epsilon) / sqrt(accum_g + epsilon);
        var(i, j) -= del;  // Update the variable
        /* Update memory */
        del_grad_var(i, j) = accum_g;
        del_var(i, j) = rho * del_var(i, j) + (1 - rho) * del * del;
      }
    }
  }

 private:
  T del_var, del_grad_var;  // Adadelta memory
};

/* Filter parameters used in the convolution model */
class Filter {

 public:
  Param<AMat, Mat> filter;
  Param<ACol, Col> bias;

  void Init(const int& rows, const int& cols) {
    filter.Init(rows, cols);
    bias.Init(rows, 1);
  }

  void AdadeltaUpdate(const double& rho, const double& epsilon) {
    filter.AdadeltaUpdate(rho, epsilon);
    bias.AdadeltaUpdate(rho, epsilon);
  }
};

/* Main class definition that learns the word vectors */
class Model {

 public:
  /* The parameters of the model */
  Filter f11, f12, f21, f22;  // Convolution
  Param<AMat, Mat> p1, p2, p3;  // Post-convolution
  Param<ACol, Col> p2_b, p3_b;  // Post-convolution
  /* Word vectors */
  unsigned window_size, src_len, tgt_len, hidden_len;
  unsigned filter_len1, filter_len2, kbest1, kbest2;
  mapStrUnsigned src_vocab, tgt_vocab;
  vector<Col> src_word_vecs, tgt_word_vecs;
      
  Model(const int& filt_len, const int& kmax, const string& src_vec_file,
        const string& tgt_vec_file) {
    ReadVecsFromFile(src_vec_file, &src_vocab, &src_word_vecs);
    ReadVecsFromFile(tgt_vec_file, &tgt_vocab, &tgt_word_vecs);
    src_len = src_word_vecs[0].size();
    tgt_len = tgt_word_vecs[0].size();
    filter_len1 = filt_len, filter_len2 = filt_len - 1;
    kbest1 = kmax, kbest2 = kmax - 2;
    if (filter_len2 <= 1 || kbest2 < 1) {
      cerr << "Minimum filter len: " << 3 << endl;
      cerr << "Minimum max len: " << 3 << endl;
      exit(0);
    }
    /* Params initialization */
    f11.Init(src_len, filter_len1);
    f12.Init(src_len, filter_len1);
    f21.Init(src_len, filter_len2);
    f22.Init(src_len, filter_len2);

    p1.Init(kbest2, 1);
    p2.Init(src_len, src_len);
    p2_b.Init(src_len, 1);
    p3.Init(tgt_len, src_len);
    p3_b.Init(tgt_len, 1);
  }

  void Convolve(const Mat& sent_mat, AMat* res) {
    AMat out11, out12, out21, out22, layer1_out;
    /* Layer 1 */
    ConvolveLayer(sent_mat, f11, kbest1, &out11);
    ConvolveLayer(sent_mat, f12, kbest1, &out12);
    /* Layer 2 */
    layer1_out = out11 + out12;
    ConvolveLayer(layer1_out, f21, kbest2, &out21);
    ConvolveLayer(layer1_out, f22, kbest2, &out22);
    /* Add the results of the two parallel convolutions */
    *res = out21 + out22;  
  }

  template <typename T>
  void ConvolveLayer(const T& mat, const Filter& filter,
                     const int& kmax, AMat* res) {
    AMat convolved;
    convolve_wide(mat, filter.filter.var, &convolved);
    Max(convolved, kmax, res);
    *res += filter.bias.var.rowwise().replicate(res->cols());
    ElemwiseSigmoid(res);
  }

  adouble PredError(const Col& src_vec, const Col& tgt_vec,
                    const ACol& sent_vec) {
    /* Pass the src_word_vec through non-linearity */
    ACol src_non_linear_vec = Prod(p2.var, src_vec) + p2_b.var;
    ElemwiseSigmoid(&src_non_linear_vec);
    /* Add the processed src word vec with context_vec & convert to tgt_len */
    ACol pred_tgt_vec = p3.var * (sent_vec + src_non_linear_vec) + p3_b.var;
    return 1 - CosineSim(pred_tgt_vec, tgt_vec);
  }

  void UpdateParams() {
    f11.AdadeltaUpdate(RHO, EPSILON);
    f12.AdadeltaUpdate(RHO, EPSILON);
    f21.AdadeltaUpdate(RHO, EPSILON);
    f22.AdadeltaUpdate(RHO, EPSILON);
    p1.AdadeltaUpdate(RHO, EPSILON);
    p2.AdadeltaUpdate(RHO, EPSILON);
    p2_b.AdadeltaUpdate(RHO, EPSILON);
    p3.AdadeltaUpdate(RHO, EPSILON);
    p3_b.AdadeltaUpdate(RHO, EPSILON);
  }

};

void Train(const string& p_corpus, const string& a_corpus, const int& num_iter,
           const int& update_every, Model* model, adept::Stack* s) {
  for (unsigned i=0; i<num_iter; ++i) {
    cerr << "\nIteration: " << i+1 << endl;
    ifstream p_file(p_corpus.c_str()), a_file(a_corpus.c_str());
    string p_line, a_line;
    vector<int> src_words, tgt_words;
    unsigned num_words = 0, erroneous_cases = 0;
    adouble total_error = 0, semi_error = 0;
    int accum = 0;
    s->new_recording();
    if (p_file.is_open() && a_file.is_open()) {
      while (getline(p_file, p_line) && getline(a_file, a_line)) {
        /* Extracting words from sentences */
        vector<string> src_tgt = split_line(p_line, '\t');
        vector<string> src = split_line(src_tgt[0], ' ');
        vector<string> tgt = split_line(src_tgt[1], ' ');
        src_words.clear();
        tgt_words.clear();
        /* Read and Clean Source sentence */
        mapUnsUns old_to_new;
        unsigned index = 0;
        for (unsigned j=0; j<src.size(); ++j) {
          if (model->src_vocab.find(src[j]) != model->src_vocab.end() &&
              ConsiderForContext(src[j])) {
            src_words.push_back(model->src_vocab[src[j]]);
            old_to_new[j] = index++;
          }
        }
        /* Make a sentence matrix */
        Mat src_sent_mat(model->src_len, src_words.size());
        for (unsigned i = 0; i < src_words.size(); ++i)
          src_sent_mat.col(i) = model->src_word_vecs[src_words[i]];
        /* Target sentence */
        for (unsigned j=0; j<tgt.size(); ++j) {
          if (model->tgt_vocab.find(tgt[j]) != model->tgt_vocab.end())
            tgt_words.push_back(model->tgt_vocab[tgt[j]]);
          else
            tgt_words.push_back(-1); // word not in vocab
        }
        /* Read the alignment line */
        AMat sent_convolved;
        ACol sent_vec;
        bool convolved = false;
        vector<string> src_tgt_pairs = split_line(a_line, ' ');
        for (unsigned j = 0; j < src_tgt_pairs.size(); ++j) {
          vector<string> index_pair = split_line(src_tgt_pairs[j], '-');
          unsigned src_ix = stoi(index_pair[0]), tgt_ix = stoi(index_pair[1]);
          unsigned tgt_word = tgt_words[tgt_ix];
          /* If both words are in the cleaned sentences, train on this */
          if (tgt_words[tgt_ix] != -1 &&
              old_to_new.find(src_ix) != old_to_new.end()) {
            if (!convolved) {
              model->Convolve(src_sent_mat, &sent_convolved);
              sent_vec = sent_convolved * model->p1.var;
              convolved = true;
            }
            src_ix = old_to_new[src_ix];
            /* Compute error as the squared error */
            Col tgt_vec = model->tgt_word_vecs[tgt_word];
            Col src_vec = model->src_word_vecs[src_words[src_ix]];
            adouble error = model->PredError(src_vec, tgt_vec, sent_vec);
            total_error += error;
            semi_error += error;
            if (++accum == update_every) {
              semi_error.set_gradient(1.0);
              s->compute_adjoint();
              model->UpdateParams();
              semi_error = 0;
              accum = 0;
              s->new_recording();
            }
            num_words += 1;
          }
        }
        cerr << num_words << "\r";
      }
      cerr << "\nError per word: "<< total_error/num_words << "\n";
      p_file.close();
      a_file.close();
    } else {
      cerr << "\nUnable to open file\n";
      break;
    }
  } 
}

int main(int argc, char **argv){
  if (argc != 10) {
    cerr << "Usage: "<< argv[0] << " parallel_corpus " << " alignment_corpus "
         << " src_vec_corpus " << " tgt_vec_corpus " << " filt_size " << "k_max"
         << " update_every " << " num_iter " << " outfilename\n";
    cerr << "Recommended: " << argv[0] << " parallel_corpus " 
         << " alignment_corpus " << " src_vec_corpus " << " tgt_vec_corpus "
         << " 5 1 50 2" << " out.txt\n";
    exit(0);
  }
  
  string parallel_corpus = argv[1];
  string align_corpus = argv[2];
  string src_vec_corpus = argv[3];
  string tgt_vec_corpus = argv[4];
  int filt_len = stoi(argv[5]);
  int kmax = stoi(argv[6]);
  int update_every = stoi(argv[7]);
  int num_iter = stoi(argv[8]);
  string outfilename = argv[9];
 
  adept::Stack s;
  Model model(filt_len, kmax, src_vec_corpus, tgt_vec_corpus);

  cerr << "Model specification" << endl;
  cerr << "----------------" << endl;
  cerr << "Input vector length: " << model.src_len << endl;
  cerr << "Filter 1 length: " << model.filter_len1 << endl;
  cerr << "k-best 1: " << model.kbest1 << endl;
  cerr << "Filter 2 length: " << model.filter_len2 << endl;
  cerr << "k-best 2: " << model.kbest2 << endl;
  cerr << "Output vector length: " << model.tgt_len << endl;
  cerr << "----------------" << endl;

  Train(parallel_corpus, align_corpus, num_iter, update_every, &model, &s);
  //WriteParamsToFile(outfilename, model.p11, model.p2, model.p3);
  return 1;
}
