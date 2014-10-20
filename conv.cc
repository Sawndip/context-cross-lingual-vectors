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

/* Main class definition that learns the word vectors */
class Model {

 public:
  /* The parameters of the model
     p1 -- filter
     p2 -- Convert output after k-max to column
     p3 -- Convert after non-linearity to target vector
  */
  AMat p1, p2, p3;
  unsigned k;
  adouble c;
  /* Adadelta memory */
  Mat ad_p1, ad_g_p1, ad_p2, ad_g_p2, ad_p3, ad_g_p3;
  double ad_c, ad_g_c;
  /* Word vectors */
  unsigned window_size, src_len, tgt_len, hidden_len;
  mapStrUnsigned src_vocab, tgt_vocab;
  vector<Col> src_word_vecs, tgt_word_vecs;
      
  Model(const int& filter_len, const int& kmax, const string& src_vec_file,
        const string& tgt_vec_file) {
    ReadVecsFromFile(src_vec_file, &src_vocab, &src_word_vecs);
    ReadVecsFromFile(tgt_vec_file, &tgt_vocab, &tgt_word_vecs);
    src_len = src_word_vecs[0].size();
    tgt_len = tgt_word_vecs[0].size();
    hidden_len = unsigned(0.25 * src_len);
    /* Params init */
    p1 = (0.6 / sqrt(filter_len * src_len)) * 
         AMat::Random(src_len, filter_len);
    p2 = (0.6 / sqrt(hidden_len * src_len)) * AMat::Random(hidden_len, src_len);
    p3 = (0.6 / sqrt(tgt_len * hidden_len)) * AMat::Random(tgt_len, hidden_len);
    c = 1;
    /* Adadelta init */
    ad_p1 = ad_g_p1 = Mat::Zero(src_len, filter_len);
    ad_p2 = ad_g_p2 = Mat::Zero(hidden_len, src_len);
    ad_p3 = ad_g_p3 = Mat::Zero(tgt_len, hidden_len);
    ad_c = ad_g_c = 0;
  }

  void convolve_narrow(const Mat& mat, const AMat& filter, AMat* res) {
    /* Sentence should be greater than filter length */
    if (mat.rows() != filter.rows() || mat.cols() < filter.cols()) {
      cerr << "Incompatible matrix dimensions." << endl;
      cerr << "Matrix: " << mat.rows() << " " << mat.cols() << endl;
      cerr << "Filter: " << filter.rows() << " " << filter.cols() << endl;
      exit(0);
    }
    unsigned slice_len = filter.cols();
    (*res) = AMat::Zero(mat.rows(), mat.cols() - slice_len + 1);
    for (unsigned i = 0; i < res->rows(); ++i) {
      for (unsigned j = 0; j < res->cols(); ++j) {
        (*res)(i, j) = DotProd(filter.row(i),
                               mat.block(i, j, 1, slice_len));
      }
    }
  }

  void convolve_wide(const Mat& mat, const AMat& filter, AMat* res) {
    /* Append extra zero vectors at the end and beginning of sentence
       for wide convolution */
    Mat zeros = Mat::Zero(mat.rows(), filter.cols() - 1);
    Mat new_sent(mat.rows(), mat.cols() + 2 * zeros.cols());
    new_sent << zeros, mat, zeros;
    convolve_narrow(new_sent, filter, res);
  }

  void max(const AMat& mat, AMat* res) {
    (*res) = mat.rowwise().maxCoeff();
  }

  adouble ComputePredError(const unsigned& src_word_id,
                           const Col& tgt_vec_gold, Mat* sent_mat) {
    /* Make a sentence matrix by appending all word vectors
       Ignore all the noisy words (denoted by -1) */
    Col zero = Col::Zero(src_len), orig_col = sent_mat->col(src_word_id);
    ACol azero = ACol::Zero(src_len);
    sent_mat->col(src_word_id) = zero;  // Set predict word to be zero
    /* Build the prediction model here */
    AMat convolved, maxed;
    ACol non_lin;
    convolve_wide(*sent_mat, p1, &convolved);  // Wide convolution with filter
    max(convolved, &maxed);  // k-max layer
    /* Add source word vector to the maxed layer */
    maxed = ElemwiseDiff(maxed, -1 * orig_col);
    non_lin = p2 * maxed;  // Reduction to a vector of hidden_len
    ElemwiseTanh(&non_lin);  // Non-linearity
    ACol tgt_vec = p3 * non_lin;  // Predicted output
    /* Prediction done, replace the predict word column */
    sent_mat->col(src_word_id) = orig_col;
    return ElemwiseDiff(tgt_vec, tgt_vec_gold).squaredNorm();
  }

  void UpdateParams() {
    AdadeltaMatUpdate(RHO, EPSILON, &p1, &ad_p1, &ad_g_p1);
    AdadeltaMatUpdate(RHO, EPSILON, &p2, &ad_p2, &ad_g_p2);
    AdadeltaMatUpdate(RHO, EPSILON, &p3, &ad_p3, &ad_g_p3);
    AdadeltaUpdate(RHO, EPSILON, &c, &ad_c, &ad_g_c);
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
              ConsiderForPred(src[j])) {
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
        vector<string> src_tgt_pairs = split_line(a_line, ' ');
        for (unsigned j = 0; j < src_tgt_pairs.size(); ++j) {
          vector<string> index_pair = split_line(src_tgt_pairs[j], '-');
          unsigned src_ix = stoi(index_pair[0]), tgt_ix = stoi(index_pair[1]);
          unsigned src_word = src_words[src_ix], tgt_word = tgt_words[tgt_ix];
          /* If both words are in the cleaned sentences, train on this */
          if (tgt_words[tgt_ix] != -1 &&
              old_to_new.find(src_ix) != old_to_new.end()) {
            src_ix = old_to_new[src_ix];
            /* Compute error as the squared error */
            auto tgt_word_vec = model->tgt_word_vecs[tgt_word];
            adouble error = model->ComputePredError(src_ix, tgt_word_vec,
                                                    &src_sent_mat);
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
      cerr << model->c << endl;
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

  cerr << "Model parameters" << endl;
  cerr << "----------------" << endl;
  cerr << "Input vector length: " << model.src_len << endl;
  cerr << "Hidden vector length: " << model.hidden_len << endl;
  cerr << "Output vector length: " << model.tgt_len << endl;
  cerr << "----------------" << endl;

  Train(parallel_corpus, align_corpus, num_iter, update_every, &model, &s);
  WriteParamsToFile(outfilename, model.p1, model.p2, model.p3);
  return 1;
}
