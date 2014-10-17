/* 
  This program learns a mapping between vectors of two languages. 
  The mappin is expressed as a neural network model and the parameters
  of the neural network are optimized using Adaptive gradient descent. 
  
  The gradient of the error function is calculated using automatic
  differentiation from library adept.h
*/
#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <numeric>
#include <cmath>
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
  /* The parameters of the model */
  AMat context_self, context_other;
  AMat convert_to_tgt;
  /* Adadelta memory */
  Mat ad_context_self, ad_g_context_self;
  Mat ad_context_other, ad_g_context_other;
  Mat ad_convert_to_tgt, ad_g_convert_to_tgt;
  /* Word vectors */
  unsigned window_size, src_vec_len, tgt_vec_len, hidden_len;
  mapStrUnsigned src_vocab, tgt_vocab;
  vector<Col> src_word_vecs, tgt_word_vecs;
      
  Model(const int& window, const string& src_vec_file,
        const string& tgt_vec_file) {
    window_size = window;
    ReadVecsFromFile(src_vec_file, &src_vocab, &src_word_vecs);
    ReadVecsFromFile(tgt_vec_file, &tgt_vocab, &tgt_word_vecs);
    src_vec_len = src_word_vecs[0].size();
    tgt_vec_len = tgt_word_vecs[0].size();
    hidden_len = int(src_vec_len);
    /* Params init */
    context_self = context_other = 0.6 / sqrt(hidden_len * src_vec_len) *
                                   AMat::Random(hidden_len, src_vec_len);
    convert_to_tgt = AMat::Random(tgt_vec_len, hidden_len);
    convert_to_tgt *= 0.6 / sqrt(tgt_vec_len * hidden_len);
    /* Adadelta init */
    ad_context_self = ad_g_context_self = ad_context_other = 
                      ad_g_context_other = Mat::Zero(hidden_len, src_vec_len);
    ad_convert_to_tgt = ad_g_convert_to_tgt = 
                        Mat::Zero(tgt_vec_len, hidden_len);
  }

  adouble ComputePredError(const unsigned& src_word,
                           const mapIntUnsigned& context_words,
                           const Col& tgt_vec_gold) {
    Col context_vec_sum = Col::Zero(src_vec_len);
    ACol hidden = ACol::Zero(hidden_len);
    for (auto it = context_words.begin(); it != context_words.end(); ++it)
      context_vec_sum += src_word_vecs[it->second];  // add the context vectors
    ProdSum(context_other, context_vec_sum, &hidden);
    ProdSum(context_self, src_word_vecs[src_word], &hidden);
    ElemwiseTanh(&hidden);
    ACol tgt_vec = convert_to_tgt * hidden;
    return ElemwiseDiff(tgt_vec, tgt_vec_gold).squaredNorm();
  }

  void UpdateParamsAdadelta() {
    AdadeltaMatUpdate(RHO, EPSILON, &context_self, &ad_context_self,
                      &ad_g_context_self);
    AdadeltaMatUpdate(RHO, EPSILON, &context_other, &ad_context_other,
                      &ad_g_context_other);
    AdadeltaMatUpdate(RHO, EPSILON, &convert_to_tgt, &ad_convert_to_tgt,
                      &ad_g_convert_to_tgt);
  }

};

void Train(const string& p_corpus, const string& a_corpus, const int& num_iter,
           const int& update_every, Model* model, adept::Stack* s) {
  for (unsigned i=0; i<num_iter; ++i) {
    cerr << "\nIteration: " << i+1 << endl;
    ifstream p_file(p_corpus.c_str()), a_file(a_corpus.c_str());
    string p_line, a_line;
    vector<unsigned> src_words, tgt_words;
    unsigned numWords = 0, erroneous_cases = 0;
    adouble total_error = 0, semi_error = 0;
    int accum = 0, print_if = 100000, print_count = 0;
    s->new_recording();
    if (p_file.is_open() && a_file.is_open()) {
      while (getline(p_file, p_line) && getline(a_file, a_line)) {
        /* Extracting words from sentences */
        vector<string> src_tgt = split_line(p_line, '\t');
        vector<string> src = split_line(src_tgt[0], ' ');
        vector<string> tgt = split_line(src_tgt[1], ' ');
        src_words.clear();
        tgt_words.clear();
        /* Source sentence */
        for (unsigned j=0; j<src.size(); ++j) {
          if (model->src_vocab.find(src[j]) != model->src_vocab.end() &&
              ConsiderString(src[j]))
            src_words.push_back(model->src_vocab[src[j]]);
          else
            src_words.push_back(-1); // word not in vocab
        }
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
          /* If both words in vocab, this is a training example */
          if (src_words[src_ix] != -1 && tgt_words[tgt_ix] != -1) {
            mapIntUnsigned context_words;
            context_words.clear();
            GetContext(src_words, src_ix, model->window_size, &context_words);
            /* Compute error as the squared error */
            auto tgt_word_vec = model->tgt_word_vecs[tgt_words[tgt_ix]];
            adouble error = model->ComputePredError(src_word, context_words,
                                                    tgt_word_vec);
            total_error += error;
            semi_error += error;
            if (++accum == update_every) {
              semi_error.set_gradient(1.0);
              s->compute_adjoint();
              model->UpdateParamsAdadelta();
              semi_error = 0;
              accum = 0;
              s->new_recording();
            }
          }
        }
        numWords += src_tgt_pairs.size();
        print_count += src_tgt_pairs.size();
        /*if (print_count > print_if) {
          print_count = 0;
          cerr << "Error per word: "<< total_error/numWords << "\n";
          numWords = 0;
          total_error = 0;
        }*/
        cerr << numWords << "\r";
      }
      cerr << "\nError per word: "<< total_error/numWords << "\n";
      p_file.close();
      a_file.close();
    } else {
      cerr << "\nUnable to open file\n";
      break;
    }
  } 
}

int main(int argc, char **argv){
  if (argc != 9) {
    cerr << "Usage: "<< argv[0] << " parallel_corpus " << " alignment_corpus "
         << " src_vec_corpus " << " tgt_vec_corpus " << " context_size "
         << " update_every " << " num_iter " << " outfilename\n";
    cerr << "Recommended: " << argv[0] << " parallel_corpus " 
         << " alignment_corpus " << " src_vec_corpus " << " tgt_vec_corpus "
         << " 5 " << " 1 " << " 1 " << " out.txt\n";
    exit(0);
  }
  
  string parallel_corpus = argv[1];
  string align_corpus = argv[2];
  string src_vec_corpus = argv[3];
  string tgt_vec_corpus = argv[4];
  int window = stoi(argv[5]);
  int update_every = stoi(argv[6]);
  int num_iter = stoi(argv[7]);
  string outfilename = argv[8];
 
  adept::Stack s;
  Model obj(window, src_vec_corpus, tgt_vec_corpus);
  Train(parallel_corpus, align_corpus, num_iter, update_every, &obj, &s);
  WriteParamsToFile(outfilename, obj.context_self, obj.context_other,
                    obj.convert_to_tgt);
  return 1;
}
