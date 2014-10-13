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

using namespace std;
using namespace Eigen;
using adept::adouble;

void GetContext(const vector<unsigned>& words, unsigned tgt_word_ix,
                int window_size, mapIntUnsigned* t_context_words) {
  mapIntUnsigned& context_words = *t_context_words;
  context_words.clear();
  for (int i = -window_size; i <= window_size; ++i) {
    int word_index = i + tgt_word_ix;
    if (word_index >= 0 && word_index < words.size()) {
      if (words[word_index] != -1) // word not in vector vocab
        context_words[i] = words[word_index];
    }
  }
}

/* Main class definition that learns the word vectors */

class Model {

 public:
  /* The parameters of the model */
  mapIntACol context;
  mapIntCol ag_context_mem;
  AMat convert_to_tgt;
  Mat ag_convert_to_tgt_mem;
  /* Word vectors */  
  unsigned window_size, src_vec_len, tgt_vec_len;
  mapStrUnsigned src_vocab, tgt_vocab;
  vector<Col> src_word_vecs, tgt_word_vecs;
      
  Model(const int& window, const string& src_vec_file,
        const string& tgt_vec_file) {
    window_size = window;
    ReadVecsFromFile(src_vec_file, &src_vocab, &src_word_vecs);
    ReadVecsFromFile(tgt_vec_file, &tgt_vocab, &tgt_word_vecs);
    src_vec_len = src_word_vecs[0].size();
    tgt_vec_len = tgt_word_vecs[0].size();

    random_acol_map(window_size, src_vec_len, &context);
    zero_col_map(window_size, src_vec_len, &ag_context_mem);
    convert_to_tgt = AMat::Random(tgt_vec_len, src_vec_len);
    ag_convert_to_tgt_mem = Mat::Zero(tgt_vec_len, src_vec_len);
  }

  adouble ComputePredError(const mapIntUnsigned& context_words,
                           const Col& tgt_vec_gold) {
    ACol hidden = ACol::Zero(src_vec_len);
    for (auto it = context_words.begin(); it != context_words.end(); ++it) {
      if (context[it->first].rows() == src_word_vecs[it->second].rows()) {
        ElemwiseProdSum(context[it->first], src_word_vecs[it->second],
                        &hidden);
      }
    }
    ElemwiseTanh(&hidden);
    ACol tgt_vec = convert_to_tgt * hidden;
    if (tgt_vec.size() == tgt_vec_gold.size())
      return ElemwiseDiff(tgt_vec, tgt_vec_gold).squaredNorm();
    else
      return -1;
  }

  void UpdateParams(const double& rate) {
    for (auto it = context.begin(); it != context.end(); ++it) {
      for (unsigned i = 0; i < src_vec_len; ++i) {
        double g = it->second(i, 0).get_gradient();
        if (g) {
          double s = ag_context_mem[it->first](i, 0) += g * g;
          it->second(i, 0) -= rate * g / sqrt(s);
        }
      }
    }

    for (unsigned i = 0; i < convert_to_tgt.rows(); ++i) {
      for (unsigned j = 0; j < convert_to_tgt.cols(); ++j) {
        double g = convert_to_tgt(i, j).get_gradient();
        if (g) {
          double s = ag_convert_to_tgt_mem(i, j) += g * g;
          convert_to_tgt(i, j) -= rate * g / sqrt(s);
        }
      }
    }
  }

};

void Train(const string& p_corpus, const string& a_corpus,
           const int& num_iter, const int& update_every,
           const double& learning_rate, Model* model, adept::Stack* s) {
  for (unsigned i=0; i<num_iter; ++i) {
    double rate = learning_rate / (i + 1);
    cerr << "\nIteration: " << i+1;
    cerr << "\nLearning rate: " << rate << "\n";
    ifstream p_file(p_corpus.c_str()), a_file(a_corpus.c_str());
    string p_line, a_line;
    vector<unsigned> src_words, tgt_words;
    unsigned numWords = 0, erroneous_cases = 0;
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
        /* Source sentence */
        for (unsigned j=0; j<src.size(); ++j) {
          if (model->src_vocab.find(src[j]) != model->src_vocab.end())
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
          unsigned src_ix = stoi(index_pair[0]);
          unsigned tgt_ix = stoi(index_pair[1]);
          /* If both words in vocab, this is a training example */
          if (src_words[src_ix] != -1 && tgt_words[tgt_ix] != -1) {
            mapIntUnsigned context_words;
            context_words.clear();
            GetContext(src_words, src_ix, model->window_size, &context_words);
            /* Compute error as the squared error */
            auto tgt_word_vec = model->tgt_word_vecs[tgt_words[tgt_ix]];
            adouble error = model->ComputePredError(context_words,
                                                    tgt_word_vec);
            if (error > 0) {
              total_error += error;
              semi_error += error;
              if (++accum == update_every) {
                semi_error.set_gradient(1.0);
                s->compute_adjoint();
                model->UpdateParams(rate);
                semi_error = 0;
                accum = 0;
                s->new_recording();
              }
            } else { 
              erroneous_cases += 1;
            }
          }
        }
        numWords += tgt_words.size();
        cerr << (numWords/1000) << "K\r";
      }
      cerr << "\nError: " << total_error << endl;
      cerr << "Erroneous: " << erroneous_cases << endl;
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
         << " src_vec_corpus " << " tgt_vec_corpus " << " context_size "
         << " update_every " << " learning_rate " << " num_iter "
         << " outfilename\n";
    cerr << "Recommended: " << argv[0] << " parallel_corpus " 
         << " alignment_corpus " << " src_vec_corpus " << " tgt_vec_corpus "
         << " 5 " << " 1 " << " 0.5 " << " 1 " << " out.txt\n";
    exit(0);
  }
  
  string parallel_corpus = argv[1];
  string align_corpus = argv[2];
  string src_vec_corpus = argv[3];
  string tgt_vec_corpus = argv[4];
  int window = stoi(argv[5]);
  int update_every = stoi(argv[6]);
  double learning_rate = stod(argv[7]);
  int num_iter = stoi(argv[8]);
  string outfilename = argv[9];
 
  adept::Stack s;
  Model obj(window, src_vec_corpus, tgt_vec_corpus);
  Train(parallel_corpus, align_corpus, num_iter, update_every, learning_rate,
        &obj, &s);
  WriteParamsToFile(outfilename, obj.context, obj.convert_to_tgt);
  return 1;
}
