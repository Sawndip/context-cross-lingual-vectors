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
  /* The parameters of the model */
  mapIntCol context;
  mapIntCol ag_context_mem;
  Mat convert_to_tgt;
  Mat ag_convert_to_tgt_mem;
    
public:
      
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
  }
      
  void InitParams() {
    random_col_map(window_size, src_vec_len, &context);
    zero_col_map(window_size, src_vec_len, &ag_context_mem);
    convert_to_tgt = Mat::Random(tgt_vec_len, src_vec_len);
    ag_convert_to_tgt_mem = Mat::Zero(tgt_vec_len, src_vec_len);
  }

  double Update(const mapIntUnsigned& context_words, const Col& tgt_vec_gold,
                const double rate) {
    Col hidden = Col::Zero(src_vec_len);
    for (auto it = context_words.begin(); it != context_words.end(); ++it) {
      if (context[it->first].rows() == src_word_vecs[it->second].rows())
        hidden += context[it->first].cwiseProduct(src_word_vecs[it->second]);
      else
        return -1; // This is erroneous, this should never happen, correct it
    }
    ElemwiseTanh(&hidden);
    Col tgt_vec = convert_to_tgt * hidden;
    if (tgt_vec.rows() == tgt_vec_gold.rows()) {
      /* Grad for the convert_to_lang parameter */
      /*Mat grad = 2 * (tgt_vec - tgt_vec_gold) * hidden.transpose();
      ag_convert_to_tgt_mem = ag_convert_to_tgt_mem.array() + grad.array().square();
      Mat mem_sqrt = ag_convert_to_tgt_mem.array().sqrt();
      convert_to_tgt -= rate * grad.cwiseQuotient(mem_sqrt);*/
      /* Grad for the context vectors */
      /*Col sech2 = Col::Ones(hidden.rows()).array() - hidden.array().square();
      double temp = 2*(tgt_vec - tgt_vec_gold).transpose() * convert_to_tgt * sech2;
      for (auto it = context_words.begin(); it != context_words.end(); ++it) {
        Col con_grad = src_word_vecs[it->second].transpose() * temp;
        ag_context_mem[it->first] = ag_context_mem[it->first].array() +
                                    con_grad.array().square();
        Col con_mem_sqrt = ag_context_mem[it->first].array().sqrt();
        context[it->first] -= rate * con_grad.cwiseQuotient(con_mem_sqrt);*/
      }
      return (tgt_vec - tgt_vec_gold).squaredNorm();
    }
    return -1; // This should erroneous, it should never happen
  }
};

void Train(const string& p_corpus, const string& a_corpus,
           const unsigned& num_iter,
           const double& learning_rate, Model* model) {
  for (unsigned i=0; i<num_iter; ++i) {
    double rate = learning_rate/(i+1);
    cerr << "\nIteration: " << i+1;
    cerr << "\nLearning rate: " << rate << "\n";
    ifstream p_file(p_corpus.c_str()), a_file(a_corpus.c_str());
    string p_line, a_line;
    vector<unsigned> src_words, tgt_words;
    unsigned numWords = 0, erroneous_cases = 0;
    adouble total_error = 0;

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
        unsigned j;
        for (j = 0; j < src_tgt_pairs.size(); ++j) {
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
            double output = model->Update(context_words, tgt_word_vec, rate);
            if (output > 0) total_error += output;
            else erroneous_cases += 1;
          }
        }
        numWords += tgt_words.size();
        cerr << numWords << "\r";
        //cerr << int(numWords/1000) << "K\r";
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
  unsigned num_iter = 5;
  int window = 2;
  double learning_rate = 0.5; 
  if (argc != 5) {
    cerr << "Usage: "<< argv[0] << " parallel_corpus " << " alignment_corpus " 
         << " src_vec_corpus " << " tgt_vec_corpus\n";
    exit(0);
  }
  adept::Stack s;
  string parallel_corpus = argv[1];
  string align_corpus = argv[2];
  string src_vec_corpus = argv[3];
  string tgt_vec_corpus = argv[4];
 
  Model obj(window, src_vec_corpus, tgt_vec_corpus);
  obj.InitParams();
  Train(parallel_corpus, align_corpus, num_iter, learning_rate, &obj);
  return 1;
}
