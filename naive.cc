/* 
  This program predicts the target word given the source word and the context.
  The context of the word is combined linearly into a vector and then projected 
  with the source word into the hidden layer.

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

#include "loss.h"
#include "utils.h"
#include "vecops.h"

#define RHO 0.95
#define EPSILON 0.000001
#define RATE 0.05

using namespace std;
using namespace Eigen;
using adept::adouble;

/* General parameters of the model */
template <typename AT, typename T>
class Param {

 public:
  AT var;

  void Init(const int& rows, const int& cols) {
    var = (0.6 / sqrt (rows + cols)) * AT::Random(rows, cols);
    del_var = T::Zero(rows, cols);
    del_grad = T::Zero(rows, cols);
  }

  void AdadeltaUpdate(const double& rho, const double& epsilon) {
    for (unsigned i = 0; i < var.rows(); ++i) {
      for (unsigned j = 0; j < var.cols(); ++j) {
        double g = var(i, j).get_gradient();
        double accum_g = rho * del_grad(i, j) + (1 - rho) * g * g;
        double del = g * sqrt(del_var(i, j) + epsilon) / sqrt(accum_g + epsilon);
        /* Update the variable */
        var(i, j) -= del;
        /* Update memory */
        del_grad(i, j) = accum_g;
        del_var(i, j) = rho * del_var(i, j) + (1 - rho) * del * del;
      }
    }
  }

  void AdagradUpdate(const double& rate) {
    for (unsigned i = 0; i < var.rows(); ++i) {
      for (unsigned j = 0; j < var.cols(); ++j) {
        double g = var(i, j).get_gradient();
        if (g) {
          /* Update memory */
          del_grad(i, j) += g * g;
          /* Update variable */
          var(i, j) -= rate * g / sqrt(del_grad(i, j));
        }
      }
    }
  }

  adouble L2() {
    adouble sum = 0;
    for (unsigned i = 0; i < var.rows(); ++i)
      for (unsigned j = 0; j < var.cols(); ++j)
        sum += var(i, j) * var(i, j);
    return sum;
  }

  void WriteToFile(ofstream& out) {
    out << var.rows() << " " << var.cols() << " ";
    for (unsigned i = 0; i < var.rows(); ++i) {
      for(unsigned j = 0; j < var.cols(); ++j) 
        out << var(i, j) << " ";
    }
    out << endl;
  }

  void ReadFromFile(ifstream& in) {
    string line;
    getline(in, line);
    vector<string> data = split_line(line, ' ');
    int rows = stoi(data[0]), cols = stoi(data[1]);
    var = AT::Zero(rows, cols);
    for (int i = 2; i < data.size(); ++i)
      var((i-2)/cols, (i-2)%cols) = stod(data[i]);
  }

 private:
  T del_var, del_grad;  // Adadelta memory
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

  void AdagradUpdate(const double& rate) {
    filter.AdagradUpdate(rate);
    bias.AdagradUpdate(rate);
  }

  void WriteToFile(ofstream& out) {
    filter.WriteToFile(out);
    bias.WriteToFile(out);
  }

  void ReadFromFile(ifstream& in) {
    filter.ReadFromFile(in);
    bias.ReadFromFile(in);
  }
  
  // The biases are not being regularized
  adouble L2() { return filter.L2(); }
};

/* Main class definition that learns the word vectors */
class Model {

 public:
  /* The parameters of the model */
  Param<AMat, Mat> to_hidden, to_tgt, left, right;
  Param<ACol, Col> hidden_bias, tgt_vec_bias, bigram_bias;
  Param<ACol, Col> context_weights, tgt_word_bias;
  int src_len, tgt_len, context_len;
  
  Model() {}
      
  Model(const int& context_size, const int& hidden_len, const int& src_vec_len,
        const int& tgt_vec_len, const int& tgt_vocab_len) {
    src_len = src_vec_len;
    tgt_len = tgt_vec_len;
    context_len = context_size;
    /* Neural network params initialization */
    to_hidden.Init(hidden_len, 2 * src_len);
    hidden_bias.Init(hidden_len, 1);
    to_tgt.Init(tgt_len, hidden_len);
    tgt_vec_bias.Init(tgt_len, 1);
    left.Init(src_len, src_len);
    right.Init(src_len, src_len);
    bigram_bias.Init(src_len, 1);
    /* Language model bias */
    tgt_word_bias.Init(tgt_vocab_len, 1);
    /* Linear addition params initialization */
    context_weights.Init(2 * context_len, 1);
  }

  template<typename T> void NonLinearity(T* vec) { ElemwiseHardTanh(vec); }
  
  void GetSentVector(const vector<int>& words, const vector<Col>& src_vecs,
                     ACol* pred_vec) {
    Col left_vec = Col::Zero(src_len), right_vec = Col::Zero(src_len);
    for (unsigned i = 0; i < words.size(); ++i) {
      if (words[i] != -1) {
        if (i < words.size() - 1) left_vec += src_vecs[words[i]];
        if (i > 0) right_vec += src_vecs[words[i]];
      }
    }
    ACol temp_left, temp_right, temp_non_linear;
    adouble unit = 1;
    ScalarProd(unit, left_vec, &temp_left);
    ScalarProd(unit, right_vec, &temp_right);
    *pred_vec = left.var * temp_left + right.var * temp_right;
    *pred_vec += bigram_bias.var;
    NonLinearity(pred_vec);
    *pred_vec /= words.size();
  }

  void TransVecInContext(const Col& src_vec, const ACol& sent_vec,
                         ACol* pred_vec) {
    ACol appended(2 * src_len), temp;
    adouble unit = 1;
    ScalarProd(unit, src_vec, &temp);
    appended << sent_vec, temp;
    /* Get the hidden layer */
    ACol hidden = to_hidden.var * appended + hidden_bias.var;
    NonLinearity(&hidden);
    *pred_vec = to_tgt.var * hidden + tgt_vec_bias.var;
  }

  void
  TransVecInContextOld(const unsigned& src_word, const vector<int>& context,
                    const vector<Col>& src_vecs, ACol* pred_vec) {
    // Add the context linearly
    ACol linear_sum = ACol::Zero(src_len, 1);
    for (unsigned i = 0; i < context.size(); ++i) {
      ACol temp;
      ScalarProd(context_weights.var[i], src_vecs[context[i]], &temp);
      linear_sum += temp;
    }
    ACol appended(2 * src_len, 1), temp;
    adouble unit = 1;
    ScalarProd(unit, src_vecs[src_word], &temp);
    appended << linear_sum, temp;
    /* Get the hidden layer */
    ACol hidden = to_hidden.var * appended + hidden_bias.var;
    NonLinearity(&hidden);
    *pred_vec = to_tgt.var * hidden + tgt_vec_bias.var;
  }

  adouble L2() {
    return to_hidden.L2() + to_tgt.L2() + context_weights.L2() + 
           left.L2() + right.L2();
  }

  void UpdateParams() {
    left.AdagradUpdate(RATE);
    right.AdagradUpdate(RATE);
    bigram_bias.AdagradUpdate(RATE);
    to_hidden.AdagradUpdate(RATE);
    to_tgt.AdagradUpdate(RATE);
    hidden_bias.AdagradUpdate(RATE);
    tgt_vec_bias.AdagradUpdate(RATE);
    //context_weights.AdagradUpdate(RATE);
    tgt_word_bias.AdagradUpdate(RATE);
  }

  void WriteParamsToFile(const string& filename) {
    ofstream outfile(filename);
    if (outfile.is_open()) {
      outfile.precision(3);
      to_hidden.WriteToFile(outfile);
      hidden_bias.WriteToFile(outfile);
      to_tgt.WriteToFile(outfile);
      tgt_vec_bias.WriteToFile(outfile);
      //context_weights.WriteToFile(outfile);
      tgt_word_bias.WriteToFile(outfile);
      outfile.close();
      cerr << "\nWritten parameters to: " << filename;
    } else {
      cerr << "\nFailed to open " << filename;
    }
  }

  void InitParamsFromFile(const string& filename) {
    ifstream infile(filename);
    if (infile.is_open()) {
      to_hidden.ReadFromFile(infile);
      hidden_bias.ReadFromFile(infile);
      to_tgt.ReadFromFile(infile);
      tgt_vec_bias.ReadFromFile(infile);
      //context_weights.ReadFromFile(infile);
      tgt_word_bias.ReadFromFile(infile);
      infile.close();
      cerr << "\nRead parameters from: " << filename;
    } else {
      cerr << "\nCould not open: " << filename;
    }
  }
};

void
EvalDev(const string& p_corpus, const string& a_corpus,
        const vector<Col>& src_word_vecs, const vector<Col>& tgt_word_vecs,
        const mapStrUnsigned& src_vocab, const mapStrUnsigned& tgt_vocab,
        Model& model, adept::Stack& s) {
  ifstream p_file(p_corpus.c_str()), a_file(a_corpus.c_str());
  string p_line, a_line;
  vector<int> src_words, tgt_words;
  unsigned num_words = 0;
  adouble nllh = 0;
  if (p_file.is_open() && a_file.is_open()) {
    while (getline(p_file, p_line) && getline(a_file, a_line)) {
      s.new_recording();
      /* Extracting words from sentences */
      vector<string> src_tgt = split_line(p_line, '\t');
      vector<string> src = split_line(src_tgt[0], ' ');
      vector<string> tgt = split_line(src_tgt[1], ' ');
      src_words.clear();
      tgt_words.clear();
      /* Read and Clean Source sentence */
      for (unsigned j = 0; j < src.size(); ++j) {
        auto it = src_vocab.find(src[j]);
        if (it != src_vocab.end()) src_words.push_back(it->second);
        else src_words.push_back(-1); // word not in vocab
      }
      /* Target sentence */
      for (unsigned j=0; j<tgt.size(); ++j) {
        auto it = tgt_vocab.find(tgt[j]);
        if (it != tgt_vocab.end()) tgt_words.push_back(it->second);
        else tgt_words.push_back(-1); // word not in vocab
      }
      ACol sent_vec;
      model.GetSentVector(src_words, src_word_vecs, &sent_vec);
      /* Read alignments and train */
      vector<string> src_tgt_pairs = split_line(a_line, ' ');
      for (unsigned j = 0; j < src_tgt_pairs.size(); ++j) {
        vector<string> index_pair = split_line(src_tgt_pairs[j], '-');
        int src_ix = stoi(index_pair[0]), tgt_ix = stoi(index_pair[1]);
        int tgt_word = tgt_words[tgt_ix], src_word = src_words[src_ix];
        /* If both words are in the cleaned sentences, train on this */
        if (tgt_word != -1 && src_word != -1) {
          Col tgt_vec = tgt_word_vecs[tgt_word];
          /* Get context */
          //vector<int> context;
          //GetContext(src_words, src, src_ix, model.context_len, &context);
          ACol pred_tgt_vec;
          model.TransVecInContext(src_word_vecs[src_word], sent_vec,
                                  &pred_tgt_vec);
          //model.TransVecInContext(src_word, context, src_word_vecs,
          //                        &pred_tgt_vec);
          adouble nlp = LogProbLoss(pred_tgt_vec, tgt_word,
                                    tgt_word_vecs, model.tgt_word_bias.var);
          nllh += nlp;
          num_words += 1;
        }
      }
    cerr << num_words << "\r";
    }
    p_file.close();
    a_file.close();
    cerr << "\nPerplexity: " << exp(nllh / num_words);
  } else {
    cerr << "\nUnable to open file\n";
  }
}


/*
   The function that trains a neural network to convert word vectors
   from source language to target language while taking the context of
   the source word in account. The translation pairs are determined using
   word alignments.

   The parameters are self-explanatory.
*/
void Train(const string& p_corpus, const string& a_corpus,
           const string& p_dev, const string& a_dev,
           const string& outfilename, const int& context_len,
           const int& hidden_len, const int& noise_size,
           const int& num_iter, const double& reg_const,
           const vector<Col>& src_word_vecs, const vector<Col>& tgt_word_vecs,
           const mapStrUnsigned& src_vocab, const mapStrUnsigned& tgt_vocab,
           adept::Stack& s) {  
  AliasSampler sampler;
  vector<double> noise_dist(tgt_vocab.size(), 0.0);
  Model model(context_len, hidden_len, src_word_vecs[0].size(),
              tgt_word_vecs[0].size(), tgt_word_vecs.size());
  GetUnigramDist(p_corpus, tgt_vocab, TARGET, &noise_dist);
  sampler.Init(noise_dist);
  for (unsigned i = 0; i < num_iter; ++i) {
    cerr << "\nIteration: " << i+1 << endl;
    ifstream p_file(p_corpus.c_str()), a_file(a_corpus.c_str());
    string p_line, a_line;
    vector<int> src_words, tgt_words;
    unsigned num_words = 0;
    adouble total_error = 0, nllh = 0;
    s.new_recording();
    if (p_file.is_open() && a_file.is_open()) {
      while (getline(p_file, p_line) && getline(a_file, a_line)) {
        /* Extracting words from sentences */
        vector<string> src_tgt = split_line(p_line, '\t');
        vector<string> src = split_line(src_tgt[0], ' ');
        vector<string> tgt = split_line(src_tgt[1], ' ');
        src_words.clear();
        tgt_words.clear();
        /* Read and Clean Source sentence */
        for (unsigned j = 0; j < src.size(); ++j) {
          auto it = src_vocab.find(src[j]);
          if (it != src_vocab.end()) src_words.push_back(it->second);
          else src_words.push_back(-1); // word not in vocab
        }
        /* Target sentence */
        for (unsigned j=0; j<tgt.size(); ++j) {
          auto it = tgt_vocab.find(tgt[j]);
          if (it != tgt_vocab.end()) tgt_words.push_back(it->second);
          else tgt_words.push_back(-1); // word not in vocab
        }
        ACol sent_vec;
        model.GetSentVector(src_words, src_word_vecs, &sent_vec);
        /* Read alignments and train */
        vector<string> src_tgt_pairs = split_line(a_line, ' ');
        for (unsigned j = 0; j < src_tgt_pairs.size(); ++j) {
          vector<string> index_pair = split_line(src_tgt_pairs[j], '-');
          int src_ix = stoi(index_pair[0]), tgt_ix = stoi(index_pair[1]);
          int tgt_word = tgt_words[tgt_ix], src_word = src_words[src_ix];
          /* If both words are in the cleaned sentences, train on this */
          if (tgt_word != -1 && src_word != -1) {
            Col tgt_vec = tgt_word_vecs[tgt_word];
            //cerr << "Here";
            /* Get context */
            //vector<int> context;
            //GetContext(src_words, src, src_ix, context_len, &context);
            ACol pred_tgt_vec;
            //model.TransVecInContext(src_word, context, src_word_vecs,
            //                        &pred_tgt_vec);
            model.TransVecInContext(src_word_vecs[src_word], sent_vec,
                                    &pred_tgt_vec);
            adouble error = NCELoss(pred_tgt_vec, tgt_word,
                                    tgt_word_vecs, model.tgt_word_bias.var,
                                    noise_size, noise_dist, sampler);
            error += reg_const * model.L2();
            total_error += error;
            /* Calcuate gradient and update parameters */
            error.set_gradient(1.0);
            s.compute_adjoint();
            model.UpdateParams();
            s.new_recording();
            num_words += 1;
          }
        }
        cerr << num_words << "\r";
      }
      p_file.close();
      a_file.close();
      cerr << "\nError: " << total_error / num_words;
      cerr << "\nL2: " << model.L2() << endl;
      EvalDev(p_dev, a_dev, src_word_vecs, tgt_word_vecs, src_vocab, tgt_vocab,
              model, s);
      //model.WriteParamsToFile(outfilename + "_i" + to_string(i+1));
    } else {
      cerr << "\nUnable to open file\n";
      break;
    }
  } 
}

/* Returns the vector representation for a word in context.

 The input is a line with two tab separated columns, where the first column is 
 the index of the word for which the vector is to be obtained and the second
 column is the sentence. "model" is the trained neural network, "word_vecs" is
 a vector of individual word vectors and "vocab" the corresponding the word
 vocabulary.

pair<bool, ACol>
Decode(const string& line, const vector<Col>& word_vecs,
       const mapStrUnsigned& vocab, Model* model) {
    // Extracting words from sentences
    vector<string> sent_and_index = split_line(line, '\t');
    vector<string> words = split_line(sent_and_index[1], ' ');
    int query_index = stoi(sent_and_index[0]);
    // Read and clean sentence & make sentence matrix 
    vector<unsigned> clean_words;
    clean_words.clear();
    unsigned index = 0, new_query_index = -1;
    for (unsigned j = 0; j < words.size(); ++j) {
      auto it = vocab.find(words[j]);
      if (j == query_index) {
        if (it == vocab.end()) {
          cerr << "Word not in vocab: " << words[j] << endl;
          ACol dummy;
          return make_pair(false, dummy);
        } else {
          clean_words.push_back(it->second);
          new_query_index = index++;
        }
      }
      if (it != vocab.end() && ConsiderForContext(words[j])) {
        clean_words.push_back(it->second);
        index++;
      }
    }
    // Make a sentence matrix 
    Mat sent_mat(word_vecs[0].size(), clean_words.size());
    for (unsigned j = 0; j < clean_words.size(); ++j)
      sent_mat.col(j) = word_vecs[clean_words[j]];
    // Read the alignment line 
    ACol sent_vec, pred_vec;
    model->GetSentVector(sent_mat, &sent_vec);
    model->VecInContext(sent_mat.col(new_query_index), sent_vec, &pred_vec);
    return make_pair(true, pred_vec);
} */

int main(int argc, char **argv){
  adept::Stack s;
  if (argc == 13) {
    string parallel_corpus = argv[1];
    string align_corpus = argv[2];
    string p_dev = argv[3];
    string a_dev = argv[4];
    string src_vec_corpus = argv[5];
    string tgt_vec_corpus = argv[6];
    int context_len = stoi(argv[7]);
    int hidden_len = stoi(argv[8]);
    int noise_size = stoi(argv[9]);
    int num_iter = stoi(argv[10]);
    double reg_const = stod(argv[11]);
    string outfilename = argv[12];

    mapStrUnsigned src_vocab, tgt_vocab;
    vector<Col> src_word_vecs, tgt_word_vecs;
    ReadVecsFromFile(parallel_corpus, SOURCE, src_vec_corpus, &src_vocab,
                     &src_word_vecs);
    ReadVecsFromFile(parallel_corpus, TARGET, tgt_vec_corpus, &tgt_vocab,
                     &tgt_word_vecs);
    int src_len = src_word_vecs[0].size();
    int tgt_len = tgt_word_vecs[0].size();

    cerr << "Model specification" << endl;
    cerr << "----------------" << endl;
    cerr << "Input vector length: " << src_len << endl;
    cerr << "Context length: " << context_len << endl;
    cerr << "Hidden length: " << hidden_len << endl;
    cerr << "Reg const: " << reg_const << endl;
    cerr << "Output vector length: " << tgt_len << endl;
    cerr << "----------------" << endl;

    Train(parallel_corpus, align_corpus, p_dev, a_dev, outfilename,
          context_len, hidden_len, noise_size, num_iter, reg_const,
          src_word_vecs, tgt_word_vecs, src_vocab, tgt_vocab, s);
  } else if (argc == 6) {
    /*
    adept::Stack s;
    string parallel_corpus = argv[1];
    string sent_file = argv[2];
    string word_vecs_file = argv[3];
    string params_file = argv[4];
    string out_file = argv[5];

    mapStrUnsigned vocab;
    vector<Col> word_vecs;
    ReadVecsFromFile(parallel_corpus, SOURCE, word_vecs_file, &vocab,
                     &word_vecs);
    cerr << "Vectors read" << endl;
    Model model;
    model.InitParamsFromFile(params_file);
    cerr << "Initialized" << endl; 
    ifstream infile(sent_file.c_str());
    if (infile.is_open()) {
      string line;
      ofstream outfile(out_file.c_str());
      while (getline(infile, line)) {
        Param<ACol, Col> word_in_context_vec;
        auto res = Decode(line, word_vecs, vocab, &model);
        if (res.first) {
          word_in_context_vec.var = res.second;
          word_in_context_vec.WriteToFile(outfile);
        } else {
          outfile << "NULL" << endl;
        }
      }
      infile.close();
      outfile.close();
    } else {
      cerr << "Could not open file " << infile << endl;
    }
    */
  } else {
    cerr << endl;
    cerr << "Usage: "<< argv[0] << " parallel_corpus alignment_corpus "
         << " src_vec_corpus tgt_vec_corpus context_len hidden_len noise_size "
         << " num_iter reg_const outfilename\n";
  }

  return 1;
}
