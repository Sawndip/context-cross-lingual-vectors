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

#include "loss.h"
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

  void AdagradUpdate(const double& rho, const double& epsilon) {
    for (unsigned i = 0; i < var.rows(); ++i) {
      for (unsigned j = 0; j < var.cols(); ++j) {
        double g = var(i, j).get_gradient();
        if (g) {
          /* Update memory */
          del_grad(i, j) += g * g;
          /* Update variable */
          var(i, j) -= 0.05 * g / sqrt(del_grad(i, j));
        }
      }
    }
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

  void AdagradUpdate(const double& rho, const double& epsilon) {
    filter.AdagradUpdate(rho, epsilon);
    bias.AdagradUpdate(rho, epsilon);
  }

  void WriteToFile(ofstream& out) {
    filter.WriteToFile(out);
    bias.WriteToFile(out);
  }

  void ReadFromFile(ifstream& in) {
    filter.ReadFromFile(in);
    bias.ReadFromFile(in);
  }
};

/* Main class definition that learns the word vectors */
class Model {

 public:
  /* The parameters of the model */
  Filter f11, f12, f21, f22;  // Convolution
  Param<AMat, Mat> p1, p2, p3;  // Post-convolution
  Param<ACol, Col> p2_b, p3_b, tgt_bias;  // Post-convolution
  int src_len, tgt_len, filter_len, kmax;
  
  Model() {}
      
  Model(const int& filt_len, const int& k, const int& src_vec_len,
        const int& tgt_vec_len, const int& num_src_words,
        const int& num_tgt_words) {
    src_len = src_vec_len;
    tgt_len = tgt_vec_len;
    filter_len = filt_len;
    kmax = k;
    if (filter_len < 3) {
      cerr << "Minimum filter len: " << 3 << endl;
      exit(0);
    }
    /* Params initialization */
    f11.Init(src_len, filter_len);
    f12.Init(src_len, filter_len);
    f21.Init(src_len, filter_len - 1);
    f22.Init(src_len, filter_len - 1);

    p1.Init(kmax, 1);
    p2.Init(src_len, src_len);
    p2_b.Init(src_len, 1);
    p3.Init(tgt_len, src_len);
    p3_b.Init(tgt_len, 1);
    tgt_bias.Init(num_tgt_words, 1);
  }

  void GetSentVector(const Mat& sent_mat, ACol* sent_vec) {
    AMat out11, out12, out21, out22, layer1_out;
    /* Layer 1 */
    int k = max(int(sent_mat.cols()/2), kmax);
    ConvolveLayer(sent_mat, f11, k, &out11);
    ConvolveLayer(sent_mat, f12, k, &out12);
    /* Layer 2 */
    layer1_out = out11 + out12;
    ConvolveLayer(layer1_out, f21, kmax, &out21);
    ConvolveLayer(layer1_out, f22, kmax, &out22);
    /* Add the results of the two parallel convolutions */
    *sent_vec = (out21 + out22) * p1.var;
  }

  template <typename T>
  void ConvolveLayer(const T& mat, const Filter& filter,
                     const int& k, AMat* res) {
    AMat convolved;
    convolve_wide(mat, filter.filter.var, &convolved);
    Max(convolved, k, res);
    *res += filter.bias.var.rowwise().replicate(res->cols());
    ElemwiseSigmoid(res);
  }

  void VecInContext(const Col& src_vec, const ACol& sent_vec, ACol* pred_vec) {
    /* Pass the src_word_vec through non-linearity */
    ACol src_non_linear_vec = Prod(p2.var, src_vec) + p2_b.var;
    ElemwiseSigmoid(&src_non_linear_vec);
    /* Add the processed src word vec with context_vec */
    *pred_vec = sent_vec + src_non_linear_vec;
  }

  void TransVecInContext(const Col& src_vec, const ACol& sent_vec,
                         ACol* pred_vec) {
    /* Pass the src_word_vec through non-linearity */
    ACol src_non_linear_vec = Prod(p2.var, src_vec) + p2_b.var;
    ElemwiseSigmoid(&src_non_linear_vec);
    /* Add the processed src word vec with context_vec & convert to tgt_len */
    *pred_vec = p3.var * (sent_vec + src_non_linear_vec) + p3_b.var;
  }

  void Baseline(const Col& src_vec, ACol* pred_vec) {
    ACol temp = Prod(p2.var, src_vec) + p2_b.var;
    ElemwiseSigmoid(&temp);
    *pred_vec = p3.var * temp + p3_b.var;
  }

  void UpdateParams() {
    f11.AdagradUpdate(RHO, EPSILON);
    f12.AdagradUpdate(RHO, EPSILON);
    f21.AdagradUpdate(RHO, EPSILON);
    f22.AdagradUpdate(RHO, EPSILON);
    p1.AdagradUpdate(RHO, EPSILON);
    p2.AdagradUpdate(RHO, EPSILON);
    p2_b.AdagradUpdate(RHO, EPSILON);
    p3.AdagradUpdate(RHO, EPSILON);
    p3_b.AdagradUpdate(RHO, EPSILON);
    tgt_bias.AdagradUpdate(RHO, EPSILON);
  }

  void WriteParamsToFile(const string& filename) {
    ofstream outfile(filename);
    if (outfile.is_open()) {
      outfile.precision(3);
      f11.WriteToFile(outfile);
      f12.WriteToFile(outfile);
      f21.WriteToFile(outfile);
      f22.WriteToFile(outfile);
      p1.WriteToFile(outfile);
      p2.WriteToFile(outfile);
      p2_b.WriteToFile(outfile);
      p3.WriteToFile(outfile);
      p3_b.WriteToFile(outfile);
      tgt_bias.WriteToFile(outfile);
      outfile.close();
      cerr << "\nWritten parameters to: " << filename;
    } else {
      cerr << "\nFailed to open " << filename;
    }
  }

  void InitParamsFromFile(const string& filename) {
    ifstream infile(filename);
    if (infile.is_open()) {
      f11.ReadFromFile(infile);
      f12.ReadFromFile(infile);
      f21.ReadFromFile(infile);
      f22.ReadFromFile(infile);
      p1.ReadFromFile(infile);
      p2.ReadFromFile(infile);
      p2_b.ReadFromFile(infile);
      p3.ReadFromFile(infile);
      p3_b.ReadFromFile(infile);
      tgt_bias.ReadFromFile(infile);
      infile.close();
      src_len = f11.filter.var.rows();
      tgt_len = p3.var.rows();
      kmax = p1.var.rows();
      filter_len = f11.filter.var.cols();
      cerr << "\nRead parameters from: " << filename;
    } else {
      cerr << "\nCould not open: " << filename;
    }
  }

};

/* The function that trains a neural network to convert word vectors
   from source language to target language while taking the context of
   the source word in account. The translation pairs are determined using
   word alignments.

   The parameters are self-explanatory.
*/
void Train(const string& p_corpus, const string& a_corpus,
           const string& outfilename, const int& num_iter,
           const int& noise_size, const int& filt_len, const int& kmax,
           const vector<Col>& src_word_vecs, const vector<Col>& tgt_word_vecs,
           const mapStrUnsigned& src_vocab, const mapStrUnsigned& tgt_vocab) {
  adept::Stack s;
  AliasSampler sampler;
  vector<double> noise_dist(tgt_vocab.size(), 0.0);
  Model model(filt_len, kmax, src_word_vecs[0].size(), tgt_word_vecs[0].size(),
              src_word_vecs.size(), tgt_word_vecs.size());
  SetUnigramBias(p_corpus, tgt_vocab, TARGET, &model.tgt_bias.var, &noise_dist);
  sampler.Init(noise_dist);
  for (unsigned i = 0; i < num_iter; ++i) {
    cerr << "\nIteration: " << i+1 << endl;
    ifstream p_file(p_corpus.c_str()), a_file(a_corpus.c_str());
    string p_line, a_line;
    vector<int> src_words, tgt_words;
    unsigned num_words = 0;
    adouble total_error = 0, nllh = 0, lnZ = 0;
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
        unsigned src_len = 0;
        for (unsigned j = 0; j < src.size(); ++j) {
          auto it = src_vocab.find(src[j]);
          if (it != src_vocab.end()) {
            src_words.push_back(it->second);
            src_len++;
          } else {
            src_words.push_back(-1); // word not in vocab
          }
        }
        if (src_len <= 1) continue;
        /* Make a sentence matrix */
        unsigned index = -1;
        Mat src_sent_mat(src_word_vecs[0].size(), src_len);
        for (unsigned j = 0; j < src_words.size(); ++j) {
          if (src_words[j] != -1)
            src_sent_mat.col(++index) = src_word_vecs[src_words[j]];
        }
        /* Target sentence */
        for (unsigned j=0; j<tgt.size(); ++j) {
          auto it = tgt_vocab.find(tgt[j]);
          if (it != tgt_vocab.end())
            tgt_words.push_back(it->second);
          else
            tgt_words.push_back(-1); // word not in vocab
        }
        /* Get sentence vector by convolution */
        ACol sent_vec;
        model.GetSentVector(src_sent_mat, &sent_vec);
        /* Read alignments and train */
        vector<string> src_tgt_pairs = split_line(a_line, ' ');
        for (unsigned j = 0; j < src_tgt_pairs.size(); ++j) {
          vector<string> index_pair = split_line(src_tgt_pairs[j], '-');
          unsigned src_ix = stoi(index_pair[0]), tgt_ix = stoi(index_pair[1]);
          unsigned tgt_word = tgt_words[tgt_ix], src_word = src_words[src_ix];
          /* If both words are in the cleaned sentences, train on this */
          if (tgt_word != -1 && src_word != -1) {
            Col tgt_vec = tgt_word_vecs[tgt_word];
            Col src_vec = src_word_vecs[src_word];
            ACol pred_tgt_vec;
            model.TransVecInContext(src_vec, sent_vec, &pred_tgt_vec);
            adouble error = LossNCE(pred_tgt_vec, tgt_vec, tgt_word,
                                    tgt_word_vecs, model.tgt_bias.var,
                                    noise_size, noise_dist, sampler);
            //auto val = NegLogProb(pred_tgt_vec, tgt_vec, tgt_word,
            //                      tgt_word_vecs, model.tgt_bias.var);
            //nllh += val.first;
            //lnZ += val.second;
            //adouble error = val.first;
            total_error += error;
            //total_error += val.first;
            /* Calcuate gradiet and update parameters */
            error.set_gradient(1.0);
            s.compute_adjoint();
            model.UpdateParams();
            s.new_recording();
            num_words += 1;
          }
        }
        cerr << num_words << "\r";
      }
      cerr << "\nError per word: "<< total_error/num_words;
      //cerr << "\nN LLH per word: "<< nllh/num_words;
      //cerr << "\nln(Z) per word: "<< lnZ/num_words;
      p_file.close();
      a_file.close();
      model.WriteParamsToFile(outfilename + "_i" + to_string(i+1));
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
*/
pair<bool, ACol>
Decode(const string& line, const vector<Col>& word_vecs,
       const mapStrUnsigned& vocab, Model* model) {
    /* Extracting words from sentences */
    vector<string> sent_and_index = split_line(line, '\t');
    vector<string> words = split_line(sent_and_index[1], ' ');
    int query_index = stoi(sent_and_index[0]);
    /* Read and clean sentence & make sentence matrix */
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
    /* Make a sentence matrix */
    Mat sent_mat(word_vecs[0].size(), clean_words.size());
    for (unsigned j = 0; j < clean_words.size(); ++j)
      sent_mat.col(j) = word_vecs[clean_words[j]];
    /* Read the alignment line */
    ACol sent_vec, pred_vec;
    model->GetSentVector(sent_mat, &sent_vec);
    model->VecInContext(sent_mat.col(new_query_index), sent_vec, &pred_vec);
    return make_pair(true, pred_vec);
}


int main(int argc, char **argv){
  if (argc == 10) {
    string parallel_corpus = argv[1];
    string align_corpus = argv[2];
    string src_vec_corpus = argv[3];
    string tgt_vec_corpus = argv[4];
    int filt_len = stoi(argv[5]);
    int kmax = stoi(argv[6]);
    int noise_size = stoi(argv[7]);
    int num_iter = stoi(argv[8]);
    string outfilename = argv[9];

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
    cerr << "Filter 1 length: " << filt_len << endl;
    cerr << "Filter 2 length: " << filt_len - 1 << endl;
    cerr << "k-max: " << kmax << endl;
    cerr << "Output vector length: " << tgt_len << endl;
    cerr << "----------------" << endl;

    Train(parallel_corpus, align_corpus, outfilename, num_iter, noise_size,
          filt_len, kmax, src_word_vecs, tgt_word_vecs, src_vocab, tgt_vocab);
  } else if (argc == 5) {
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
  } else {
    cerr << "Usage: "<< argv[0] << " parallel_corpus " << " alignment_corpus "
         << " src_vec_corpus " << " tgt_vec_corpus " << " filt_size " << "k_max"
         << " noise_size " << " num_iter " << " outfilename\n";
    cerr << "Recommended: " << argv[0] << " parallel_corpus "
         << " alignment_corpus " << " src_vec_corpus " << " tgt_vec_corpus "
         << " 3 5 100 2" << " out.txt\n\n";

    cerr << "Usage:" << argv[0] << "benchmark word_vecs_file"
         << "context_params_file out_vectors\n";
  }

  return 1;
}
