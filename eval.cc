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

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, 1> Col;
typedef Matrix<double, 1, Dynamic> Row;
typedef Matrix<double, Dynamic, Dynamic> Mat;

typedef std::tr1::unordered_map<string, unsigned> mapStrUnsigned;
typedef std::tr1::unordered_map<int, Col> mapIntCol;
typedef std::tr1::unordered_map<int, Mat> mapIntMat;
typedef std::tr1::unordered_map<int, unsigned> mapIntUnsigned;
typedef std::tr1::unordered_map<unsigned, unsigned> mapUnsUns;
typedef std::tr1::unordered_map<unsigned, double> mapUnsDouble;

typedef std::tr1::unordered_map<string, unsigned> mapStrUnsigned;
typedef std::tr1::unordered_map<int, unsigned> mapIntUnsigned;
typedef std::tr1::unordered_map<string, bool> mapStrBool;

void ElemwiseSigmoid(Col* m) {
  for (unsigned i = 0; i < m->rows(); ++i)
      (*m)(i, 0) /= (1 + exp(-(*m)(i, 0)));
}

void ElemwiseSigmoid(Mat* m) {
  for (unsigned i = 0; i < m->rows(); ++i) {
    for (unsigned j = 0; j < m->cols(); ++j)
      (*m)(i, j) = 1/(1 + exp(-(*m)(i, j)));
  }
}

double LogAdd(double lna, double lnb) {
  if (lna == 1.0) return lnb;
  if (lnb == 1.0) return lna;

  double diff = lna - lnb;
  if (diff < 500.0) return log(exp(diff) + 1) + lnb;
  else return lna;
}

vector<string> split_line(const string& line, char delim) {
  vector<string> words;
  stringstream ss(line);
  string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty())
      words.push_back(item);
  }
  return words;
}

Row TopKVals(Row r, int k) {
  Row res(k);
  /* If the row size <= k, put zeros on the extra columns */
  if (r.cols() <= k) {
    for (int i = 0; i < k; ++i) {
      if (i < r.cols()) res(0, i) = r(0, i);
      else res(0, i) = 0;
    }
    return res;
  }
  vector<double> temp;
  for (int i = 0; i < r.cols(); ++i)
    temp.push_back(r(0, i));
  nth_element(temp.begin(), temp.begin()+k-1, temp.end(),
              std::greater<double>());
  double kth_element = temp[k-1];
  /* Collect all elements >= kth_element */
  int index = 0;
  for (int i = 0; i < r.cols(); ++i) {
    if (index >= res.cols()) return res;
    if (r[i] >= kth_element) res(0, index++) = r[i];
  }
  return res;
}


void Max(const Mat& mat, const int& k, Mat* res) {
  if (k == 1) {
    (*res) = mat.rowwise().maxCoeff();
  } else {
    (*res) = Mat(mat.rows(), k);
    for (unsigned i = 0; i < mat.rows(); ++i)
      res->row(i) = TopKVals(mat.row(i), k);
  }
}

void convolve_narrow(const Mat& mat, const Mat& filter, Mat* res) {
  /* Sentence should be greater than filter length */
  unsigned slice_len = filter.cols();
  (*res) = Mat::Zero(mat.rows(), mat.cols() - slice_len + 1);
  for (unsigned i = 0; i < res->rows(); ++i) {
    for (unsigned j = 0; j < res->cols(); ++j) {
      Row r = mat.block(i, j, 1, slice_len);
      (*res)(i, j) = r.dot(filter.row(i));
    }
  }
}

void convolve_wide(const Mat& mat, const Mat& filter, Mat* res) {
  /* Append extra zero vectors at the end and beginning of sentence
     for wide convolution */
  Mat zeros = Mat::Zero(mat.rows(), filter.cols() - 1);
  Mat new_sent(mat.rows(), mat.cols() + 2 * zeros.cols());
  new_sent << zeros, mat, zeros;
  convolve_narrow(new_sent, filter, res);
}

mapStrBool CONSIDER_CONTEXT;
bool ConsiderForContext(const string& a) {
  /* See if already computed */
  auto it = CONSIDER_CONTEXT.find(a);
  if (it != CONSIDER_CONTEXT.end())
    return it->second;
  /* False if its a punctuation */
  if (a.length() == 1) {
    if (a.at(0) >= 33 && a.at(0) <= 64) {
      CONSIDER_CONTEXT[a] = false;
      return false;
    }
  }
  /* False if it contains a digit */
  CONSIDER_CONTEXT[a] = !(any_of(a.begin(), a.end(), ::isdigit));
  return CONSIDER_CONTEXT[a];
}

double NegLogProb(const Col& hidden, const Col& tgt_vec,
                  const unsigned& tgt_word, const vector<Col>& tgt_vecs,
                  const Col& tgt_bias) {
  double lp = 0.0;
  double word_score = hidden.dot(tgt_vec) + tgt_bias[tgt_word];
  // Sum over the vocab here
  for (int i = 0; i < tgt_vecs.size(); ++i) {
    double score = hidden.dot(tgt_vecs[i]) + tgt_bias[i];
    lp = LogAdd(lp, score);
  }
  lp = word_score - lp;
  return -1.0 * lp;
}

void ReadFromFile(ifstream& in, Col* var) {
  string line;
  getline(in, line);
  vector<string> data = split_line(line, ' ');
  int rows = stoi(data[0]), cols = stoi(data[1]);
  *var = Col::Zero(rows);
  for (int i = 2; i < data.size(); ++i)
    (*var)((i-2)%cols) = stod(data[i]);
}

void ReadFromFile(ifstream& in, Mat* var) {
  string line;
  getline(in, line);
  vector<string> data = split_line(line, ' ');
  int rows = stoi(data[0]), cols = stoi(data[1]);
  *var = Mat::Zero(rows, cols);
  for (int i = 2; i < data.size(); ++i)
    (*var)((i-2)/cols, (i-2)%cols) = stod(data[i]);
}

void ReadVecsFromFile(const string& vec_file_name, mapStrUnsigned* t_vocab,
                      vector<Col>* word_vecs) {
  ifstream vec_file(vec_file_name.c_str());
  mapStrUnsigned& vocab = *t_vocab;
  if (vec_file.is_open()) {
    string line;
    vocab.clear();
    while (getline(vec_file, line)) {
      vector<string> vector_stuff = split_line(line, ' ');
      string word = vector_stuff[0];
      Col word_vec = Col::Zero(vector_stuff.size()-1);
      for (unsigned i = 0; i < word_vec.size(); ++i)
        word_vec(i, 0) = stof(vector_stuff[i+1]);
      vocab[word] = vocab.size();
      word_vecs->push_back(word_vec);
    }
    cerr << "Read: " << vec_file_name << endl;
    cerr << "Vocab length: " << word_vecs->size() << endl;
    cerr << "Vector length: " << (*word_vecs)[0].size() << endl << endl;
    vec_file.close();
  } else {
    cerr << "Could not open " << vec_file;
    exit(0);
  }
}

/* Main class definition that learns the word vectors */
class Model {

 public:
  /* The parameters of the model */
  Mat f11, f12, f21, f22;  // Convolution
  Col f11_b, f12_b, f21_b, f22_b;
  Mat p1, p2, p3;  // Post-convolution
  Col p2_b, p3_b, tgt_bias;  // Post-convolution
  int src_len, tgt_len, filter_len, kmax;
  
  Model() {}

  void Convolve(const Mat& sent_mat, Mat* res) {
    Mat out11, out12, out21, out22, layer1_out;
    /* Layer 1 */
    int k = max(int(sent_mat.cols()/2), kmax);
    ConvolveLayer(sent_mat, f11, f11_b, k, &out11);
    ConvolveLayer(sent_mat, f12, f12_b, k, &out12);
    /* Layer 2 */
    layer1_out = out11 + out12;
    ConvolveLayer(layer1_out, f21, f21_b, kmax, &out21);
    ConvolveLayer(layer1_out, f22, f22_b, kmax, &out22);
    /* Add the results of the two parallel convolutions */
    *res = out21 + out22;  
  }

  void ConvolveLayer(const Mat& mat, const Mat& filter, const Col& filt_bias,
                     const int& k, Mat* res) {
    Mat convolved;
    convolve_wide(mat, filter, &convolved);
    Max(convolved, k, res);
    *res += filt_bias.rowwise().replicate(res->cols());
    ElemwiseSigmoid(res);
  }

  void VecInContext(const Col& src_vec, const Col& sent_vec, Col* pred_vec) {
    /* Pass the src_word_vec through non-linearity */
    Col src_non_linear_vec = p2 * src_vec + p2_b;
    ElemwiseSigmoid(&src_non_linear_vec);
    /* Add the processed src word vec with context_vec */
    *pred_vec = sent_vec + src_non_linear_vec;
  }

  void TransVecInContext(const Col& src_vec, const Col& sent_vec,
                         Col* pred_vec) {
    /* Pass the src_word_vec through non-linearity */
    Col src_non_linear_vec = p2 *src_vec + p2_b;
    ElemwiseSigmoid(&src_non_linear_vec);
    /* Add the processed src word vec with context_vec & convert to tgt_len */
    *pred_vec = p3 * (sent_vec + src_non_linear_vec) + p3_b;
  }

  void InitParamsFromFile(const string& filename) {
    ifstream infile(filename);
    if (infile.is_open()) {
      ReadFromFile(infile, &f11);
      ReadFromFile(infile, &f11_b);
      ReadFromFile(infile, &f12);
      ReadFromFile(infile, &f12_b);
      ReadFromFile(infile, &f21);
      ReadFromFile(infile, &f21_b);
      ReadFromFile(infile, &f22);
      ReadFromFile(infile, &f22_b);
      ReadFromFile(infile, &p1);
      ReadFromFile(infile, &p2);
      ReadFromFile(infile, &p2_b);
      ReadFromFile(infile, &p3);
      ReadFromFile(infile, &p3_b);
      ReadFromFile(infile, &tgt_bias);
      infile.close();
      src_len = f11.rows();
      tgt_len = p3.rows();
      kmax = p1.rows();
      filter_len = f11.cols();
      cerr << "Read parameters from: " << filename << endl;
    } else {
      cerr << "Could not open: " << filename << endl;
    }
  }

};

void Evaluate(const string& p_corpus, const string& a_corpus,
           const string& param_file,
           const vector<Col>& src_word_vecs, const vector<Col>& tgt_word_vecs,
           const mapStrUnsigned& src_vocab, const mapStrUnsigned& tgt_vocab) {
  Model model;
  model.InitParamsFromFile(param_file);
  ifstream p_file(p_corpus.c_str()), a_file(a_corpus.c_str());
  string p_line, a_line;
  vector<int> src_words, tgt_words;
  unsigned num_words = 0;
  double total_error = 0;
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
        auto it = src_vocab.find(src[j]);
        if (it != src_vocab.end() && ConsiderForContext(src[j])) {
          src_words.push_back(it->second);
          old_to_new[j] = index++;
        }
      }
      /* Make a sentence matrix */
      Mat src_sent_mat(src_word_vecs[0].size(), src_words.size());
      for (unsigned j = 0; j < src_words.size(); ++j)
        src_sent_mat.col(j) = src_word_vecs[src_words[j]];
      /* Target sentence */
      for (unsigned j=0; j<tgt.size(); ++j) {
        auto it = tgt_vocab.find(tgt[j]);
        if (it != tgt_vocab.end())
          tgt_words.push_back(it->second);
        else
          tgt_words.push_back(-1); // word not in vocab
      }
      Mat sent_convolved;
      Col sent_vec;
      vector<string> src_tgt_pairs = split_line(a_line, ' ');
      model.Convolve(src_sent_mat, &sent_convolved);
      sent_vec = sent_convolved * model.p1;
      unsigned j;
      #pragma omp parallel num_threads(5) shared(total_error, num_words)
      #pragma omp for nowait private(j)
      for (j = 0; j < src_tgt_pairs.size(); ++j) {
        vector<string> index_pair = split_line(src_tgt_pairs[j], '-');
        unsigned src_ix = stoi(index_pair[0]), tgt_ix = stoi(index_pair[1]);
        unsigned tgt_word = tgt_words[tgt_ix];
        /* If both words are in the cleaned sentences, train on this */
        if (tgt_words[tgt_ix] != -1 &&
            old_to_new.find(src_ix) != old_to_new.end()) {
          src_ix = old_to_new[src_ix];
          Col tgt_vec = tgt_word_vecs[tgt_word];
          Col src_vec = src_word_vecs[src_words[src_ix]];
          Col pred_tgt_vec;
          model.TransVecInContext(src_vec, sent_vec, &pred_tgt_vec);
          double error = NegLogProb(pred_tgt_vec, tgt_vec, tgt_word,
                                     tgt_word_vecs, model.tgt_bias);
          #pragma omp critical
          {
            total_error += error;
            num_words += 1;
          }
        }
      }
      cerr << num_words << "\r"; 
    } 
    cerr << "\nNeg lp per word: "<< total_error/num_words << "\n";
    cerr << "\nPerplexity: "<< exp(total_error/num_words) << "\n";
    p_file.close();
    a_file.close();
  } else {
    cerr << "\nUnable to open file\n";
    return;
  }
}

int main(int argc, char **argv){
  string parallel_corpus = argv[1];
  string align_corpus = argv[2];
  string src_vec_corpus = argv[3];
  string tgt_vec_corpus = argv[4];
  string param_file = argv[5];

  mapStrUnsigned src_vocab, tgt_vocab;
  vector<Col> src_word_vecs, tgt_word_vecs;
  ReadVecsFromFile(src_vec_corpus, &src_vocab, &src_word_vecs);
  ReadVecsFromFile(tgt_vec_corpus, &tgt_vocab, &tgt_word_vecs);
  int src_len = src_word_vecs[0].size();
  int tgt_len = tgt_word_vecs[0].size();

  Evaluate(parallel_corpus, align_corpus, param_file, 
           src_word_vecs, tgt_word_vecs, src_vocab, tgt_vocab);
  return 1;
}
