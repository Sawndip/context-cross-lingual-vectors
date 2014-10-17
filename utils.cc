#include "utils.h"
#include "lexical.h"

using namespace std;
using namespace Eigen;

mapStrBool CONSIDER_STRING;

bool ConsiderString(const string& a) {
  /* See if already computed */
  auto it = CONSIDER_STRING.find(a);
  if (it != CONSIDER_STRING.end())
    return it->second;
  /* False if its a punctuation */
  if (a.length() == 1) {
    if (a.at(0) >= 33 && a.at(0) <= 64) {
      CONSIDER_STRING[a] = false;
      return false;
    }
  }
  /* False if its a stop word */
  string *f = std::find(STOP_WORDS, STOP_WORDS + NUM_STOP_WORDS, a);
  if (f != STOP_WORDS + NUM_STOP_WORDS) {
    CONSIDER_STRING[a] = false;
    return false;
  }
  /* False if it contains a digit */
  CONSIDER_STRING[a] = !(any_of(a.begin(), a.end(), ::isdigit));
  return CONSIDER_STRING[a];
}

void GetContext(const vector<unsigned>& words, const vector<string>& words_raw,
                unsigned tgt_word_ix,
                int window_size, mapIntUnsigned* t_context_words) {
  mapIntUnsigned& context_words = *t_context_words;
  context_words.clear();
  for (int i = -window_size; i <= window_size; ++i) {
    int word_index = i + tgt_word_ix;
    if (word_index >= 0 && word_index < words.size() &&
        word_index != tgt_word_ix && ConsiderString(words_raw[word_index])) {
      if (words[word_index] != -1)  // word not in vector vocab
        context_words[i] = words[word_index];
    }
  }
}

void AdadeltaMatUpdate(const double& rho, const double& epsilon,
                       AMat* mat, Mat* mat_delta, Mat* mat_grad) {
  for (unsigned i = 0; i < mat->rows(); ++i) {
    for (unsigned j = 0; j < mat->cols(); ++j) {
      double g = (*mat)(i, j).get_gradient();
      double accum_g = rho * (*mat_grad)(i, j) + (1 - rho) * g * g;
      double del = sqrt((*mat_delta)(i, j) + epsilon);
      del /= sqrt(accum_g + epsilon);
      del *= g;
      (*mat)(i, j) -= del;  // Update the variable
      /* Update memory */
      (*mat_grad)(i, j) = accum_g;
      (*mat_delta)(i, j) = rho * (*mat_delta)(i, j);
      (*mat_delta)(i, j) += (1 - rho) * del * del;
    }
  }
}

void AdadeltaUpdate(const double& rho, const double& epsilon,
                    adouble* d, double* d_delta, double* d_grad) {
  double g = d->get_gradient();
  double accum_g = rho * (*d_grad) + (1 - rho) * g * g;
  double del = sqrt((*d_delta) + epsilon);
  del /= sqrt(accum_g + epsilon);
  del *= g;
  *d -= del;  // Update the variable
  /* Update memory */
  *d_grad = accum_g;
  *d_delta = rho * (*d_delta);
  *d_delta += (1 - rho) * del * del;
}

/* Try splitting over all whitespaces not just space */
vector<string> split_line(string& line, char delim) {
  vector<string> words;
  stringstream ss(line);
  string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty())
      words.push_back(item);
  }
  return words;
}

void random_amat_map(int context_len, unsigned rows, unsigned cols,
                     mapIntAMat *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = (0.6 / sqrt(rows*cols)) * AMat::Random(rows, cols);
}

void random_acol_map(int context_len, unsigned vec_len,
                     mapIntACol *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = (0.6 / sqrt(vec_len)) * ACol::Random(vec_len);
}

void random_col_map(int context_len, unsigned vec_len,
                    mapIntCol *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = (0.6 / sqrt(vec_len)) * Col::Random(vec_len);
}

void zero_amat_map(int context_len, unsigned rows, unsigned cols,
                   mapIntAMat *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = AMat::Zero(rows, cols);
}

void zero_mat_map(int context_len, unsigned rows, unsigned cols,
                  mapIntMat *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = Mat::Zero(rows, cols);
}


void zero_acol_map(int context_len, unsigned vec_len,
                   mapIntACol *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = ACol::Zero(vec_len);
}

void zero_col_map(int context_len, unsigned vec_len,
                  mapIntCol *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = Col::Zero(vec_len);
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

/* Writes the matrix to the file in row-major order
   Writes the vectors to the file in column-major order */
void WriteParamsToFile(const string& filename, const mapIntACol& vecs,
                       const AMat& mat) {
  ofstream outfile(filename);
  if (outfile.is_open()) {
    outfile.precision(3);
    outfile << mat.rows() << " " << mat.cols() << " ";
    for (unsigned i = 0; i < mat.rows(); ++i) {
      for(unsigned j = 0; j < mat.cols(); ++j) {
        outfile << mat(i, j) << " ";
      }
    }
    outfile << endl;

    for (auto it = vecs.begin(); it != vecs.end(); ++it) {
      auto vec = it->second;
      outfile << it->first << " " << vec.rows() << " ";
      for (unsigned i = 0; i < vec.rows(); ++i)
        outfile << vec(i, 0) << " ";
      outfile << endl;
    }
    outfile.close();
    cerr << "Written parameters to: " << filename << endl;
  } else {
    cerr << "Could not open: " << filename << endl;
  }
}

/* All parameters are matrices which are written in row-major order */
void WriteParamsToFile(const string& filename, const AMat& self_mat,
                       const AMat& context_mat, const AMat& convert_mat) {
  ofstream outfile(filename);
  if (outfile.is_open()) {
    outfile.precision(3);
    outfile << self_mat.rows() << " " << self_mat.cols() << " ";
    for (unsigned i = 0; i < self_mat.rows(); ++i) {
      for(unsigned j = 0; j < self_mat.cols(); ++j) {
        outfile << self_mat(i, j) << " ";
      }
    }
    outfile << endl;

    outfile << context_mat.rows() << " " << context_mat.cols() << " ";
    for (unsigned i = 0; i < context_mat.rows(); ++i) {
      for(unsigned j = 0; j < context_mat.cols(); ++j) {
        outfile << context_mat(i, j) << " ";
      }
    }
    outfile << endl;

    outfile << convert_mat.rows() << " " << convert_mat.cols() << " ";
    for (unsigned i = 0; i < convert_mat.rows(); ++i) {
      for(unsigned j = 0; j < convert_mat.cols(); ++j) {
        outfile << convert_mat(i, j) << " ";
      }
    }
    outfile << endl;
    outfile.close();
    cerr << "Written parameters to: " << filename << endl;
  } else {
    cerr << "Could not open: " << filename << endl;
  }
}

void ReadEntropicWords(const string& filename, const mapStrUnsigned& vocab,
                       mapUnsUns* res) {
  ifstream infile(filename);
  if (infile.is_open()) {
    string line;
    while (getline(infile, line)) {
      auto it = vocab.find(line);
      if (it != vocab.end())
        (*res)[it->second] = 0;
    }
    infile.close();
    cerr << "Read entropic words from: " << filename << endl;
  } else {
    cerr << "Could not open: " << filename << endl;
  }
}

/* Reads the elements in the matrix in row-major order,
   Reads the elements in the vector in column-major order */
void ReadParamsFromFile(const string& filename, mapIntACol* vecs, AMat* mat) {
  ifstream infile(filename);
  if (infile.is_open()) {
    string line;
    getline(infile, line);
    vector<string> mat_stuff = split_line(line, ' ');
    int row = stoi(mat_stuff[0]), col = stoi(mat_stuff[1]);
    (*mat) = AMat::Zero(row, col);
    for (int i = 2; i < mat_stuff.size(); ++i)
      (*mat)((i-2)/col, (i-2)%col) = stod(mat_stuff[i]);

    while (getline(infile, line)) {
      vector<string> vec_stuff = split_line(line, ' ');
      int index = stoi(vec_stuff[0]), vec_len = stoi(vec_stuff[1]);
      (*vecs)[index] = ACol::Zero(vec_len);
      for (int i = 2; i < vec_stuff.size(); ++i)
        (*vecs)[index](i-2, 0) = stod(vec_stuff[i]);
    }
    infile.close();
    cerr << "Read parameters from: " << filename << endl;
  } else {
    cerr << "Could not open: " << filename << endl;
  }
}
