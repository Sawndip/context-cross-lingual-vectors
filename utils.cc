#include "utils.h"

using namespace std;
using namespace Eigen;

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

void random_acol_map(int context_len, unsigned vec_len,
                     mapIntACol *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = ACol::Random(vec_len);
}

void random_col_map(int context_len, unsigned vec_len,
                    mapIntCol *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = Col::Random(vec_len);
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

void random_arow_map(int context_len, unsigned vec_len,
                     mapIntARow *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = ARow::Random(vec_len);
}

void random_row_map(int context_len, unsigned vec_len,
                     mapIntRow *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = Row::Random(vec_len);
}

void zero_arow_map(int context_len, unsigned vec_len,
                   mapIntARow *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = ARow::Zero(vec_len);
}

void zero_row_map(int context_len, unsigned vec_len,
                  mapIntRow *result) {
  for (int i = -context_len; i <= context_len; ++i)
    (*result)[i] = Row::Zero(vec_len);
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
