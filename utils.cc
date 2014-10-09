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


void ReadVecsFromFile(string vec_file_name, mapStrUnsigned* t_vocab,
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

void ReadVecsFromFile(string vec_file_name, mapStrUnsigned* t_vocab,
                      vector<Row>* word_vecs) {
  ifstream vec_file(vec_file_name.c_str());
  mapStrUnsigned& vocab = *t_vocab;
  if (vec_file.is_open()) {
    string line;
    vocab.clear();
    while (getline(vec_file, line)) {
      vector<string> vector_stuff = split_line(line, ' ');
      string word = vector_stuff[0];
      Row word_vec = Row::Zero(vector_stuff.size()-1);
      //cerr << vector_stuff.size()-1 << " ";
      for (unsigned i = 0; i < word_vec.size(); ++i)
        word_vec(0, i) = stof(vector_stuff[i+1]);
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

