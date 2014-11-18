#include "utils.h"
#include "lexical.h"

using namespace std;
using namespace Eigen;

mapStrBool CONSIDER_CONTEXT;
mapStrBool CONSIDER_PRED;

bool ConsiderForPred(const string& a) {
  /* See if already computed */
  auto it = CONSIDER_PRED.find(a);
  if (it != CONSIDER_PRED.end())
    return it->second;
  bool pred_res = ConsiderForContext(a);
  if (pred_res == false) {
    CONSIDER_PRED[a] = false;
    return false;
  }
  /* False if its a stop word */
  string *f = std::find(STOP_WORDS, STOP_WORDS + NUM_STOP_WORDS, a);
  if (f != STOP_WORDS + NUM_STOP_WORDS) {
    CONSIDER_PRED[a] = false;
    return false;
  }
  CONSIDER_PRED[a] = true;
  return CONSIDER_PRED[a];
}

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

adouble LogAdd(adouble lna, adouble lnb) {
  if (lna == 1.0) return lnb;
  if (lnb == 1.0) return lna;

  adouble diff = lna - lnb;
  if (diff < 500.0) return log(exp(diff) + 1) + lnb;
  else return lna;
}

ARow TopKVals(ARow r, int k) {
  ARow res(k);
  /* If the row size <= k, put zeros on the extra columns */
  if (r.cols() <= k) {
    for (int i = 0; i < k; ++i) {
      if (i < r.cols()) res(0, i) = r(0, i);
      else res(0, i) = 0;
    }
    return res;
  }
  vector<adouble> temp;
  for (int i = 0; i < r.cols(); ++i)
    temp.push_back(r(0, i));
  nth_element(temp.begin(), temp.begin()+k-1, temp.end(),
              std::greater<adouble>());
  adouble kth_element = temp[k-1];
  /* Collect all elements >= kth_element */
  int index = 0;
  for (int i = 0; i < r.cols(); ++i) {
    if (index >= res.cols()) return res;
    if (r[i] >= kth_element) res(0, index++) = r[i];
  }
  return res;
}

void GetContext(const vector<unsigned>& words, const vector<string>& words_raw,
                unsigned tgt_word_ix, int window_size,
                mapIntUnsigned* t_context_words) {
  mapIntUnsigned& context_words = *t_context_words;
  context_words.clear();
  for (int i = -window_size; i <= window_size; ++i) {
    int word_index = i + tgt_word_ix;
    if (word_index >= 0 && word_index < words.size() &&
        word_index != tgt_word_ix && words[word_index] != -1 &&
        ConsiderForContext(words_raw[word_index])) {
        context_words[i] = words[word_index];
    }
  }
}

void GetUnigramDist(const string& p_file, const mapStrUnsigned& vocab,
                    const int& column, vector<double>* dist) {
  ifstream infile(p_file.c_str());
  if (infile.is_open()) {
    string line;
    while (getline(infile, line)) {
      vector<string> lines = split_line(line, '\t');
      line = lines[column];
      vector<string> words = split_line(line, ' ');
      for (unsigned j = 0; j < words.size(); ++j) {
        auto it = vocab.find(words[j]);
        if (it != vocab.end())
          (*dist)[it->second] += 1;
      }
    }
    /* Normalize the counts to get unigram distribution */
    double sum = accumulate(dist->begin(), dist->end(), 0);
    for (int i = 0; i < dist->size(); ++i)
      (*dist)[i] /= sum;
  } else {
    cerr <<"\nCould not open file: " << p_file;
    exit(0);
  }
}

/* Try splitting over all whitespaces not just space */
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

void ReadVecsFromFile(const string& p_corpus, const int& column,
                      const string& vec_file_name, mapStrUnsigned* t_vocab,
                      vector<Col>* word_vecs) {
  /* First read words from the training file */
  mapStrUnsigned train_vocab;
  ifstream infile(p_corpus.c_str());
  if (infile.is_open()) {
    string line;
    while (getline(infile, line)) {
      vector<string> lines = split_line(line, '\t');
      line = lines[column];
      vector<string> words = split_line(line, ' ');
      for (unsigned j = 0; j < words.size(); ++j) {
        auto it = train_vocab.find(words[j]);
        if (it != train_vocab.end())
          train_vocab[words[j]] += 1;
        else if (ConsiderForContext(words[j]))
          train_vocab[words[j]] = 1;
      }
    }
  } else {
    cerr << "Could not open " << p_corpus;
    exit(0);
  }

  /* Read vectors for only words in the train file vocab */
  ifstream vec_file(vec_file_name.c_str());
  mapStrUnsigned& vocab = *t_vocab;
  unsigned vocab_size = 0;
  if (vec_file.is_open()) {
    string line;
    vocab.clear();
    while (getline(vec_file, line)) {
      vector<string> vector_stuff = split_line(line, ' ');
      string word = vector_stuff[0];
      auto it = train_vocab.find(word);
      if (it != train_vocab.end()) {
        Col word_vec = Col::Zero(vector_stuff.size()-1);
        for (unsigned i = 0; i < word_vec.size(); ++i)
          word_vec(i, 0) = stof(vector_stuff[i+1]);
        vocab[word] = vocab_size++;
        word_vecs->push_back(word_vec);
      }
    }
    cerr << "Read: " << vec_file_name << endl;
    cerr << "Vocab length: " << word_vecs->size() << endl;
    cerr << "Vector length: " << (*word_vecs)[0].size() << endl << endl;
    vec_file.close();

    assert (word_vecs->size() == vocab.size());
  } else {
    cerr << "Could not open " << vec_file;
    exit(0);
  }
}
