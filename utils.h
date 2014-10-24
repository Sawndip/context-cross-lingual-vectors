#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <functional>
#include <cmath>
#include <ctype.h>
#include <string>
#include <tr1/unordered_map>
#include <Eigen/Core>
#include <adept.h>

using adept::adouble;
using namespace std;
using namespace Eigen;

typedef Matrix<adouble, Dynamic, 1> ACol;
typedef Matrix<adouble, 1, Dynamic> ARow;
typedef Matrix<adouble, Dynamic, Dynamic> AMat;

typedef Matrix<double, Dynamic, 1> Col;
typedef Matrix<double, 1, Dynamic> Row;
typedef Matrix<double, Dynamic, Dynamic> Mat;

typedef std::tr1::unordered_map<string, unsigned> mapStrUnsigned;
typedef std::tr1::unordered_map<int, Col> mapIntCol;
typedef std::tr1::unordered_map<int, ACol> mapIntACol;
typedef std::tr1::unordered_map<int, Mat> mapIntMat;
typedef std::tr1::unordered_map<int, AMat> mapIntAMat;
typedef std::tr1::unordered_map<int, unsigned> mapIntUnsigned;
typedef std::tr1::unordered_map<unsigned, unsigned> mapUnsUns;

typedef std::tr1::unordered_map<string, unsigned> mapStrUnsigned;
typedef std::tr1::unordered_map<int, unsigned> mapIntUnsigned;
typedef std::tr1::unordered_map<string, bool> mapStrBool;

bool ConsiderForContext(const string&);
bool ConsiderForPred(const string&);
void GetContext(const vector<unsigned>&, const vector<string>&, unsigned,
                int, mapIntUnsigned*);

ARow TopKVals(ARow, int);

string normalize_word(string&);
vector<string> split_line(string&, char);

void random_acol_map(int, unsigned, mapIntACol*);
void random_amat_map(int, unsigned, unsigned, mapIntAMat*);
void random_col_map(int, unsigned, mapIntCol*);
void zero_acol_map(int, unsigned, mapIntACol*);
void zero_amat_map(int, unsigned, unsigned, mapIntAMat*);
void zero_mat_map(int, unsigned, unsigned, mapIntMat*);
void zero_col_map(int, unsigned, mapIntCol*);

void ReadVecsFromFile(const string&, mapStrUnsigned*, vector<Col>*);
void ReadVecsFromFile(const string&, mapStrUnsigned*, vector<Row>*);
void ReadEntropicWords(const string&, const mapStrUnsigned&, mapUnsUns*);

void WriteParamsToFile(const string&, const mapIntACol&, const AMat&);
void WriteParamsToFile(const string&, const AMat&, const AMat&, const AMat&); 
void ReadParamsFromFile(const string&, mapIntACol*, AMat*);

#endif
