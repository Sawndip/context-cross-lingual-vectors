#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <functional>
#include <cmath>
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
typedef std::tr1::unordered_map<int, unsigned> mapIntUnsigned;

typedef std::tr1::unordered_map<string, unsigned> mapStrUnsigned;
typedef std::tr1::unordered_map<int, Row> mapIntRow;
typedef std::tr1::unordered_map<int, ARow> mapIntARow;
typedef std::tr1::unordered_map<int, unsigned> mapIntUnsigned;

string normalize_word(string&);
vector<string> split_line(string&, char);

void random_arow_map(int, unsigned, mapIntARow*);
void random_row_map(int, unsigned, mapIntRow*);
void zero_arow_map(int, unsigned, mapIntARow*);
void zero_row_map(int, unsigned, mapIntRow*);

void random_acol_map(int, unsigned, mapIntACol*);
void random_col_map(int, unsigned, mapIntCol*);
void zero_acol_map(int, unsigned, mapIntACol*);
void zero_col_map(int, unsigned, mapIntCol*);

void ReadVecsFromFile(string, mapStrUnsigned*, vector<Col>*);
void ReadVecsFromFile(string, mapStrUnsigned*, vector<ARow>*);
void ReadVecsFromFile(string, mapStrUnsigned*, vector<Row>*);
