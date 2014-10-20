#include "utils.h"

void ElemwiseTanh(Col* v) {
  for (unsigned i = 0; i < v->cols(); ++i)
    (*v)(i, 0) = tanh((*v)(i, 0));
}

void ElemwiseTanh(ACol* v) {
  for (unsigned i = 0; i < v->cols(); ++i)
    (*v)(i, 0) = tanh((*v)(i, 0));
}

void ElemwiseProdSum(const ACol& ac, const Col& c, ACol* res) {
  for (unsigned i = 0; i < c.rows(); ++i)
    (*res)(i, 0) += c(i, 0) * ac(i, 0);
}

adouble DotProd(const ARow& a, const Row& b) {
  adouble sum = 0;
  for (unsigned i = 0; i < a.cols(); ++i)
    sum += a(0, i) * b(0, i);
  return sum;
}


ACol Prod(const AMat& mat, const Col& c) {
  ACol res = ACol::Zero(mat.rows());
  for (unsigned i = 0; i < mat.rows(); ++i) {
    for (unsigned j = 0; j < mat.cols(); ++j)
      res(i, 0) += mat(i, j) * c(j, 0);
  }
  return res;
}

ACol Prod(const ACol& c, const Mat& mat) {
  ACol res = ACol::Zero(c.rows());
  for (unsigned i = 0; i < mat.rows(); ++i) {
    for (unsigned j = 0; j < mat.cols(); ++j)
      res(i, 0) += mat(i, j) * c(j, 0);
  }
  return res;
}

void ProdSum(const AMat& am, const Col& c, ACol* res) {
  *res += Prod(am, c);
}


ACol ElemwiseProd(const ACol& ac, const Col& c) {
  ACol res = ACol::Zero(c.rows());
  for (unsigned i = 0; i < c.rows(); ++i)
    res(i, 0) = c(i, 0) * ac(i, 0);
  return res;
}

ACol ElemwiseDiff(const ACol& ac, const Col& c) {
  ACol res = ACol::Zero(c.rows());
  for (unsigned i = 0; i < c.rows(); ++i)
    res(i, 0) = ac(i, 0) - c(i, 0);
  return res;
}
