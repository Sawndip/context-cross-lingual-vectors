#include "utils.h"

void ElemwiseTanh(Col* v) {
  for (unsigned i = 0; i < v->cols(); ++i)
    (*v)(i, 0) = tanh((*v)(i, 0));
}

void ElemwiseTanh(ACol* v) {
  for (unsigned i = 0; i < v->cols(); ++i)
    (*v)(i, 0) = tanh((*v)(i, 0));
}

void ElemwiseTanh(Row* v) {
  for (unsigned i = 0; i < v->cols(); ++i)
    (*v)(0, i) = tanh((*v)(0, i));
}

void ElemwiseTanh(ARow* v) {
  for (unsigned i = 0; i < v->cols(); ++i)
    (*v)(0, i) = tanh((*v)(0, i));
}

ARow ElemwiseProd(const ARow& ar, const Row& r) {
  ARow res = ARow::Zero(r.cols());
  for (unsigned i = 0; i < r.cols(); ++i)
    res(0, i) = r(0, i) * ar(0, i);
  return res;
}

ARow Prod(const ARow& r, const Mat& mat) {
  ARow res = ARow::Zero(r.cols());
  for (unsigned j = 0; j < mat.cols(); ++j)
    for (unsigned i = 0; i < mat.rows(); ++i)
      res(0, j) += r(0, j) * mat(i, j);
  return res;
}

ARow ElemwiseDiff(const ARow& ar, const Row& r) {
  ARow res = ARow::Zero(r.cols());
  for (unsigned i = 0; i < r.cols(); ++i)
    res(0, i) = ar(0, i) - r(0, i);
  return res;
}

void ElemwiseProdSum(const ACol& ac, const Col& c, ACol* res) {
  for (unsigned i = 0; i < c.rows(); ++i)
    (*res)(i, 0) += c(i, 0) * ac(i, 0);
}


ACol ElemwiseProd(const ACol& ac, const Col& c) {
  ACol res = ACol::Zero(c.rows());
  for (unsigned i = 0; i < c.rows(); ++i)
    res(i, 0) = c(i, 0) * ac(i, 0);
  return res;
}

ACol Prod(const ACol& c, const Mat& mat) {
  ACol res = ACol::Zero(c.rows());
  for (unsigned i = 0; i < mat.rows(); ++i)
    for (unsigned j = 0; j < mat.cols(); ++j)
      res(i, 0) += mat(i, j) * c(j, 0);
  return res;
}

ACol ElemwiseDiff(const ACol& ac, const Col& c) {
  ACol res = ACol::Zero(c.rows());
  for (unsigned i = 0; i < c.rows(); ++i)
    res(i, 0) = ac(i, 0) - c(i, 0);
  return res;
}
