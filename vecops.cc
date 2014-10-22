#include "utils.h"

adouble DotProdRow(const ARow& a, const Row& b) {
  adouble sum = 0;
  for (unsigned i = 0; i < a.cols(); ++i)
    sum += a(0, i) * b(0, i);
  return sum;
}

adouble DotProdARow(const ARow& a, const ARow& b) {
  adouble sum = 0;
  for (unsigned i = 0; i < a.cols(); ++i)
    sum += a(0, i) * b(0, i);
  return sum;
}

adouble DotProdCol(const ACol& a, const Col& b) {
  adouble sum = 0;
  for (unsigned i = 0; i < a.rows(); ++i)
    sum += a(i, 0) * b(i, 0);
  return sum;
}

void convolve_narrow(const Mat& mat, const AMat& filter, AMat* res) {
  /* Sentence should be greater than filter length */
  if (mat.rows() != filter.rows() || mat.cols() < filter.cols()) {
    cerr << "Incompatible matrix dimensions." << endl;
    cerr << "Matrix: " << mat.rows() << " " << mat.cols() << endl;
    cerr << "Filter: " << filter.rows() << " " << filter.cols() << endl;
    exit(0);
  }
  unsigned slice_len = filter.cols();
  (*res) = AMat::Zero(mat.rows(), mat.cols() - slice_len + 1);
  for (unsigned i = 0; i < res->rows(); ++i) {
    for (unsigned j = 0; j < res->cols(); ++j) {
      (*res)(i, j) = DotProdRow(filter.row(i), mat.block(i, j, 1, slice_len));
    }
  }
}

void convolve_wide(const Mat& mat, const AMat& filter, AMat* res) {
  /* Append extra zero vectors at the end and beginning of sentence
     for wide convolution */
  Mat zeros = Mat::Zero(mat.rows(), filter.cols() - 1);
  Mat new_sent(mat.rows(), mat.cols() + 2 * zeros.cols());
  new_sent << zeros, mat, zeros;
  convolve_narrow(new_sent, filter, res);
}

void convolve_narrow(const AMat& mat, const AMat& filter, AMat* res) {
  /* Sentence should be greater than filter length */
  if (mat.rows() != filter.rows() || mat.cols() < filter.cols()) {
    cerr << "Incompatible matrix dimensions." << endl;
    cerr << "Matrix: " << mat.rows() << " " << mat.cols() << endl;
    cerr << "Filter: " << filter.rows() << " " << filter.cols() << endl;
    exit(0);
  }
  unsigned slice_len = filter.cols();
  (*res) = AMat::Zero(mat.rows(), mat.cols() - slice_len + 1);
  for (unsigned i = 0; i < res->rows(); ++i) {
    for (unsigned j = 0; j < res->cols(); ++j)
      (*res)(i, j) = DotProdARow(filter.row(i), mat.block(i, j, 1, slice_len));
  }
}

void convolve_wide(const AMat& mat, const AMat& filter, AMat* res) {
  /* Append extra zero vectors at the end and beginning of sentence
     for wide convolution */
  AMat zeros = AMat::Zero(mat.rows(), filter.cols() - 1);
  AMat new_sent(mat.rows(), mat.cols() + 2 * zeros.cols());
  new_sent << zeros, mat, zeros;
  convolve_narrow(new_sent, filter, res);
}

void Max(const AMat& mat, const int& k, AMat* res) {
  if (k == 1) {
    (*res) = mat.rowwise().maxCoeff();
  } else {
    (*res) = AMat(mat.rows(), k);
    for (unsigned i = 0; i < mat.rows(); ++i)
      res->row(i) = TopKVals(mat.row(i), k);
  }
}

void ElemwiseSigmoid(AMat* m) {
  for (unsigned i = 0; i < m->rows(); ++i) {
    for (unsigned j = 0; j < m->cols(); ++j)
      (*m)(i, j) = 1/(1 + exp(-(*m)(i, j)));
  }
}


void ElemwiseSigmoid(ACol* m) {
  for (unsigned i = 0; i < m->rows(); ++i)
      (*m)(i, 0) /= (1 + exp(-(*m)(i, 0)));
}

void ElemwiseAbsRatio(ACol* m) {
  for (unsigned i = 0; i < m->rows(); ++i)
      (*m)(i, 0) /= (1 + abs((*m)(i, 0)));
}


void ElemwiseTanh(AMat* m) {
  for (unsigned i = 0; i < m->rows(); ++i) {
    for (unsigned j = 0; j < m->cols(); ++j)
      (*m)(i, j) = tanh((*m)(i, j));
  }
}

void ElemwiseHardTanh(AMat* m) {
  for (unsigned i = 0; i < m->rows(); ++i) {
    for (unsigned j = 0; j < m->cols(); ++j) {
      if ((*m)(i, j) < -1)
        (*m)(i, j) = -1;
      else if ((*m)(i, j) > 1)
        (*m)(i, j) = 1;
    }
  }
}

void ElemwiseHardTanh(ACol* v) {
  for (unsigned i = 0; i < v->cols(); ++i)
    if ((*v)(i, 0) < -1)
      (*v)(i, 0) = -1;
    else if ((*v)(i, 0) > 1)
      (*v)(i, 0) = 1;
}

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

adouble CosineSim(const ACol& ac, const Col& c) {
  return DotProdCol(ac, c)/sqrt(ac.squaredNorm() * c.squaredNorm());
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
