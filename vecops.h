#ifndef VECOPS_H
#define VECOPS_H

#include "utils.h"

void convolve_narrow(const Mat&, const AMat&, AMat*);
void convolve_narrow(const AMat&, const AMat&, AMat*);
void convolve_wide(const Mat&, const AMat&, AMat*);
void convolve_wide(const AMat&, const AMat&, AMat*);

void max(const AMat&, const int&, AMat*);

void ElemwiseSigmoid(AMat*);
void ElemwiseSigmoid(ACol*);
void ElemwiseAbsRatio(ACol*);

void ElemwiseTanh(AMat*);
void ElemwiseHardTanh(AMat*);

void ElemwiseHardTanh(ACol*);
void ElemwiseTanh(Col*);
void ElemwiseTanh(ACol*);

void ElemwiseTanh(Row*);
void ElemwiseTanh(ARow*);

ACol ElemwiseProd(const ACol&, const Col&);
void ElemwiseProdSum(const ACol&, const Col&, ACol*);

adouble DotProdARow(const ARow&, const ARow&);
adouble DotProdCol(const ACol&, const Col&);
adouble DotProdRow(const ARow&, const Row&);
adouble CosineSim(const ACol&, const Col&); 

void ProdSum(const AMat&, const Col&, ACol*);
ACol Prod(const ACol&, const Mat&);
ACol Prod(const AMat&, const Col&);
ACol ElemwiseDiff(const ACol&, const Col&);

#endif
