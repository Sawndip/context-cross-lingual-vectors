#ifndef VECOPS_H
#define VECOPS_H

#include "utils.h"

void ElemwiseHardTanh(ACol*);
void ElemwiseTanh(Col*);
void ElemwiseTanh(ACol*);
void ElemwiseTanh(Row*);
void ElemwiseTanh(ARow*);

ACol ElemwiseProd(const ACol&, const Col&);
void ElemwiseProdSum(const ACol&, const Col&, ACol*);

adouble DotProdCol(const ACol&, const Col&);
adouble DotProdRow(const ARow&, const Row&);
adouble CosineSim(const ACol&, const Col&); 

void ProdSum(const AMat&, const Col&, ACol*);
ACol Prod(const ACol&, const Mat&);
ACol Prod(const AMat&, const Col&);
ACol ElemwiseDiff(const ACol&, const Col&);

#endif
