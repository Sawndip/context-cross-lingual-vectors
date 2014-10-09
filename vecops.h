#include "utils.h"

void ElemwiseTanh(Col*);
void ElemwiseTanh(ACol*);
void ElemwiseTanh(Row*);
void ElemwiseTanh(ARow*);

ARow ElemwiseProd(const ARow&, const Row&);
ARow Prod(const ARow&, const Mat&);
ARow ElemwiseDiff(const ARow&, const Row&);

ACol ElemwiseProd(const ACol&, const Col&);
ACol Prod(const ACol&, const Mat&);
ACol ElemwiseDiff(const ACol&, const Col&);
