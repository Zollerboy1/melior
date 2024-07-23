#ifndef TOY_TOY_DIALECT_H
#define TOY_TOY_DIALECT_H

#include "dialects.h"

#include "mlir-c/IR.h" // IWYU pragma: keep

#ifdef __cplusplus
extern "C" {
#endif

MLIR_TOY_DECLARE_CAPI_DIALECT_REGISTRATION(Toy, toy);

#ifdef __cplusplus
}
#endif

#endif // TOY_TOY_DIALECT_H
