#ifndef TOY_TOY_H
#define TOY_TOY_H

// IWYU pragma: begin_keep
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
// IWYU pragma: end_keep

#include "ToyDialect.h.inc"
#define GET_OP_CLASSES
#include "ToyOps.h.inc"
#define GET_TYPEDEF_CLASSES
#include "ToyTypes.h.inc"

#endif // TOY_TOY_H
