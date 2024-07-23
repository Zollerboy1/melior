#ifndef TOY_IR_H
#define TOY_IR_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
    struct MlirToy##name {                                                     \
        storage *ptr;                                                          \
    };                                                                         \
    typedef struct MlirToy##name MlirToy##name

DEFINE_C_API_STRUCT(OpBuilder, void);
DEFINE_C_API_STRUCT(OpBuilderListener, void);

#undef DEFINE_C_API_STRUCT

struct MlirToyOpBuilderBlockLocation {
    MlirRegion region;
    void * iterator;
};
typedef struct MlirToyOpBuilderBlockLocation MlirToyOpBuilderBlockLocation;

struct MlirToyOpBuilderInsertPoint {
    MlirBlock block;
    void * iterator;
};
typedef struct MlirToyOpBuilderInsertPoint MlirToyOpBuilderInsertPoint;

#ifdef __cplusplus
}
#endif

#endif // TOY_IR_H
