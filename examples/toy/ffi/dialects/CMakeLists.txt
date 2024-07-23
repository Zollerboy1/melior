include_directories(${PROJECT_SOURCE_DIR}/dialects)
include_directories(${PROJECT_BINARY_DIR}/dialects)

set(LLVM_TARGET_DEFINITIONS toy.td)
mlir_tablegen(ToyOps.h.inc -gen-op-decls)
mlir_tablegen(ToyOps.cpp.inc -gen-op-defs)
mlir_tablegen(ToyTypes.h.inc -gen-typedef-decls -typedefs-dialect=toy)
mlir_tablegen(ToyTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=toy)
mlir_tablegen(ToyDialect.h.inc -gen-dialect-decls -dialect=toy)
mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs -dialect=toy)
add_public_tablegen_target(MLIRToyOpsIncGen)
add_dependencies(mlir-headers MLIRToyOpsIncGen)

# set(LLVM_TARGET_DEFINITIONS ToyPatterns.td)
# mlir_tablegen(ToyCanonicalize.cpp.inc -gen-rewriters)
# add_public_tablegen_target(MLIRToyCanonicalizationIncGen)

add_mlir_dialect_library(MLIRToyDialect
    toy.cpp
    LINK_LIBS PUBLIC)
