#include "toy.h"
#include "toy-dialect.h" // IWYU pragma: keep

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/CAPI/Registration.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "ToyDialect.cpp.inc"
#define GET_OP_CLASSES
#include "ToyOps.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "ToyTypes.cpp.inc"


static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    llvm::SMLoc operandsLoc = parser.getCurrentLocation();
    mlir::Type type;
    if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
        return mlir::failure();

    // If the type is a function type, it contains the input and result types of
    // this operation.
    if (mlir::FunctionType funcType = llvm::dyn_cast<mlir::FunctionType>(type)) {
        if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                                result.operands))
            return mlir::failure();
        result.addTypes(funcType.getResults());
        return mlir::success();
    }

    // Otherwise, the parsed type is the type of both operands and results.
    if (parser.resolveOperands(operands, type, result.operands))
        return mlir::failure();
    result.addTypes(type);
    return mlir::success();
}

static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
    printer << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    // If all of the types are the same, print the type directly.
    mlir::Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(),
                     [=](mlir::Type type) { return type == resultType; })) {
        printer << resultType;
        return;
    }

    // Otherwise, print a functional type.
    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}


void toy::ToyDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "ToyOps.cpp.inc"
      >();
}

void toy::ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
    auto dataType = mlir::RankedTensorType::get({}, builder.getF64Type());
    auto dataAttribute = mlir::DenseElementsAttr::get(dataType, value);
    toy::ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::ParseResult toy::ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
    mlir::DenseElementsAttr value;
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes))
        return mlir::failure();

    result.addTypes(value.getType());
    return mlir::success();
}

void toy::ConstantOp::print(mlir::OpAsmPrinter &printer) {
    printer << " ";
    printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
    printer << getValue();
}

mlir::LogicalResult toy::ConstantOp::verify() {
    // If the return type of the constant is not an unranked tensor, the shape
    // must match the shape of the attribute holding the data.
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
    if (!resultType)
        return mlir::success();

    // Check that the rank of the attribute type matches the rank of the constant
    // result type.
    auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
    if (attrType.getRank() != resultType.getRank()) {
        return emitOpError("return type must match the one of the attached value "
                           "attribute: ")
            << attrType.getRank() << " != " << resultType.getRank();
    }

    // Check that each of the dimensions match between the two types.
    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
        if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
        return emitOpError(
                    "return type shape mismatches its attribute at dimension ")
                << dim << ": " << attrType.getShape()[dim]
                << " != " << resultType.getShape()[dim];
        }
    }
    return mlir::success();
}


void toy::AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult toy::AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void toy::AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }


MLIR_TOY_DEFINE_CAPI_DIALECT_REGISTRATION(Toy, toy, toy::ToyDialect)
