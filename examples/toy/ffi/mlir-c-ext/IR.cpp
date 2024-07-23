#include "IR.h"

#include "mlir/CAPI/IR.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/ilist_node_options.h"

#include <functional>

struct MlirToyOpBuilderListenerImpl: public mlir::OpBuilder::Listener {
    std::function<void(mlir::Operation *, mlir::OpBuilder::InsertPoint)> operationInserted;
    std::function<void(mlir::Block *, mlir::Region *, mlir::Region::iterator)> blockInserted;

    virtual void notifyOperationInserted(mlir::Operation * op, mlir::OpBuilder::InsertPoint ip) override {
        operationInserted(op, ip);
    }
};

static inline MlirToyOpBuilderInsertPoint wrap(mlir::OpBuilder::InsertPoint ip) {
    return {wrap(ip.getBlock()), ip.getPoint().getNodePtr()};
}

static inline mlir::OpBuilder::InsertPoint unwrap(MlirToyOpBuilderInsertPoint ip) {
    using NodePtr = llvm::ilist_detail::IteratorTraits<llvm::ilist_detail::compute_node_options<mlir::Block>::type, false>::node_pointer;

    return mlir::OpBuilder::InsertPoint(unwrap(ip.block), mlir::Block::iterator(*static_cast<NodePtr>(ip.iterator)));
}
