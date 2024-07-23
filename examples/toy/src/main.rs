use melior::{
    dialect::{DialectHandle, DialectRegistry},
    ir::{Location, Module},
    utility::register_all_dialects,
    Context,
};

use crate::dialect::{toy, DialectHandleExt as _};

mod dialect;

fn main() -> anyhow::Result<()> {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    DialectHandle::toy().insert_dialect(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let location = Location::unknown(&context);
    let module = Module::new(location);

    let constant = toy::constant(&context, &[2, 2], &[1.0, 2.0, 3.0, 4.0], location)?;
    let output = constant.result(0)?.into();

    let add = toy::add(&context, output, output, location)?;

    module.body().append_operation(constant);
    module.body().append_operation(add);

    module.as_operation().dump();

    Ok(())
}
