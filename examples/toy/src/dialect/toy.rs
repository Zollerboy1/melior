use melior::{
    ir::{
        attribute::{DenseElementsAttribute, FloatAttribute},
        operation::OperationBuilder,
        r#type::RankedTensorType,
        Identifier, Location, Operation, Type, Value,
    },
    Context, Error,
};

pub(in crate::dialect) mod internal {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]

    type MlirDialectHandle = mlir_sys::MlirDialectHandle;

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub fn constant<'c>(
    context: &'c Context,
    dimensions: &[u64],
    elements: &[f64],
    location: Location<'c>,
) -> Result<Operation<'c>, Error> {
    let float_type = Type::float64(context);

    let tensor_type = Type::from(RankedTensorType::new(dimensions, float_type, None));

    let elements = elements
        .iter()
        .map(|x| FloatAttribute::new(&context, float_type, *x).into())
        .collect::<Vec<_>>();

    let attribute = DenseElementsAttribute::new(tensor_type, &elements)?;

    OperationBuilder::new("toy.constant", location)
        .add_attributes(&[(Identifier::new(&context, "value"), attribute.into())])
        .add_results(&[tensor_type])
        .build()
}

pub fn add<'c>(
    context: &'c Context,
    lhs: Value<'c, '_>,
    rhs: Value<'c, '_>,
    location: Location<'c>,
) -> Result<Operation<'c>, Error> {
    OperationBuilder::new("toy.add", location)
        .add_operands(&[lhs, rhs])
        .add_results(&[Type::float64(context)])
        .build()
}
