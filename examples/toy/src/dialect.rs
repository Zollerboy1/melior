use melior::dialect::DialectHandle;

pub mod toy;

pub trait DialectHandleExt {
    fn toy() -> DialectHandle;
}

impl DialectHandleExt for DialectHandle {
    fn toy() -> DialectHandle {
        unsafe { DialectHandle::from_raw(toy::internal::mlirToyGetDialectHandle__toy__()) }
    }
}
