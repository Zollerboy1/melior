#ifndef TOY_DIALECTS_H
#define TOY_DIALECTS_H

#define MLIR_TOY_DECLARE_CAPI_DIALECT_REGISTRATION(Name, Namespace)\
    MlirDialectHandle mlirToyGetDialectHandle__##Namespace##__(void)

#define MLIR_TOY_DEFINE_CAPI_DIALECT_REGISTRATION(Name, Namespace, ClassName)\
    MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Name, Namespace, ClassName);\
    inline MlirDialectHandle mlirToyGetDialectHandle__##Namespace##__() {\
      return mlirGetDialectHandle__##Namespace##__();\
    }

#endif // TOY_DIALECTS_H
