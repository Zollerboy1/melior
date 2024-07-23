use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use convert_case::{Case, Casing};
use walkdir::WalkDir;

const LLVM_MAJOR_VERSION: usize = 18;

fn main() -> Result<(), Box<dyn Error>> {
    let mlir_prefix_var = format!("MLIR_SYS_{}0_PREFIX", LLVM_MAJOR_VERSION);

    println!("cargo:rerun-if-env-changed={}", mlir_prefix_var);

    let llvm_dir = llvm_config("--cmakedir", &mlir_prefix_var)?;
    let mlir_dir_path = Path::new(&llvm_dir)
        .parent()
        .ok_or("Invalid CMake directory")?
        .join("mlir");

    let cmake_build_path = cmake::Config::new("ffi")
        .define("LLVM_DIR", llvm_dir)
        .define("MLIR_DIR", mlir_dir_path)
        .build();

    println!("cargo:rustc-link-search={}/lib", cmake_build_path.display());

    let dialects = fs::read_dir("ffi/dialects")?
        .filter_map(|res| {
            res.map(|e| {
                e.file_name()
                    .to_str()?
                    .strip_suffix("-dialect.h")
                    .map(ToString::to_string)
            })
            .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;

    for dialect in dialects {
        println!(
            "cargo:rustc-link-lib=MLIR{}Dialect",
            dialect.to_case(Case::Pascal)
        );

        let dialect_bindings = bindgen::Builder::default()
            .header(format!("ffi/dialects/{dialect}-dialect.h"))
            .allowlist_function("mlirToy.*")
            .allowlist_type("MlirToy.*")
            .allowlist_var("MlirToy.*")
            .allowlist_recursively(false)
            .generate()
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from(env::var("OUT_DIR")?).join(format!(
            "{}_dialect_bindings.rs",
            dialect.to_case(Case::Snake)
        ));
        dialect_bindings.write_to_file(out_path)?;
    }

    println!("cargo:rustc-link-lib=MLIRExt");

    let ext_bindings = bindgen::Builder::default()
        .header("ffi/mlir-c-ext/mlir-c-ext.h")
        .allowlist_function("mlirToy.*")
        .allowlist_type("MlirToy.*")
        .allowlist_var("MlirToy.*")
        .allowlist_recursively(false)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR")?).join("ext_bindings.rs");
    ext_bindings.write_to_file(out_path)?;

    for entry in WalkDir::new("ffi").into_iter().filter_entry(|e| {
        if !e.file_type().is_file() {
            return true;
        }

        let path = e.path();

        if path.starts_with("ffi/build") {
            return false;
        }

        if e.file_name() == "CMakeLists.txt" {
            return true;
        }

        ["cpp", "h", "td"]
            .iter()
            .any(|ext| path.extension().map_or(false, |e| e == *ext))
    }) {
        let entry = entry?;

        if entry.file_type().is_file() {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }

    Ok(())
}

fn llvm_config(argument: &str, prefix_var: &str) -> Result<String, Box<dyn std::error::Error>> {
    let prefix = env::var(prefix_var)
        .map(|path| Path::new(&path).join("bin"))
        .unwrap_or_default();
    let call = format!(
        "{} --link-static {argument}",
        prefix.join("llvm-config").display()
    );

    Ok(std::str::from_utf8(
        &if cfg!(target_os = "windows") {
            Command::new("cmd").args(["/C", &call]).output()?
        } else {
            Command::new("sh").arg("-c").arg(&call).output()?
        }
        .stdout,
    )?
    .trim()
    .to_string())
}
