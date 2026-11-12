fn main() {
    println!("cargo:rerun-if-env-changed=KAPITSA_LIB_DIR");
    if let Some(dir) = std::env::var_os("KAPITSA_LIB_DIR") {
        println!(
            "cargo:rustc-link-search=native={}",
            std::path::Path::new(&dir).display()
        );
    }
    println!("cargo:rustc-link-lib=dylib=kapitsa_sdk");
}
