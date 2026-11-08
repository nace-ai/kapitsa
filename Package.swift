// swift-tools-version:5.9
// Kapitsa Swift package — wraps the C API (include/kapitsa.h) as a Swift module.
//
// This package exposes the C header so that Swift programs can call kapitsa_init(), kapitsa_load(),
// kapitsa_generate() and the rest without a bridging header. The underlying C library must be built separately
// (CMake) and linked by the consuming Xcode project or SwiftPM target. See docs/c-api.md.
//
// Usage in a Swift project's Package.swift:
//   .package(path: "../kapitsa"),              // local checkout
//   .target(name: "MyApp", dependencies: [
//     .product(name: "Kapitsa", package: "kapitsa"),
//   ]),

import PackageDescription

let package = Package(
    name: "kapitsa",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "Kapitsa", targets: ["CKapitsa"]),
    ],
    targets: [
        .systemLibrary(
            name: "CKapitsa",
            path: "swift",
            pkgConfig: nil,
            providers: []
        ),
    ]
)
