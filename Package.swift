// swift-tools-version:5.9
// Kapitsa Swift package — wraps the existing C API (sdk/c/kapitsa.h) as a Swift module.
//
// The underlying shared library (libkapitsa_sdk) must be built with CMake
// (cmake -DKAPITSA_BUILD_SDK=ON) and linked by the consuming Xcode project.
// See docs/c-api.md for the API reference.
//
// Usage in a Swift project's Package.swift:
//   .package(path: "../kapitsa"),
//   .target(name: "MyApp", dependencies: [
//     .product(name: "Kapitsa", package: "kapitsa"),
//   ]),

import PackageDescription

let package = Package(
    name: "kapitsa",
    platforms: [.macOS(.v14)],
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
