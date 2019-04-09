// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "FastaiClient",
    products: [],
    dependencies: [
        .package(path: "../dev_swift/FastaiNotebook_04_callbacks")
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "FastaiClient",
            dependencies: ["FastaiNotebook_04_callbacks"]),
        .testTarget(
            name: "FastaiClientTests",
            dependencies: ["FastaiClient"]),
    ]
)
