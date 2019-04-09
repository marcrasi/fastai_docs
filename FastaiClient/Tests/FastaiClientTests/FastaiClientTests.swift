import XCTest
@testable import FastaiClient

final class FastaiClientTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(FastaiClient().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
