import XCTest
@testable import Reduced

final class ReducedTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(Reduced().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
