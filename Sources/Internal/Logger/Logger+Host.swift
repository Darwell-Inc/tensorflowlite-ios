import Foundation
import os.log

internal class BundleIdentifierMark {}

internal let host = Bundle(for: BundleIdentifierMark.self).bundleIdentifier!
