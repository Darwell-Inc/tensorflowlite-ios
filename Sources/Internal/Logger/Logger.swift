import os.log

/// Logs a message to console with specified log type, category and privacy level.
internal func log(
    _ message: String,
    logType: OSLogType = .default,
    category: os.Logger = .default,
    privacy: _OSLogPrivacy = .private,
    includeCallerLocation: Bool = true,
    fileID: String = #fileID,
    functionName: String = #function,
    lineNumber: Int = #line
) {
    
    var message = message
    if includeCallerLocation {
        let moduleAndFileName = fileID.replacingOccurrences(of: ".swift", with: "")
        let moduleName = String("\(fileID)".prefix(while: { $0 != "/" }))
        let fileName = moduleAndFileName
            .split(separator: "/")
            .suffix(1)
            .description
            .replacingOccurrences(of: "[", with: "")
            .replacingOccurrences(of: "\"", with: "")
            .replacingOccurrences(of: "]", with: "")
        let logLocationDescription = "\(lineNumber):\(moduleName).\(fileName).\(functionName)"
        message = "\(message) \n> location: \(logLocationDescription)"
    }
    
    // privacy argument must be resolved on compile time, hence ugly workaround
    // more info:
    // https://stackoverflow.com/questions/62675874/xcode-12-and-oslog-os-log-wrapping-oslogmessage-causes-compile-error-argumen#63036815
    switch privacy {
    case .private:
        category.log(level: logType, "\(message, align: .left(columns: 30), privacy: .private)")
    case .public:
        category.log(level: logType, "\(message, align: .left(columns: 30), privacy: .public)")
    case .auto:
        category.log(level: logType, "\(message, align: .left(columns: 30), privacy: .auto)")
    case .sensitive:
        category.log(level: logType, "\(message, align: .left(columns: 30), privacy: .sensitive)")
//    @unknown default:
//       category.log(level: logType, "\(message, align: .left(columns: 30), privacy: .private)")
    }
}


/// Logs an error to console using predefined error-logger category.
internal func log<E: Error>(
    _ error: E,
    privacy: _OSLogPrivacy = .private,
    fileID: String = #fileID,
    functionName: String = #function,
    lineNumber: Int = #line
) where E: CustomStringConvertible {
    
    // its important to pass magic ids to log, otherwise location is not forwarded
    log(
        error.description,
        logType: .error,
        category: .error,
        privacy: privacy,
        includeCallerLocation: true,
        fileID: fileID,
        functionName: functionName,
        lineNumber: lineNumber
    )
}


/// Logs an error to console using predefined error-logger category.
internal func log(
        _ error: Error,
        privacy: _OSLogPrivacy = .private,
        fileID: String = #fileID,
        functionName: String = #function,
        lineNumber: Int = #line
) {

    // its important to pass magic ids to log, otherwise location is not forwarded
    log(
            "\(error.localizedDescription): \(error)",
            logType: .error,
            category: .error,
            privacy: privacy,
            includeCallerLocation: true,
            fileID: fileID,
            functionName: functionName,
            lineNumber: lineNumber
    )
}



internal func debugEarlyExit(
    _ message: String,
    fileID: String = #fileID,
    functionName: String = #function,
    lineNumber: Int = #line
) {
    #if DEBUG
    let guardMessage = "early exit from \(functionName)\n> \(message)"

    // its important to pass magic ids to log, otherwise location is not forwarded
    log(
        guardMessage,
        logType: .debug,
        category: .debug,
        includeCallerLocation: true,
        fileID: fileID,
        functionName: functionName,
        lineNumber: lineNumber
    )
    #endif
}

/// Logs a debug message to console; Works only in DEBUG build configuration.
internal func debug(
    _ message: String,
    fileID: String = #fileID,
    functionName: String = #function,
    lineNumber: Int = #line
) {
    #if DEBUG
    log(
        message,
        logType: .debug,
        category: .debug,
        includeCallerLocation: true,
        fileID: fileID,
        functionName: functionName,
        lineNumber: lineNumber
    )
    #endif
}



