import os.log

@available(iOS 14.0, tvOS 14.0, watchOS 7.0, macOS 11, *)
internal extension os.Logger {
   /// A logger instance that logs to '❗️Error' category within framewrok subsystem.
   static let error = os.Logger(subsystem: host, category: "❗️ Error")
   
   /// A logger instance that logs to '⚠️Warning' category within framewrok subsystem.
   static let warning = os.Logger(subsystem: host, category: "⚠️ Warning")
   
   /// A logger instance that logs to '♦️Debug' category within framewrok subsystem.
   static let debug = os.Logger(subsystem: host, category: "♦️ Debug")
   
   /// A logger instance that logs to '🔤Default' category within framewrok subsystem.
   static let `default` = os.Logger(subsystem: host, category: "🔤 Default")
   
   /// A logger instance that logs to '🔤Default' category within framewrok subsystem.
   static let success = os.Logger(subsystem: host, category: "🟢 Success")
   
   /// A logger instance that logs to '🟡 InProgess' category within framewrok subsystem.
   static let inProgress = os.Logger(subsystem: host, category: "🟡 InProgess")
   
   /// A logger instance that logs to '🔴Failure' category within framewrok subsystem.
   static let failure = os.Logger(subsystem: host, category: "🔴 Failure")
   
   /// A logger instance that logs to '🎭 Performance' category within framewrok subsystem.
   static let performanceLogger = os.Logger(subsystem: host, category: "🎭 Performance")
}


@available(iOS 14.0, tvOS 14.0, watchOS 7.0, macOS 11, *)
internal let performanceOSLog = OSLog(subsystem: host, category: "🎭 Performance")

