
#import <stdint.h>
#import <stdarg.h>
#import <stdlib.h>
#import <stdbool.h>
#import <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif


// MARK: - c_api_types.


// Forward declare so dependent structs and methods can reference these types
// prior to the struct definitions.
//struct TfLiteContext;
//struct TfLiteDelegate;
//struct TfLiteRegistration;

// --------------------------------------------------------------------------
// TfLiteVersion returns a string describing version information of the
// TensorFlow Lite library. TensorFlow Lite uses semantic versioning.
// extern const char* TfLiteVersion(void);
extern const char* TfLiteVersion(void);


typedef enum {
  kTfLiteOk = 0,

  // Generally referring to an error in the runtime (i.e. interpreter)
  kTfLiteError = 1,

  // Generally referring to an error from a TfLiteDelegate itself.
  kTfLiteDelegateError = 2,

  // Generally referring to an error in applying a delegate due to
  // incompatibility between runtime and delegate, e.g., this error is returned
  // when trying to apply a TfLite delegate onto a model graph that's already
  // immutable.
  kTfLiteApplicationError = 3,

  // Generally referring to serialized delegate data not being found.
  // See tflite::delegates::Serialization.
  kTfLiteDelegateDataNotFound = 4,

  // Generally referring to data-writing issues in delegate serialization.
  // See tflite::delegates::Serialization.
  kTfLiteDelegateDataWriteError = 5,

  // Generally referring to data-reading issues in delegate serialization.
  // See tflite::delegates::Serialization.
  kTfLiteDelegateDataReadError = 6,
} TfLiteStatus;

// Types supported by tensor
typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10,
  kTfLiteFloat64 = 11,
  kTfLiteComplex128 = 12,
  kTfLiteUInt64 = 13,
  kTfLiteResource = 14,
  kTfLiteVariant = 15,
  kTfLiteUInt32 = 16,
} TfLiteType;

// Legacy. Will be deprecated in favor of TfLiteAffineQuantization.
// If per-layer quantization is specified this field will still be populated in
// addition to TfLiteAffineQuantization.
// Parameters for asymmetric quantization. Quantized values can be converted
// back to float using:
//     real_value = scale * (quantized_value - zero_point)
typedef struct TfLiteQuantizationParams {
  float scale;
  int32_t zero_point;
} TfLiteQuantizationParams;


//  MARK: - common.h


#define kTfLiteOptionalTensor (-1)

// Fixed size list of integers. Used for dimensions and inputs/outputs tensor
// indices
typedef struct TfLiteIntArray {
  int size;
// gcc 6.1+ have a bug where flexible members aren't properly handled
// https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
#if (!defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
     __GNUC_MINOR__ >= 1) ||                                      \
    defined(HEXAGON) ||                                           \
    (defined(__clang__) && __clang_major__ == 7 && __clang_minor__ == 1)
  int data[0];
#else
  int data[];
#endif
} TfLiteIntArray;

// Given the size (number of elements) in a TfLiteIntArray, calculate its size
// in bytes.
int TfLiteIntArrayGetSizeInBytes(int size);

#ifndef TF_LITE_STATIC_MEMORY
// Create a array of a given `size` (uninitialized entries).
// This returns a pointer, that you must free using TfLiteIntArrayFree().
TfLiteIntArray* TfLiteIntArrayCreate(int size);
#endif

// Check if two intarrays are equal. Returns 1 if they are equal, 0 otherwise.
int TfLiteIntArrayEqual(struct TfLiteIntArray* a, struct TfLiteIntArray* b);

// Check if an intarray equals an array. Returns 1 if equals, 0 otherwise.
int TfLiteIntArrayEqualsArray(struct TfLiteIntArray* a, int b_size,
                              const int b_data[]);

#ifndef TF_LITE_STATIC_MEMORY
// Create a copy of an array passed as `src`.
// You are expected to free memory with TfLiteIntArrayFree
TfLiteIntArray* TfLiteIntArrayCopy(struct TfLiteIntArray* src);

// Free memory of array `a`.
void TfLiteIntArrayFree(struct TfLiteIntArray* a);
#endif  // TF_LITE_STATIC_MEMORY

// Fixed size list of floats. Used for per-channel quantization.
typedef struct TfLiteFloatArray {
  int size;
// gcc 6.1+ have a bug where flexible members aren't properly handled
// https://github.com/google/re2/commit/b94b7cd42e9f02673cd748c1ac1d16db4052514c
// This also applies to the toolchain used for Qualcomm Hexagon DSPs.
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 6 && \
    __GNUC_MINOR__ >= 1
  float data[0];
#else
  float data[];
#endif
} TfLiteFloatArray;

// Given the size (number of elements) in a TfLiteFloatArray, calculate its size
// in bytes.
int TfLiteFloatArrayGetSizeInBytes(int size);

#ifndef TF_LITE_STATIC_MEMORY
// Create a array of a given `size` (uninitialized entries).
// This returns a pointer, that you must free using TfLiteFloatArrayFree().
TfLiteFloatArray* TfLiteFloatArrayCreate(int size);

// Free memory of array `a`.
void TfLiteFloatArrayFree(struct TfLiteFloatArray* a);
#endif  // TF_LITE_STATIC_MEMORY

// Since we must not depend on any libraries, define a minimal subset of
// error macros while avoiding names that have pre-conceived meanings like
// assert and check.

// Try to make all reporting calls through TF_LITE_KERNEL_LOG rather than
// calling the context->ReportError function directly, so that message strings
// can be stripped out if the binary size needs to be severely optimized.
#ifndef TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_KERNEL_LOG(context, ...)            \
  do {                                              \
    (context)->ReportError((context), __VA_ARGS__); \
  } while (false)

#define TF_LITE_MAYBE_KERNEL_LOG(context, ...)        \
  do {                                                \
    if ((context) != nullptr) {                       \
      (context)->ReportError((context), __VA_ARGS__); \
    }                                                 \
  } while (false)
#else  // TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_KERNEL_LOG(context, ...)
#define TF_LITE_MAYBE_KERNEL_LOG(context, ...)
#endif  // TF_LITE_STRIP_ERROR_STRINGS

// Check whether value is true, and if not return kTfLiteError from
// the current function (and report the error string msg).
#define TF_LITE_ENSURE_MSG(context, value, msg)        \
  do {                                                 \
    if (!(value)) {                                    \
      TF_LITE_KERNEL_LOG((context), __FILE__ " " msg); \
      return kTfLiteError;                             \
    }                                                  \
  } while (0)

// Check whether the value `a` is true, and if not return kTfLiteError from
// the current function, while also reporting the location of the error.
#define TF_LITE_ENSURE(context, a)                                      \
  do {                                                                  \
    if (!(a)) {                                                         \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s was not true.", __FILE__, \
                         __LINE__, #a);                                 \
      return kTfLiteError;                                              \
    }                                                                   \
  } while (0)

#define TF_LITE_ENSURE_STATUS(a) \
  do {                           \
    const TfLiteStatus s = (a);  \
    if (s != kTfLiteOk) {        \
      return s;                  \
    }                            \
  } while (0)

// Check whether the value `a == b` is true, and if not return kTfLiteError from
// the current function, while also reporting the location of the error.
// `a` and `b` may be evaluated more than once, so no side effects or
// extremely expensive computations should be done.
// NOTE: Use TF_LITE_ENSURE_TYPES_EQ if comparing TfLiteTypes.
#define TF_LITE_ENSURE_EQ(context, a, b)                                   \
  do {                                                                     \
    if ((a) != (b)) {                                                      \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s != %s (%d != %d)", __FILE__, \
                         __LINE__, #a, #b, (a), (b));                      \
      return kTfLiteError;                                                 \
    }                                                                      \
  } while (0)

#define TF_LITE_ENSURE_TYPES_EQ(context, a, b)                             \
  do {                                                                     \
    if ((a) != (b)) {                                                      \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s != %s (%s != %s)", __FILE__, \
                         __LINE__, #a, #b, TfLiteTypeGetName(a),           \
                         TfLiteTypeGetName(b));                            \
      return kTfLiteError;                                                 \
    }                                                                      \
  } while (0)

#define TF_LITE_ENSURE_NEAR(context, a, b, epsilon)                          \
  do {                                                                       \
    auto delta = ((a) > (b)) ? ((a) - (b)) : ((b) - (a));                    \
    if (delta > epsilon) {                                                   \
      TF_LITE_KERNEL_LOG((context), "%s:%d %s not near %s (%f != %f)",       \
                         __FILE__, __LINE__, #a, #b, static_cast<double>(a), \
                         static_cast<double>(b));                            \
      return kTfLiteError;                                                   \
    }                                                                        \
  } while (0)

#define TF_LITE_ENSURE_OK(context, status) \
  do {                                     \
    const TfLiteStatus s = (status);       \
    if ((s) != kTfLiteOk) {                \
      return s;                            \
    }                                      \
  } while (0)

// Single-precision complex data type compatible with the C99 definition.
typedef struct TfLiteComplex64 {
  float re, im;  // real and imaginary parts, respectively.
} TfLiteComplex64;

// Double-precision complex data type compatible with the C99 definition.
typedef struct TfLiteComplex128 {
  double re, im;  // real and imaginary parts, respectively.
} TfLiteComplex128;

// Half precision data type compatible with the C99 definition.
typedef struct TfLiteFloat16 {
  uint16_t data;
} TfLiteFloat16;

// Return the name of a given type, for error reporting purposes.
const char* TfLiteTypeGetName(TfLiteType type);

// SupportedQuantizationTypes.
typedef enum TfLiteQuantizationType {
  // No quantization.
  kTfLiteNoQuantization = 0,
  // Affine quantization (with support for per-channel quantization).
  // Corresponds to TfLiteAffineQuantization.
  kTfLiteAffineQuantization = 1,
} TfLiteQuantizationType;

// Structure specifying the quantization used by the tensor, if-any.
typedef struct TfLiteQuantization {
  // The type of quantization held by params.
  TfLiteQuantizationType type;
  // Holds an optional reference to a quantization param structure. The actual
  // type depends on the value of the `type` field (see the comment there for
  // the values and corresponding types).
  void* params;
} TfLiteQuantization;

// Parameters for asymmetric quantization across a dimension (i.e per output
// channel quantization).
// quantized_dimension specifies which dimension the scales and zero_points
// correspond to.
// For a particular value in quantized_dimension, quantized values can be
// converted back to float using:
//     real_value = scale * (quantized_value - zero_point)
typedef struct TfLiteAffineQuantization {
  struct TfLiteFloatArray* scale;
  struct TfLiteIntArray* zero_point;
  int32_t quantized_dimension;
} TfLiteAffineQuantization;

/* A union of pointers that points to memory for a given tensor. */
typedef union TfLitePtrUnion {
  /* Do not access these members directly, if possible, use
   * GetTensorData<TYPE>(tensor) instead, otherwise only access .data, as other
   * members are deprecated. */
  int32_t* i32;
  uint32_t* u32;
  int64_t* i64;
  uint64_t* u64;
  float* f;
  struct TfLiteFloat16* f16;
  double* f64;
  char* raw;
  const char* raw_const;
  uint8_t* uint8;
  bool* b;
  int16_t* i16;
  struct TfLiteComplex64* c64;
  struct TfLiteComplex128* c128;
  int8_t* int8;
  /* Only use this member. */
  void* data;
} TfLitePtrUnion;

// Memory allocation strategies.
//  * kTfLiteMmapRo: Read-only memory-mapped data, or data externally allocated.
//  * kTfLiteArenaRw: Arena allocated with no guarantees about persistence,
//        and available during eval.
//  * kTfLiteArenaRwPersistent: Arena allocated but persistent across eval, and
//        only available during eval.
//  * kTfLiteDynamic: Allocated during eval, or for string tensors.
//  * kTfLitePersistentRo: Allocated and populated during prepare. This is
//        useful for tensors that can be computed during prepare and treated
//        as constant inputs for downstream ops (also in prepare).
//  * kTfLiteCustom: Custom memory allocation provided by the user. See
//        TfLiteCustomAllocation below.
typedef enum TfLiteAllocationType {
  kTfLiteMemNone = 0,
  kTfLiteMmapRo,
  kTfLiteArenaRw,
  kTfLiteArenaRwPersistent,
  kTfLiteDynamic,
  kTfLitePersistentRo,
  kTfLiteCustom,
} TfLiteAllocationType;

// The delegates should use zero or positive integers to represent handles.
// -1 is reserved from unallocated status.
typedef int TfLiteBufferHandle;
enum {
  kTfLiteNullBufferHandle = -1,
};

// Storage format of each dimension in a sparse tensor.
typedef enum TfLiteDimensionType {
  kTfLiteDimDense = 0,
  kTfLiteDimSparseCSR,
} TfLiteDimensionType;

// Metadata to encode each dimension in a sparse tensor.
typedef struct TfLiteDimensionMetadata {
  TfLiteDimensionType format;
  int dense_size;
  struct TfLiteIntArray* array_segments;
  struct TfLiteIntArray* array_indices;
} TfLiteDimensionMetadata;

// Parameters used to encode a sparse tensor. For detailed explanation of each
// field please refer to lite/schema/schema.fbs.
typedef struct TfLiteSparsity {
  struct TfLiteIntArray* traversal_order;
  struct TfLiteIntArray* block_map;
  struct TfLiteDimensionMetadata* dim_metadata;
  int dim_metadata_size;
} TfLiteSparsity;

// Defines a custom memory allocation not owned by the runtime.
// `data` should be aligned to kDefaultTensorAlignment defined in
// lite/util.h. (Currently 64 bytes)
// NOTE: See Interpreter.SetCustomAllocationForTensor for details on usage.
typedef struct TfLiteCustomAllocation {
  void* data;
  size_t bytes;
} TfLiteCustomAllocation;

// The flags used in `Interpreter::SetCustomAllocationForTensor`.
// Note that this is a bitmask, so the values should be 1, 2, 4, 8, ...etc.
typedef enum TfLiteCustomAllocationFlags {
  kTfLiteCustomAllocationFlagsNone = 0,
  // Skips checking whether allocation.data points to an aligned buffer as
  // expected by the TFLite runtime.
  // NOTE: Setting this flag can cause crashes when calling Invoke().
  // Use with caution.
  kTfLiteCustomAllocationFlagsSkipAlignCheck = 1,
} TfLiteCustomAllocationFlags;

// A tensor in the interpreter system which is a wrapper around a buffer of
// data including a dimensionality (or NULL if not currently defined).
#ifndef TF_LITE_STATIC_MEMORY
typedef struct TfLiteTensor {
  // The data type specification for data stored in `data`. This affects
  // what member of `data` union should be used.
  TfLiteType type;
  // A union of data pointers. The appropriate type should be used for a typed
  // tensor based on `type`.
  TfLitePtrUnion data;
  // A pointer to a structure representing the dimensionality interpretation
  // that the buffer should have. NOTE: the product of elements of `dims`
  // and the element datatype size should be equal to `bytes` below.
  struct TfLiteIntArray* dims;
  // Quantization information.
  struct TfLiteQuantizationParams params;
  // How memory is mapped
  //  kTfLiteMmapRo: Memory mapped read only.
  //  i.e. weights
  //  kTfLiteArenaRw: Arena allocated read write memory
  //  (i.e. temporaries, outputs).
  TfLiteAllocationType allocation_type;
  // The number of bytes required to store the data of this Tensor. I.e.
  // (bytes of each element) * dims[0] * ... * dims[n-1].  For example, if
  // type is kTfLiteFloat32 and dims = {3, 2} then
  // bytes = sizeof(float) * 3 * 2 = 4 * 3 * 2 = 24.
  size_t bytes;

  // An opaque pointer to a tflite::MMapAllocation
  const void* allocation;

  // Null-terminated name of this tensor.
  const char* name;

  // The delegate which knows how to handle `buffer_handle`.
  // WARNING: This is an experimental interface that is subject to change.
  struct TfLiteDelegate* delegate;

  // An integer buffer handle that can be handled by `delegate`.
  // The value is valid only when delegate is not null.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteBufferHandle buffer_handle;

  // If the delegate uses its own buffer (e.g. GPU memory), the delegate is
  // responsible to set data_is_stale to true.
  // `delegate->CopyFromBufferHandle` can be called to copy the data from
  // delegate buffer.
  // WARNING: This is an // experimental interface that is subject to change.
  bool data_is_stale;

  // True if the tensor is a variable.
  bool is_variable;

  // Quantization information. Replaces params field above.
  struct TfLiteQuantization quantization;

  // Parameters used to encode a sparse tensor.
  // This is optional. The field is NULL if a tensor is dense.
  // WARNING: This is an experimental interface that is subject to change.
  struct TfLiteSparsity* sparsity;

  // Optional. Encodes shapes with unknown dimensions with -1. This field is
  // only populated when unknown dimensions exist in a read-write tensor (i.e.
  // an input or output tensor). (e.g.  `dims` contains [1, 1, 1, 3] and
  // `dims_signature` contains [1, -1, -1, 3]).
  const TfLiteIntArray* dims_signature;
} TfLiteTensor;

// A structure representing an instance of a node.
// This structure only exhibits the inputs, outputs, user defined data and some
// node properties (like statefulness), not other features like the type.
typedef struct TfLiteNode {
  // Inputs to this node expressed as indices into the simulator's tensors.
  struct TfLiteIntArray* inputs;

  // Outputs to this node expressed as indices into the simulator's tensors.
  struct TfLiteIntArray* outputs;

  // intermediate tensors to this node expressed as indices into the simulator's
  // tensors.
  struct TfLiteIntArray* intermediates;

  // Temporary tensors uses during the computations. This usually contains no
  // tensors, but ops are allowed to change that if they need scratch space of
  // any sort.
  struct TfLiteIntArray* temporaries;

  // Opaque data provided by the node implementer through `Registration.init`.
  void* user_data;

  // Opaque data provided to the node if the node is a builtin. This is usually
  // a structure defined in builtin_op_data.h
  void* builtin_data;

  // Custom initial data. This is the opaque data provided in the flatbuffer.
  // WARNING: This is an experimental interface that is subject to change.
  const void* custom_initial_data;
  int custom_initial_data_size;

  // The pointer to the delegate. This is non-null only when the node is
  // created by calling `interpreter.ModifyGraphWithDelegate`.
  // WARNING: This is an experimental interface that is subject to change.
  struct TfLiteDelegate* delegate;

  // Whether this op might have side effect (e.g. stateful op).
  bool might_have_side_effect;
} TfLiteNode;
#else   // defined(TF_LITE_STATIC_MEMORY)?
// NOTE: This flag is opt-in only at compile time.
//
// Specific reduced TfLiteTensor struct for TF Micro runtime. This struct
// contains only the minimum fields required to initialize and prepare a micro
// inference graph. The fields in this struct have been ordered from
// largest-to-smallest for optimal struct sizeof.
//
// This struct does not use:
// - allocation
// - buffer_handle
// - data_is_stale
// - delegate
// - dims_signature
// - name
// - sparsity
typedef struct TfLiteTensor {
  // TODO(b/155784997): Consider consolidating these quantization fields:
  // Quantization information. Replaces params field above.
  struct TfLiteQuantization quantization;

  // Quantization information.
  struct TfLiteQuantizationParams params;

  // A union of data pointers. The appropriate type should be used for a typed
  // tensor based on `type`.
  TfLitePtrUnion data;

  // A pointer to a structure representing the dimensionality interpretation
  // that the buffer should have. NOTE: the product of elements of `dims`
  // and the element datatype size should be equal to `bytes` below.
  struct TfLiteIntArray* dims;

  // The number of bytes required to store the data of this Tensor. I.e.
  // (bytes of each element) * dims[0] * ... * dims[n-1].  For example, if
  // type is kTfLiteFloat32 and dims = {3, 2} then
  // bytes = sizeof(float) * 3 * 2 = 4 * 3 * 2 = 24.
  size_t bytes;

  // The data type specification for data stored in `data`. This affects
  // what member of `data` union should be used.
  TfLiteType type;

  // How memory is mapped
  //  kTfLiteMmapRo: Memory mapped read only.
  //  i.e. weights
  //  kTfLiteArenaRw: Arena allocated read write memory
  //  (i.e. temporaries, outputs).
  TfLiteAllocationType allocation_type;

  // True if the tensor is a variable.
  bool is_variable;
} TfLiteTensor;

// Specific reduced TfLiteNode struct for TF Micro runtime. This struct contains
// only the minimum fields required to represent a node.
//
// This struct does not use:
// - delegate
// - intermediates
// - temporaries
typedef struct TfLiteNode {
  // Inputs to this node expressed as indices into the simulator's tensors.
  struct TfLiteIntArray* inputs;

  // Outputs to this node expressed as indices into the simulator's tensors.
  struct TfLiteIntArray* outputs;

  // intermediate tensors to this node expressed as indices into the simulator's
  // tensors.
  struct TfLiteIntArray* intermediates;

  // Opaque data provided by the node implementer through `Registration.init`.
  void* user_data;

  // Opaque data provided to the node if the node is a builtin. This is usually
  // a structure defined in builtin_op_data.h
  void* builtin_data;

  // Custom initial data. This is the opaque data provided in the flatbuffer.
  // WARNING: This is an experimental interface that is subject to change.
  const void* custom_initial_data;
  int custom_initial_data_size;
} TfLiteNode;
#endif  // TF_LITE_STATIC_MEMORY

// Light-weight tensor struct for TF Micro runtime. Provides the minimal amount
// of information required for a kernel to run during TfLiteRegistration::Eval.
// TODO(b/160955687): Move this field into TF_LITE_STATIC_MEMORY when TFLM
// builds with this flag by default internally.
typedef struct TfLiteEvalTensor {
  // A union of data pointers. The appropriate type should be used for a typed
  // tensor based on `type`.
  TfLitePtrUnion data;

  // A pointer to a structure representing the dimensionality interpretation
  // that the buffer should have.
  struct TfLiteIntArray* dims;

  // The data type specification for data stored in `data`. This affects
  // what member of `data` union should be used.
  TfLiteType type;
} TfLiteEvalTensor;

#ifndef TF_LITE_STATIC_MEMORY
// Free data memory of tensor `t`.
void TfLiteTensorDataFree(struct TfLiteTensor* t);

// Free quantization data.
void TfLiteQuantizationFree(struct TfLiteQuantization* quantization);

// Free sparsity parameters.
void TfLiteSparsityFree(struct TfLiteSparsity* sparsity);

// Free memory of tensor `t`.
void TfLiteTensorFree(struct TfLiteTensor* t);

// Set all of a tensor's fields (and free any previously allocated data).
void TfLiteTensorReset(TfLiteType type, const char* name, struct TfLiteIntArray* dims,
                       struct TfLiteQuantizationParams quantization, char* buffer,
                       size_t size, TfLiteAllocationType allocation_type,
                       const void* allocation, bool is_variable,
                       TfLiteTensor* tensor);

// Resize the allocated data of a (dynamic) tensor. Tensors with allocation
// types other than kTfLiteDynamic will be ignored.
void TfLiteTensorRealloc(size_t num_bytes, TfLiteTensor* tensor);
#endif  // TF_LITE_STATIC_MEMORY

// WARNING: This is an experimental interface that is subject to change.
//
// Currently, TfLiteDelegateParams has to be allocated in a way that it's
// trivially destructable. It will be stored as `builtin_data` field in
// `TfLiteNode` of the delegate node.
//
// See also the `CreateDelegateParams` function in `interpreter.cc` details.
typedef struct TfLiteDelegateParams {
  struct TfLiteDelegate* delegate;
  struct TfLiteIntArray* nodes_to_replace;
  struct TfLiteIntArray* input_tensors;
  struct TfLiteIntArray* output_tensors;
} TfLiteDelegateParams;

typedef struct TfLiteContext {
  // Number of tensors in the context.
  size_t tensors_size;

  // The execution plan contains a list of the node indices in execution
  // order. execution_plan->size is the current number of nodes. And,
  // execution_plan->data[0] is the first node that needs to be run.
  // TfLiteDelegates can traverse the current execution plan by iterating
  // through each member of this array and using GetNodeAndRegistration() to
  // access details about a node. i.e.
  //
  // TfLiteIntArray* execution_plan;
  // TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &execution_plan));
  // for (int exec_index = 0; exec_index < execution_plan->size; exec_index++) {
  //    int node_index = execution_plan->data[exec_index];
  //    TfLiteNode* node;
  //    TfLiteRegistration* reg;
  //    context->GetNodeAndRegistration(context, node_index, &node, &reg);
  // }
  // Note: the memory pointed by '`*execution_plan` is OWNED by TfLite runtime.
  // Future calls to GetExecutionPlan invalidates earlier outputs. The following
  // code snippet shows the issue of such an invocation pattern. After calling
  // CheckNode, subsequent access to `plan_1st` is undefined.
  //
  // void CheckNode(const TfLiteNode* node) {
  //   ...
  //   TfLiteIntArray* plan_2nd;
  //   TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan_2nd));
  //   ...
  // }
  //
  // TfLiteIntArray* plan_1st;
  // TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan_1st));
  // for (int exec_index = 0; exec_index < plan_1st->size; exec_index++) {
  //    int node_index = plan_1st->data[exec_index];
  //    TfLiteNode* node;
  //    TfLiteRegistration* reg;
  //    context->GetNodeAndRegistration(context, node_index, &node, &reg);
  //    CheckNode(node);
  // }
  //
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*GetExecutionPlan)(struct TfLiteContext* context,
                                   struct TfLiteIntArray** execution_plan);

  // An array of tensors in the interpreter context (of length `tensors_size`)
  TfLiteTensor* tensors;

  // opaque full context ptr (an opaque c++ data structure)
  void* impl_;

  // Request memory pointer be resized. Updates dimensions on the tensor.
  // NOTE: ResizeTensor takes ownership of newSize.
  TfLiteStatus (*ResizeTensor)(struct TfLiteContext*, TfLiteTensor* tensor,
                               struct TfLiteIntArray* new_size);
  // Request that an error be reported with format string msg.
  void (*ReportError)(struct TfLiteContext*, const char* msg, ...);

  // Add `tensors_to_add` tensors, preserving pre-existing Tensor entries.  If
  // non-null, the value pointed to by `first_new_tensor_index` will be set to
  // the index of the first new tensor.
  TfLiteStatus (*AddTensors)(struct TfLiteContext*, int tensors_to_add,
                             int* first_new_tensor_index);

  // Get a Tensor node by node_index.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*GetNodeAndRegistration)(
      struct TfLiteContext*, int node_index, struct TfLiteNode** node,
      struct TfLiteRegistration** registration);

  // Replace ops with one or more stub delegate operations. This function
  // does not take ownership of `nodes_to_replace`.
  TfLiteStatus (*ReplaceNodeSubsetsWithDelegateKernels)(
      struct TfLiteContext*, struct TfLiteRegistration registration,
      struct TfLiteIntArray* nodes_to_replace, struct TfLiteDelegate* delegate);

  // Number of threads that are recommended to subsystems like gemmlowp and
  // eigen.
  int recommended_num_threads;

  // Flag for allowing float16 precision for FP32 calculation.
  // default: false.
  // WARNING: This is an experimental API and subject to change.
  bool allow_fp32_relax_to_fp16;

  // Pointer to the op-level profiler, if set; nullptr otherwise.
  void* profiler;

  // Allocate persistent buffer which has the same life time as the interpreter.
  // Returns nullptr on failure.
  // The memory is allocated from heap for TFL, and from tail in TFLM.
  // This method is only available in Init or Prepare stage.
  // WARNING: This is an experimental interface that is subject to change.
  void* (*AllocatePersistentBuffer)(struct TfLiteContext* ctx, size_t bytes);

  // Allocate a buffer which will be deallocated right after invoke phase.
  // The memory is allocated from heap in TFL, and from volatile arena in TFLM.
  // This method is only available in invoke stage.
  // NOTE: If possible use RequestScratchBufferInArena method to avoid memory
  // allocation during inference time.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*AllocateBufferForEval)(struct TfLiteContext* ctx, size_t bytes,
                                        void** ptr);

  // Request a scratch buffer in the arena through static memory planning.
  // This method is only available in Prepare stage and the buffer is allocated
  // by the interpreter between Prepare and Eval stage. In Eval stage,
  // GetScratchBuffer API can be used to fetch the address.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*RequestScratchBufferInArena)(struct TfLiteContext* ctx,
                                              size_t bytes, int* buffer_idx);

  // Get the scratch buffer pointer.
  // This method is only available in Eval stage.
  // WARNING: This is an experimental interface that is subject to change.
  void* (*GetScratchBuffer)(struct TfLiteContext* ctx, int buffer_idx);

  // Resize the memory pointer of the `tensor`. This method behaves the same as
  // `ResizeTensor`, except that it makes a copy of the shape array internally
  // so the shape array could be deallocated right afterwards.
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*ResizeTensorExplicit)(struct TfLiteContext* ctx,
                                       TfLiteTensor* tensor, int dims,
                                       const int* shape);

  // This method provides a preview of post-delegation partitioning. Each
  // TfLiteDelegateParams in the referenced array corresponds to one instance of
  // the delegate kernel.
  // Example usage:
  //
  // TfLiteIntArray* nodes_to_replace = ...;
  // TfLiteDelegateParams* params_array;
  // int num_partitions = 0;
  // TF_LITE_ENSURE_STATUS(context->PreviewDelegatePartitioning(
  //    context, delegate, nodes_to_replace, &params_array, &num_partitions));
  // for (int idx = 0; idx < num_partitions; idx++) {
  //    const auto& partition_params = params_array[idx];
  //    ...
  // }
  //
  // NOTE: The context owns the memory referenced by partition_params_array. It
  // will be cleared with another call to PreviewDelegateParitioning, or after
  // TfLiteDelegateParams::Prepare returns.
  //
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*PreviewDelegatePartitioning)(
      struct TfLiteContext* context, struct TfLiteIntArray* nodes_to_replace,
      struct TfLiteDelegateParams** partition_params_array, int* num_partitions);

  // Returns a TfLiteTensor struct for a given index.
  // WARNING: This is an experimental interface that is subject to change.
  // WARNING: This method may not be available on all platforms.
  TfLiteTensor* (*GetTensor)(struct TfLiteContext* context,
                             int tensor_idx);

  // Returns a TfLiteEvalTensor struct for a given index.
  // WARNING: This is an experimental interface that is subject to change.
  // WARNING: This method may not be available on all platforms.
  TfLiteEvalTensor* (*GetEvalTensor)(struct TfLiteContext* context,
                                     int tensor_idx);

  // Retrieves named metadata buffer from the TFLite model.
  // Returns kTfLiteOk if metadata is successfully obtained from the flatbuffer
  // Model: that is, there exists a `metadata` entry with given `name` string.
  // (see TFLite's schema.fbs).
  // The corresponding `buffer` information is populated in `ptr` & `bytes`.
  // The data from `ptr` is valid for the lifetime of the Interpreter.
  //
  // WARNING: This is an experimental interface that is subject to change.
  TfLiteStatus (*GetModelMetadata)(struct TfLiteContext* context,
                                   const char* name, const char** ptr,
                                   size_t* bytes);
} TfLiteContext;

// The list of external context types known to TF Lite. This list exists solely
// to avoid conflicts and to ensure ops can share the external contexts they
// need. Access to the external contexts is controlled by one of the
// corresponding support files.
typedef enum TfLiteExternalContextType {
  kTfLiteEigenContext = 0,       // include eigen_support.h to use.
  kTfLiteGemmLowpContext = 1,    // include gemm_support.h to use.
  kTfLiteEdgeTpuContext = 2,     // Placeholder for Edge TPU support.
  kTfLiteCpuBackendContext = 3,  // include cpu_backend_context.h to use.
  kTfLiteMaxExternalContexts = 4
} TfLiteExternalContextType;

// An external context is a collection of information unrelated to the TF Lite
// framework, but useful to a subset of the ops. TF Lite knows very little
// about the actual contexts, but it keeps a list of them, and is able to
// refresh them if configurations like the number of recommended threads
// change.
typedef struct TfLiteExternalContext {
  TfLiteExternalContextType type;
  TfLiteStatus (*Refresh)(TfLiteContext* context);
} TfLiteExternalContext;

// Access external contexts by type.
// WARNING: This is an experimental interface that is subject to change.
TfLiteExternalContext* (*GetExternalContext)(struct TfLiteContext*,
                                             TfLiteExternalContextType);
// Set the value of a external context. Does not take ownership of the
// pointer.
// WARNING: This is an experimental interface that is subject to change.
void (*SetExternalContext)(struct TfLiteContext*, TfLiteExternalContextType,
                           struct TfLiteExternalContext*);

typedef struct TfLiteRegistration {
  // Initializes the op from serialized data.
  // If a built-in op:
  //   `buffer` is the op's params data (TfLiteLSTMParams*).
  //   `length` is zero.
  // If custom op:
  //   `buffer` is the op's `custom_options`.
  //   `length` is the size of the buffer.
  //
  // Returns a type-punned (i.e. void*) opaque data (e.g. a primitive pointer
  // or an instance of a struct).
  //
  // The returned pointer will be stored with the node in the `user_data` field,
  // accessible within prepare and invoke functions below.
  // NOTE: if the data is already in the desired format, simply implement this
  // function to return `nullptr` and implement the free function to be a no-op.
  void* (*init)(struct TfLiteContext* context, const char* buffer, size_t length);

  // The pointer `buffer` is the data previously returned by an init invocation.
  void (*free)(struct TfLiteContext* context, void* buffer);

  // prepare is called when the inputs this node depends on have been resized.
  // context->ResizeTensor() can be called to request output tensors to be
  // resized.
  //
  // Returns kTfLiteOk on success.
  TfLiteStatus (*prepare)(struct TfLiteContext* context, struct TfLiteNode* node);

  // Execute the node (should read node->inputs and output to node->outputs).
  // Returns kTfLiteOk on success.
  TfLiteStatus (*invoke)(struct TfLiteContext* context, struct TfLiteNode* node);

  // profiling_string is called during summarization of profiling information
  // in order to group executions together. Providing a value here will cause a
  // given op to appear multiple times is the profiling report. This is
  // particularly useful for custom ops that can perform significantly
  // different calculations depending on their `user-data`.
  const char* (*profiling_string)(struct TfLiteContext* context,
                                  struct TfLiteNode* node);

  // Builtin codes. If this kernel refers to a builtin this is the code
  // of the builtin. This is so we can do marshaling to other frameworks like
  // NN API.
  // Note: It is the responsibility of the registration binder to set this
  // properly.
  int32_t builtin_code;

  // Custom op name. If the op is a builtin, this will be null.
  // Note: It is the responsibility of the registration binder to set this
  // properly.
  // WARNING: This is an experimental interface that is subject to change.
  const char* custom_name;

  // The version of the op.
  // Note: It is the responsibility of the registration binder to set this
  // properly.
  int version;
} TfLiteRegistration;

// The flags used in `TfLiteDelegate`. Note that this is a bitmask, so the
// values should be 1, 2, 4, 8, ...etc.
typedef enum TfLiteDelegateFlags {
  kTfLiteDelegateFlagsNone = 0,
  // The flag is set if the delegate can handle dynamic sized tensors.
  // For example, the output shape of a `Resize` op with non-constant shape
  // can only be inferred when the op is invoked.
  // In this case, the Delegate is responsible for calling
  // `SetTensorToDynamic` to mark the tensor as a dynamic tensor, and calling
  // `ResizeTensor` when invoking the op.
  //
  // If the delegate isn't capable to handle dynamic tensors, this flag need
  // to be set to false.
  kTfLiteDelegateFlagsAllowDynamicTensors = 1,

  // This flag can be used by delegates (that allow dynamic tensors) to ensure
  // applicable tensor shapes are automatically propagated in the case of tensor
  // resizing.
  // This means that non-dynamic (allocation_type != kTfLiteDynamic) I/O tensors
  // of a delegate kernel will have correct shapes before its Prepare() method
  // is called. The runtime leverages TFLite builtin ops in the original
  // execution plan to propagate shapes.
  //
  // A few points to note:
  // 1. This requires kTfLiteDelegateFlagsAllowDynamicTensors. If that flag is
  // false, this one is redundant since the delegate kernels are re-initialized
  // every time tensors are resized.
  // 2. Enabling this flag adds some overhead to AllocateTensors(), since extra
  // work is required to prepare the original execution plan.
  // 3. This flag requires that the original execution plan only have ops with
  // valid registrations (and not 'dummy' custom ops like with Flex).
  // WARNING: This feature is experimental and subject to change.
  kTfLiteDelegateFlagsRequirePropagatedShapes = 2
} TfLiteDelegateFlags;

// WARNING: This is an experimental interface that is subject to change.
typedef struct TfLiteDelegate {
  // Data that delegate needs to identify itself. This data is owned by the
  // delegate. The delegate is owned in the user code, so the delegate is
  // responsible for doing this when it is destroyed.
  void* data_;

  // Invoked by ModifyGraphWithDelegate. This prepare is called, giving the
  // delegate a view of the current graph through TfLiteContext*. It typically
  // will look at the nodes and call ReplaceNodeSubsetsWithDelegateKernels()
  // to ask the TensorFlow lite runtime to create macro-nodes to represent
  // delegated subgraphs of the original graph.
  TfLiteStatus (*Prepare)(struct TfLiteContext* context,
                          struct TfLiteDelegate* delegate);

  // Copy the data from delegate buffer handle into raw memory of the given
  // 'tensor'. Note that the delegate is allowed to allocate the raw bytes as
  // long as it follows the rules for kTfLiteDynamic tensors, in which case this
  // cannot be null.
  TfLiteStatus (*CopyFromBufferHandle)(struct TfLiteContext* context,
                                       struct TfLiteDelegate* delegate,
                                       TfLiteBufferHandle buffer_handle,
                                       TfLiteTensor* tensor);

  // Copy the data from raw memory of the given 'tensor' to delegate buffer
  // handle. This can be null if the delegate doesn't use its own buffer.
  TfLiteStatus (*CopyToBufferHandle)(struct TfLiteContext* context,
                                     struct TfLiteDelegate* delegate,
                                     TfLiteBufferHandle buffer_handle,
                                     TfLiteTensor* tensor);

  // Free the Delegate Buffer Handle. Note: This only frees the handle, but
  // this doesn't release the underlying resource (e.g. textures). The
  // resources are either owned by application layer or the delegate.
  // This can be null if the delegate doesn't use its own buffer.
  void (*FreeBufferHandle)(struct TfLiteContext* context,
                           struct TfLiteDelegate* delegate,
                           TfLiteBufferHandle* handle);

  // Bitmask flags. See the comments in `TfLiteDelegateFlags`.
  int64_t flags;
} TfLiteDelegate;

// Build a 'null' delegate, with all the fields properly set to their default
// values.
TfLiteDelegate TfLiteDelegateCreate(void);

// MARK: - c_api.h

// --------------------------------------------------------------------------
/// C API for TensorFlow Lite.
///
/// The API leans towards simplicity and uniformity instead of convenience, as
/// most usage will be by language-specific wrappers. It provides largely the
/// same set of functionality as that of the C++ TensorFlow Lite `Interpreter`
/// API, but is useful for shared libraries where having a stable ABI boundary
/// is important.
///
/// Conventions:
/// * We use the prefix TfLite for everything in the API.
/// * size_t is used to represent byte sizes of objects that are
///   materialized in the address space of the calling process.
/// * int is used as an index into arrays.
///
/// Usage:
/// <pre><code>
/// // Create the model and interpreter options.
/// TfLiteModel* model = TfLiteModelCreateFromFile("/path/to/model.tflite");
/// TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
/// TfLiteInterpreterOptionsSetNumThreads(options, 2);
///
/// // Create the interpreter.
/// TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
///
/// // Allocate tensors and populate the input tensor data.
/// TfLiteInterpreterAllocateTensors(interpreter);
/// TfLiteTensor* input_tensor =
///     TfLiteInterpreterGetInputTensor(interpreter, 0);
/// TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
///                            input.size() * sizeof(float));
///
/// // Execute inference.
/// TfLiteInterpreterInvoke(interpreter);
///
/// // Extract the output tensor data.
/// const TfLiteTensor* output_tensor =
//      TfLiteInterpreterGetOutputTensor(interpreter, 0);
/// TfLiteTensorCopyToBuffer(output_tensor, output.data(),
///                          output.size() * sizeof(float));
///
/// // Dispose of the model and interpreter objects.
/// TfLiteInterpreterDelete(interpreter);
/// TfLiteInterpreterOptionsDelete(options);
/// TfLiteModelDelete(model);


// --------------------------------------------------------------------------
// Opaque types used by the C API.

// Allows delegation of nodes to alternative backends.
//typedef struct TfLiteDelegate TfLiteDelegate;


// TfLiteModel wraps a loaded TensorFlow Lite model.
typedef struct TfLiteModel TfLiteModel;

// TfLiteInterpreterOptions allows customized interpreter configuration.
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;


// TfLiteInterpreter provides inference from a provided model.
typedef struct TfLiteInterpreter TfLiteInterpreter;

//// A tensor in the interpreter system which is a wrapper around a buffer of
//// data including a dimensionality (or NULL if not currently defined).
//typedef struct TfLiteTensor TfLiteTensor;
//


// Returns a model from the provided buffer, or null on failure.
//
// NOTE: The caller retains ownership of the `model_data` and should ensure that
// the lifetime of the `model_data` must be at least as long as the lifetime
// of the `TfLiteModel`.
 extern TfLiteModel* TfLiteModelCreate(const void* model_data,
                                                      size_t model_size);

// Returns a model from the provided file, or null on failure.
 extern TfLiteModel* TfLiteModelCreateFromFile(
    const char* model_path);

// Destroys the model instance.
 extern void TfLiteModelDelete(struct TfLiteModel* model);

// Returns a new interpreter options instances.
 extern TfLiteInterpreterOptions*
TfLiteInterpreterOptionsCreate();

// Destroys the interpreter options instance.
 extern void TfLiteInterpreterOptionsDelete(
    TfLiteInterpreterOptions* options);

// Sets the number of CPU threads to use for the interpreter.
 extern void TfLiteInterpreterOptionsSetNumThreads(
    TfLiteInterpreterOptions* options, int32_t num_threads);

// Adds a delegate to be applied during `TfLiteInterpreter` creation.
//
// If delegate application fails, interpreter creation will also fail with an
// associated error logged.
//
// NOTE: The caller retains ownership of the delegate and should ensure that it
// remains valid for the duration of any created interpreter's lifetime.
 extern void TfLiteInterpreterOptionsAddDelegate(
    TfLiteInterpreterOptions* options, struct TfLiteDelegate* delegate);

// Sets a custom error reporter for interpreter execution.
//
// * `reporter` takes the provided `user_data` object, as well as a C-style
//   format string and arg list (see also vprintf).
// * `user_data` is optional. If non-null, it is owned by the client and must
//   remain valid for the duration of the interpreter lifetime.
 extern void TfLiteInterpreterOptionsSetErrorReporter(
    struct TfLiteInterpreterOptions* options,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data);

// Returns a new interpreter using the provided model and options, or null on
// failure.
//
// * `model` must be a valid model instance. The caller retains ownership of the
//   object, and can destroy it immediately after creating the interpreter; the
//   interpreter will maintain its own reference to the underlying model data.
// * `optional_options` may be null. The caller retains ownership of the object,
//   and can safely destroy it immediately after creating the interpreter.
//
// NOTE: The client *must* explicitly allocate tensors before attempting to
// access input tensor data or invoke the interpreter.
 extern TfLiteInterpreter* TfLiteInterpreterCreate(
    struct TfLiteModel* model, struct TfLiteInterpreterOptions* optional_options);

// Destroys the interpreter.
 extern void TfLiteInterpreterDelete(
    struct TfLiteInterpreter* interpreter);

// Returns the number of input tensors associated with the model.
 extern int32_t TfLiteInterpreterGetInputTensorCount(
    struct TfLiteInterpreter* interpreter);

// Returns the tensor associated with the input index.
// REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
 extern TfLiteTensor* TfLiteInterpreterGetInputTensor(
    struct TfLiteInterpreter* interpreter, int32_t input_index);

// Resizes the specified input tensor.
//
// NOTE: After a resize, the client *must* explicitly allocate tensors before
// attempting to access the resized tensor data or invoke the interpreter.
//
// REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
//
// This function makes a copy of the input dimensions, so the client can safely
// deallocate `input_dims` immediately after this function returns.
 extern TfLiteStatus TfLiteInterpreterResizeInputTensor(
    struct TfLiteInterpreter* interpreter, int32_t input_index, const int* input_dims,
    int32_t input_dims_size);

// Updates allocations for all tensors, resizing dependent tensors using the
// specified input tensor dimensionality.
//
// This is a relatively expensive operation, and need only be called after
// creating the graph and/or resizing any inputs.
 extern TfLiteStatus TfLiteInterpreterAllocateTensors(
    struct TfLiteInterpreter* interpreter);

// Runs inference for the loaded graph.
//
// Before calling this function, the caller should first invoke
// TfLiteInterpreterAllocateTensors() and should also set the values for the
// input tensors.  After successfully calling this function, the values for the
// output tensors will be set.
//
// NOTE: It is possible that the interpreter is not in a ready state to
// evaluate (e.g., if AllocateTensors() hasn't been called, or if a
// ResizeInputTensor() has been performed without a subsequent call to
// AllocateTensors()).
//
//   If the (experimental!) delegate fallback option was enabled in the
//   interpreter options, then the interpreter will automatically fall back to
//   not using any delegates if execution with delegates fails. For details, see
//   TfLiteInterpreterOptionsSetEnableDelegateFallback in c_api_experimental.h.
//
// Returns one of the following status codes:
//  - kTfLiteOk: Success. Output is valid.
//  - kTfLiteDelegateError: Execution with delegates failed, due to a problem
//    with the delegate(s). If fallback was not enabled, output is invalid.
//    If fallback was enabled, this return value indicates that fallback
//    succeeded, the output is valid, and all delegates previously applied to
//    the interpreter have been undone.
//  - kTfLiteApplicationError: Same as for kTfLiteDelegateError, except that
//    the problem was not with the delegate itself, but rather was
//    due to an incompatibility between the delegate(s) and the
//    interpreter or model.
//  - kTfLiteError: Unexpected/runtime failure. Output is invalid.

 extern TfLiteStatus TfLiteInterpreterInvoke(
    struct TfLiteInterpreter* interpreter);

// Returns the number of output tensors associated with the model.
 extern int32_t TfLiteInterpreterGetOutputTensorCount(
    struct TfLiteInterpreter* interpreter);

// Returns the tensor associated with the output index.
// REQUIRES: 0 <= output_index < TfLiteInterpreterGetOutputTensorCount(tensor)
//
// NOTE: The shape and underlying data buffer for output tensors may be not
// be available until after the output tensor has been both sized and allocated.
// In general, best practice is to interact with the output tensor *after*
// calling TfLiteInterpreterInvoke().
 extern const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    struct TfLiteInterpreter* interpreter, int32_t output_index);

// --------------------------------------------------------------------------
// TfLiteTensor wraps data associated with a graph tensor.
//
// Note that, while the TfLiteTensor struct is not currently opaque, and its
// fields can be accessed directly, these methods are still convenient for
// language bindings. In the future the tensor struct will likely be made opaque
// in the public API.

// Returns the type of a tensor element.
// extern TfLiteType TfLiteTensorType(const TfLiteTensor* tensor);
extern TfLiteType TfLiteTensorType(struct TfLiteTensor* tensor);

// Returns the number of dimensions that the tensor has.
 extern int32_t TfLiteTensorNumDims(struct TfLiteTensor* tensor);

// Returns the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TFLiteTensorNumDims(tensor)
 extern int32_t TfLiteTensorDim(struct TfLiteTensor* tensor,
                                               int32_t dim_index);

// Returns the size of the underlying data in bytes.
 extern size_t TfLiteTensorByteSize(struct TfLiteTensor* tensor);

// Returns a pointer to the underlying data buffer.
//
// NOTE: The result may be null if tensors have not yet been allocated, e.g.,
// if the Tensor has just been created or resized and `TfLiteAllocateTensors()`
// has yet to be called, or if the output tensor is dynamically sized and the
// interpreter hasn't been invoked.
 extern void* TfLiteTensorData(struct TfLiteTensor* tensor);

// Returns the (null-terminated) name of the tensor.
 extern const char* TfLiteTensorName(struct TfLiteTensor* tensor);

// Returns the parameters for asymmetric quantization. The quantization
// parameters are only valid when the tensor type is `kTfLiteUInt8` and the
// `scale != 0`. Quantized values can be converted back to float using:
//    real_value = scale * (quantized_value - zero_point);
 extern TfLiteQuantizationParams TfLiteTensorQuantizationParams(
    struct TfLiteTensor* tensor);

// Copies from the provided input buffer into the tensor's buffer.
// REQUIRES: input_data_size == TfLiteTensorByteSize(tensor)
 extern TfLiteStatus TfLiteTensorCopyFromBuffer(struct
    TfLiteTensor* tensor, const void* input_data, size_t input_data_size);

// Copies to the provided output buffer from the tensor's buffer.
// REQUIRES: output_data_size == TfLiteTensorByteSize(tensor)
extern TfLiteStatus TfLiteTensorCopyToBuffer(
    struct TfLiteTensor* output_tensor, void* output_data,
    size_t output_data_size);



// MARK: - builtin_ops.h

// The enum for builtin operators.
// Note: CUSTOM, DELEGATE, and PLACEHOLDER_FOR_GREATER_OP_CODES are 3 special
// ops which are not real built-in ops.
typedef enum {
  kTfLiteBuiltinAdd = 0,
  kTfLiteBuiltinAveragePool2d = 1,
  kTfLiteBuiltinConcatenation = 2,
  kTfLiteBuiltinConv2d = 3,
  kTfLiteBuiltinDepthwiseConv2d = 4,
  kTfLiteBuiltinDepthToSpace = 5,
  kTfLiteBuiltinDequantize = 6,
  kTfLiteBuiltinEmbeddingLookup = 7,
  kTfLiteBuiltinFloor = 8,
  kTfLiteBuiltinFullyConnected = 9,
  kTfLiteBuiltinHashtableLookup = 10,
  kTfLiteBuiltinL2Normalization = 11,
  kTfLiteBuiltinL2Pool2d = 12,
  kTfLiteBuiltinLocalResponseNormalization = 13,
  kTfLiteBuiltinLogistic = 14,
  kTfLiteBuiltinLshProjection = 15,
  kTfLiteBuiltinLstm = 16,
  kTfLiteBuiltinMaxPool2d = 17,
  kTfLiteBuiltinMul = 18,
  kTfLiteBuiltinRelu = 19,
  kTfLiteBuiltinReluN1To1 = 20,
  kTfLiteBuiltinRelu6 = 21,
  kTfLiteBuiltinReshape = 22,
  kTfLiteBuiltinResizeBilinear = 23,
  kTfLiteBuiltinRnn = 24,
  kTfLiteBuiltinSoftmax = 25,
  kTfLiteBuiltinSpaceToDepth = 26,
  kTfLiteBuiltinSvdf = 27,
  kTfLiteBuiltinTanh = 28,
  kTfLiteBuiltinConcatEmbeddings = 29,
  kTfLiteBuiltinSkipGram = 30,
  kTfLiteBuiltinCall = 31,
  kTfLiteBuiltinCustom = 32,
  kTfLiteBuiltinEmbeddingLookupSparse = 33,
  kTfLiteBuiltinPad = 34,
  kTfLiteBuiltinUnidirectionalSequenceRnn = 35,
  kTfLiteBuiltinGather = 36,
  kTfLiteBuiltinBatchToSpaceNd = 37,
  kTfLiteBuiltinSpaceToBatchNd = 38,
  kTfLiteBuiltinTranspose = 39,
  kTfLiteBuiltinMean = 40,
  kTfLiteBuiltinSub = 41,
  kTfLiteBuiltinDiv = 42,
  kTfLiteBuiltinSqueeze = 43,
  kTfLiteBuiltinUnidirectionalSequenceLstm = 44,
  kTfLiteBuiltinStridedSlice = 45,
  kTfLiteBuiltinBidirectionalSequenceRnn = 46,
  kTfLiteBuiltinExp = 47,
  kTfLiteBuiltinTopkV2 = 48,
  kTfLiteBuiltinSplit = 49,
  kTfLiteBuiltinLogSoftmax = 50,
  kTfLiteBuiltinDelegate = 51,
  kTfLiteBuiltinBidirectionalSequenceLstm = 52,
  kTfLiteBuiltinCast = 53,
  kTfLiteBuiltinPrelu = 54,
  kTfLiteBuiltinMaximum = 55,
  kTfLiteBuiltinArgMax = 56,
  kTfLiteBuiltinMinimum = 57,
  kTfLiteBuiltinLess = 58,
  kTfLiteBuiltinNeg = 59,
  kTfLiteBuiltinPadv2 = 60,
  kTfLiteBuiltinGreater = 61,
  kTfLiteBuiltinGreaterEqual = 62,
  kTfLiteBuiltinLessEqual = 63,
  kTfLiteBuiltinSelect = 64,
  kTfLiteBuiltinSlice = 65,
  kTfLiteBuiltinSin = 66,
  kTfLiteBuiltinTransposeConv = 67,
  kTfLiteBuiltinSparseToDense = 68,
  kTfLiteBuiltinTile = 69,
  kTfLiteBuiltinExpandDims = 70,
  kTfLiteBuiltinEqual = 71,
  kTfLiteBuiltinNotEqual = 72,
  kTfLiteBuiltinLog = 73,
  kTfLiteBuiltinSum = 74,
  kTfLiteBuiltinSqrt = 75,
  kTfLiteBuiltinRsqrt = 76,
  kTfLiteBuiltinShape = 77,
  kTfLiteBuiltinPow = 78,
  kTfLiteBuiltinArgMin = 79,
  kTfLiteBuiltinFakeQuant = 80,
  kTfLiteBuiltinReduceProd = 81,
  kTfLiteBuiltinReduceMax = 82,
  kTfLiteBuiltinPack = 83,
  kTfLiteBuiltinLogicalOr = 84,
  kTfLiteBuiltinOneHot = 85,
  kTfLiteBuiltinLogicalAnd = 86,
  kTfLiteBuiltinLogicalNot = 87,
  kTfLiteBuiltinUnpack = 88,
  kTfLiteBuiltinReduceMin = 89,
  kTfLiteBuiltinFloorDiv = 90,
  kTfLiteBuiltinReduceAny = 91,
  kTfLiteBuiltinSquare = 92,
  kTfLiteBuiltinZerosLike = 93,
  kTfLiteBuiltinFill = 94,
  kTfLiteBuiltinFloorMod = 95,
  kTfLiteBuiltinRange = 96,
  kTfLiteBuiltinResizeNearestNeighbor = 97,
  kTfLiteBuiltinLeakyRelu = 98,
  kTfLiteBuiltinSquaredDifference = 99,
  kTfLiteBuiltinMirrorPad = 100,
  kTfLiteBuiltinAbs = 101,
  kTfLiteBuiltinSplitV = 102,
  kTfLiteBuiltinUnique = 103,
  kTfLiteBuiltinCeil = 104,
  kTfLiteBuiltinReverseV2 = 105,
  kTfLiteBuiltinAddN = 106,
  kTfLiteBuiltinGatherNd = 107,
  kTfLiteBuiltinCos = 108,
  kTfLiteBuiltinWhere = 109,
  kTfLiteBuiltinRank = 110,
  kTfLiteBuiltinElu = 111,
  kTfLiteBuiltinReverseSequence = 112,
  kTfLiteBuiltinMatrixDiag = 113,
  kTfLiteBuiltinQuantize = 114,
  kTfLiteBuiltinMatrixSetDiag = 115,
  kTfLiteBuiltinRound = 116,
  kTfLiteBuiltinHardSwish = 117,
  kTfLiteBuiltinIf = 118,
  kTfLiteBuiltinWhile = 119,
  kTfLiteBuiltinNonMaxSuppressionV4 = 120,
  kTfLiteBuiltinNonMaxSuppressionV5 = 121,
  kTfLiteBuiltinScatterNd = 122,
  kTfLiteBuiltinSelectV2 = 123,
  kTfLiteBuiltinDensify = 124,
  kTfLiteBuiltinSegmentSum = 125,
  kTfLiteBuiltinBatchMatmul = 126,
  kTfLiteBuiltinPlaceholderForGreaterOpCodes = 127,
  kTfLiteBuiltinCumsum = 128,
  kTfLiteBuiltinCallOnce = 129,
  kTfLiteBuiltinBroadcastTo = 130,
  kTfLiteBuiltinRfft2d = 131,
  kTfLiteBuiltinConv3d = 132,
  kTfLiteBuiltinImag = 133,
  kTfLiteBuiltinReal = 134,
  kTfLiteBuiltinComplexAbs = 135,
  kTfLiteBuiltinHashtable = 136,
  kTfLiteBuiltinHashtableFind = 137,
  kTfLiteBuiltinHashtableImport = 138,
  kTfLiteBuiltinHashtableSize = 139,
  kTfLiteBuiltinReduceAll = 140,
  kTfLiteBuiltinConv3dTranspose = 141,
  kTfLiteBuiltinVarHandle = 142,
  kTfLiteBuiltinReadVariable = 143,
  kTfLiteBuiltinAssignVariable = 144,
  kTfLiteBuiltinBroadcastArgs = 145,
  kTfLiteBuiltinRandomStandardNormal = 146,
} TfLiteBuiltinOperator;

// MARK: - c_api_experimental.h

/// Resets all variable tensors to zero.
///
/// WARNING: This is an experimental API and subject to change.
 extern TfLiteStatus TfLiteInterpreterResetVariableTensors(
    struct TfLiteInterpreter* interpreter);

/// Adds an op registration for a builtin operator.
///
/// Op registrations are used to map ops referenced in the flatbuffer model
/// to executable function pointers (`TfLiteRegistration`s).
///
/// NOTE: The interpreter will make a shallow copy of `registration` internally,
/// so the caller should ensure that its contents (function pointers, etc...)
/// remain valid for the duration of the interpreter's lifetime. A common
/// practice is making the provided `TfLiteRegistration` instance static.
///
/// Code that uses this function should NOT call
/// `TfLiteInterpreterOptionsSetOpResolver` on the same options object.
///
/// WARNING: This is an experimental API and subject to change.
 void TfLiteInterpreterOptionsAddBuiltinOp(
    struct TfLiteInterpreterOptions* options, TfLiteBuiltinOperator op,
    struct TfLiteRegistration* registration, int32_t min_version,
    int32_t max_version);

/// Adds an op registration for a custom operator.
///
/// Op registrations are used to map ops referenced in the flatbuffer model
/// to executable function pointers (`TfLiteRegistration`s).
///
/// NOTE: The interpreter will make a shallow copy of `registration` internally,
/// so the caller should ensure that its contents (function pointers, etc...)
/// remain valid for the duration of any created interpreter's lifetime. A
/// common practice is making the provided `TfLiteRegistration` instance static.
///
/// The lifetime of the string pointed to by `name` must be at least as long
/// as the lifetime of the `TfLiteInterpreterOptions`.
///
/// Code that uses this function should NOT call
/// `TfLiteInterpreterOptionsSetOpResolver` on the same options object.
///
/// WARNING: This is an experimental API and subject to change.
 void TfLiteInterpreterOptionsAddCustomOp(
    struct TfLiteInterpreterOptions* options, const char* name,
    struct TfLiteRegistration* registration, int32_t min_version,
    int32_t max_version);

/// Registers callbacks for resolving builtin or custom operators.
///
/// The `TfLiteInterpreterOptionsSetOpResolver` function provides an alternative
/// method for registering builtin ops and/or custom ops, by providing operator
/// resolver callbacks.  Unlike using `TfLiteInterpreterOptionsAddBuiltinOp`
/// and/or `TfLiteInterpreterOptionsAddAddCustomOp`, these let you register all
/// the operators in a single call.
///
/// Code that uses this function should NOT call
/// `TfLiteInterpreterOptionsAddBuiltin` or
/// `TfLiteInterpreterOptionsAddCustomOp` on the same options object.
///
/// If `op_resolver_user_data` is non-null, its lifetime must be at least as
/// long as the lifetime of the `TfLiteInterpreterOptions`.
///
/// WARNING: This is an experimental API and subject to change.
void TfLiteInterpreterOptionsSetOpResolver(
    struct TfLiteInterpreterOptions* options,
    struct TfLiteRegistration* (*find_builtin_op)(void* user_data,
                                                 TfLiteBuiltinOperator op,
                                                 int version),
    struct TfLiteRegistration* (*find_custom_op)(void* user_data,
                                                const char* custom_op,
                                                int version),
    void* op_resolver_user_data);

/// Returns a new interpreter using the provided model and options, or null on
/// failure, where the model uses only the operators explicitly added to the
/// options.  This is the same as `TFLiteInterpreterCreate` from `c_api.h`,
/// except that the only operators that are supported are the ones registered
/// in `options` via calls to `TfLiteInterpreterOptionsSetOpResolver`,
/// `TfLiteInterpreterOptionsAddBuiltinOp`, and/or
/// `TfLiteInterpreterOptionsAddCustomOp`.
///
/// * `model` must be a valid model instance. The caller retains ownership of
///   the object, and can destroy it immediately after creating the interpreter;
///   the interpreter will maintain its own reference to the underlying model
///   data.
/// * `options` should not be null. The caller retains ownership of the object,
///   and can safely destroy it immediately after creating the interpreter.
///
/// NOTE: The client *must* explicitly allocate tensors before attempting to
/// access input tensor data or invoke the interpreter.
///
/// WARNING: This is an experimental API and subject to change.
 extern TfLiteInterpreter*
TfLiteInterpreterCreateWithSelectedOps(struct TfLiteModel* model,
                                       struct TfLiteInterpreterOptions* options);

/// Enable or disable the NN API delegate for the interpreter (true to enable).
///
/// WARNING: This is an experimental API and subject to change.
 extern void TfLiteInterpreterOptionsSetUseNNAPI(
    struct TfLiteInterpreterOptions* options, bool enable);

/// Enable or disable CPU fallback for the interpreter (true to enable).
/// If enabled, TfLiteInterpreterInvoke will do automatic fallback from
/// executing with delegate(s) to regular execution without delegates
/// (i.e. on CPU).
///
/// Allowing the fallback is suitable only if both of the following hold:
/// - The caller is known not to cache pointers to tensor data across
///   TfLiteInterpreterInvoke calls.
/// - The model is not stateful (no variables, no LSTMs) or the state isn't
///   needed between batches.
///
/// When delegate fallback is enabled, TfLiteInterpreterInvoke will
/// behave as follows:
///   If one or more delegates were set in the interpreter options
///   (see TfLiteInterpreterOptionsAddDelegate),
///   AND inference fails,
///   then the interpreter will fall back to not using any delegates.
///   In that case, the previously applied delegate(s) will be automatically
///   undone, and an attempt will be made to return the interpreter to an
///   invokable state, which may invalidate previous tensor addresses,
///   and the inference will be attempted again, using input tensors with
///   the same value as previously set.
///
/// WARNING: This is an experimental API and subject to change.
 extern void TfLiteInterpreterOptionsSetEnableDelegateFallback(
    struct TfLiteInterpreterOptions* options, bool enable);

// Set if buffer handle output is allowed.
//
/// When using hardware delegation, Interpreter will make the data of output
/// tensors available in `tensor->data` by default. If the application can
/// consume the buffer handle directly (e.g. reading output from OpenGL
/// texture), it can set this flag to false, so Interpreter won't copy the
/// data from buffer handle to CPU memory. WARNING: This is an experimental
/// API and subject to change.
 extern void TfLiteSetAllowBufferHandleOutput(
    struct TfLiteInterpreter* interpreter, bool allow_buffer_handle_output);

/// Allow a delegate to look at the graph and modify the graph to handle
/// parts of the graph themselves. After this is called, the graph may
/// contain new nodes that replace 1 more nodes.
/// 'delegate' must outlive the interpreter.
/// Use `TfLiteInterpreterOptionsAddDelegate` instead of this unless
/// absolutely required.
/// Returns one of the following three status codes:
/// 1. kTfLiteOk: Success.
/// 2. kTfLiteDelegateError: Delegation failed due to an error in the
/// delegate. The Interpreter has been restored to its pre-delegation state.
/// NOTE: This undoes all delegates previously applied to the Interpreter.
/// 3. kTfLiteError: Unexpected/runtime failure.
/// WARNING: This is an experimental API and subject to change.
 extern TfLiteStatus TfLiteInterpreterModifyGraphWithDelegate(
    struct TfLiteInterpreter* interpreter, struct TfLiteDelegate* delegate);

/// Returns the tensor index corresponding to the input tensor
///
/// WARNING: This is an experimental API and subject to change.
 extern int32_t TfLiteInterpreterGetInputTensorIndex(
    struct TfLiteInterpreter* interpreter, int32_t input_index);

/// Returns the tensor index corresponding to the output tensor
///
/// WARNING: This is an experimental API and subject to change.
 extern int32_t TfLiteInterpreterGetOutputTensorIndex(
    struct TfLiteInterpreter* interpreter, int32_t output_index);

// MARK: - xnnpack_delegate.h
typedef struct TfLiteXNNPackDelegateOptions {
  // Number of threads to use in the thread pool.
  // 0 or negative value means no thread pool used.
  int32_t num_threads;
} TfLiteXNNPackDelegateOptions;

// Returns a structure with the default XNNPack delegate options.
 TfLiteXNNPackDelegateOptions
TfLiteXNNPackDelegateOptionsDefault();

// Creates a new delegate instance that need to be destroyed with
// `TfLiteXNNPackDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the following default values are used:
 TfLiteDelegate* TfLiteXNNPackDelegateCreate(
    struct TfLiteXNNPackDelegateOptions* options);

// Returns the pthreadpool_t object used for parallelization in XNNPACK.
// Can return NULL if the XNNPack delegate is single-threaded.
//
// WARNING: This API is experimental and subject to change.
 void* TfLiteXNNPackDelegateGetThreadPool(struct 
    TfLiteDelegate* delegate);

// Destroys a delegate created with `TfLiteXNNPackDelegateCreate` call.
 void TfLiteXNNPackDelegateDelete(struct TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

