cc_library(
    name = "onmttok_ops",
    srcs = [
        "cc/kernels/base_op.cc",
        "cc/kernels/base_op.h",
        "cc/kernels/detokenize_kernels.cc",
        "cc/kernels/tokenize_kernels.cc",
        "cc/ops/detokenize_ops.cc",
        "cc/ops/tokenize_ops.cc",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@opennmt_tokenizer//:OpenNMTTokenizer",
    ]
)
