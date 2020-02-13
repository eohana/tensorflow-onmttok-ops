#include "base_op.h"

#include <vector>

using namespace tensorflow;

class DetokenizeOp : public BaseOp {
public:
    explicit DetokenizeOp(OpKernelConstruction *ctx) : BaseOp(ctx) {
    }

    void Compute(OpKernelContext *ctx) override {
        const Tensor *input_tensor;
        OP_REQUIRES_OK(ctx, ctx->input("tokens", &input_tensor));
        const auto &input_flat = input_tensor->flat<tstring>();

        std::vector<tstring> tokens;
        for (int i = 0; i < input_flat.size(); ++i) {
            tokens.emplace_back(input_flat(i).data());
        }

        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("text", TensorShape({1,}), &output_tensor));
        auto output = output_tensor->flat<tstring>();

        output(0) = tokenizer_.detokenize(tokens);
    }
};

REGISTER_KERNEL_BUILDER(Name("Detokenize").Device(DEVICE_CPU), DetokenizeOp);
