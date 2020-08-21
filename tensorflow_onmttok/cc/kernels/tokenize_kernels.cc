#include "base_op.h"

#include <algorithm>
#include <vector>

using namespace tensorflow;

class TokenizeOp : public BaseOp {
public:
    explicit TokenizeOp(OpKernelConstruction *ctx) : BaseOp(ctx) {
    }

    void Compute(OpKernelContext *ctx) override {
        const Tensor *input_tensor;
        OP_REQUIRES_OK(ctx, ctx->input("text", &input_tensor));
        const auto &input_flat = input_tensor->flat<tstring>();

        std::vector<std::string> tokens;
        for (int i = 0; i < input_flat.size(); ++i) {
            tokenizer_.tokenize(input_flat(i).data(), tokens);
        }

        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("tokens", TensorShape({(int) tokens.size(),}), &output_tensor));
        auto output = output_tensor->flat<tstring>();

        std::copy_n(tokens.begin(), tokens.size(), output.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("Tokenize").Device(DEVICE_CPU), TokenizeOp);
