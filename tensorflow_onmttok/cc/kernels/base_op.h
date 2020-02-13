#ifndef TENSORFLOW_ONMTTOK_BASE_OP_H_
#define TENSORFLOW_ONMTTOK_BASE_OP_H_

#include <onmt/Tokenizer.h>

#include "tensorflow/core/framework/op_kernel.h"

// Base Operation class that initialize and expose an
// onmt::Tokenizer object configured from operation attributes.
class BaseOp : public tensorflow::OpKernel {
public:
    explicit BaseOp(tensorflow::OpKernelConstruction *ctx);

protected:
    onmt::Tokenizer tokenizer_;

private:
    int build_flags_list(const bool &no_substitution,
                         const bool &case_feature,
                         const bool &case_markup,
                         const bool &soft_case_regions,
                         const bool &joiner_annotate,
                         const bool &joiner_new,
                         const bool &spacer_annotate,
                         const bool &spacer_new,
                         const bool &preserve_placeholders,
                         const bool &preserve_segmented_tokens,
                         const bool &support_prior_joiners,
                         const bool &segment_case,
                         const bool &segment_numbers,
                         const bool &segment_alphabet_change);
};

#endif // TENSORFLOW_ONMTTOK_BASE_OP_H_
