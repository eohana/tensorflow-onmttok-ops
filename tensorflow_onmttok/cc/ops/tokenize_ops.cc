#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Tokenize")
    .Attr("mode: {'conservative', 'aggressive', 'char', 'space'}")
    .Attr("no_substitution: bool = false")
    .Attr("case_feature: bool = false")
    .Attr("case_markup: bool = false")
    .Attr("soft_case_regions: bool = false")
    .Attr("joiner_annotate: bool = false")
    .Attr("joiner: string = '￭'")
    .Attr("joiner_new: bool = false")
    .Attr("spacer_annotate: bool = false")
    .Attr("spacer_new: bool = false")
    .Attr("preserve_placeholders: bool = false")
    .Attr("preserve_segmented_tokens: bool = false")
    .Attr("support_prior_joiners: bool = false")
    .Attr("segment_case: bool = false")
    .Attr("segment_numbers: bool = false")
    .Attr("segment_alphabet: list(string) = []")
    .Attr("segment_alphabet_change: bool = false")
    .Input("text: string")
    .Output("tokens: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Vector(shape_inference::InferenceContext::kUnknownDim));
        return Status::OK();
    });
