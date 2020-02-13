#include "base_op.h"

#include <string>
#include <vector>

using namespace onmt;
using namespace tensorflow;

BaseOp::BaseOp(OpKernelConstruction *ctx) : OpKernel(ctx),
                                            tokenizer_(Tokenizer(Tokenizer::Mode::None)) {
    std::string mode, joiner;
    std::vector<std::string> segment_alphabet;
    bool no_substitution,
            case_feature,
            case_markup,
            soft_case_regions,
            joiner_annotate,
            joiner_new,
            spacer_annotate,
            spacer_new,
            preserve_placeholders,
            preserve_segmented_tokens,
            support_prior_joiners,
            segment_case,
            segment_numbers,
            segment_alphabet_change;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("no_substitution", &no_substitution));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("case_feature", &case_feature));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("case_markup", &case_markup));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("soft_case_regions", &soft_case_regions));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("joiner_annotate", &joiner_annotate));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("joiner", &joiner));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("joiner_new", &joiner_new));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("spacer_annotate", &spacer_annotate));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("spacer_new", &spacer_new));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preserve_placeholders", &preserve_placeholders));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preserve_segmented_tokens", &preserve_segmented_tokens));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("support_prior_joiners", &support_prior_joiners));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("segment_case", &segment_case));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("segment_numbers", &segment_numbers));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("segment_alphabet", &segment_alphabet));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("segment_alphabet_change", &segment_alphabet_change));

    tokenizer_ = Tokenizer(Tokenizer::str_to_mode(mode),
                           build_flags_list(no_substitution,
                                            case_feature,
                                            case_markup,
                                            soft_case_regions,
                                            joiner_annotate,
                                            joiner_new,
                                            spacer_annotate,
                                            spacer_new,
                                            preserve_placeholders,
                                            preserve_segmented_tokens,
                                            support_prior_joiners,
                                            segment_case,
                                            segment_numbers,
                                            segment_alphabet_change),
                           "",
                           joiner);

    for (const auto &seg : segment_alphabet) {
        tokenizer_.add_alphabet_to_segment(seg);
    }
}

int BaseOp::build_flags_list(const bool &no_substitution,
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
                             const bool &segment_alphabet_change) {
    int flags = 0;

    if (no_substitution)
        flags |= Tokenizer::Flags::NoSubstitution;
    if (case_feature)
        flags |= Tokenizer::Flags::CaseFeature;
    if (case_markup)
        flags |= Tokenizer::Flags::CaseMarkup;
    if (soft_case_regions)
        flags |= Tokenizer::Flags::SoftCaseRegions;
    if (joiner_annotate)
        flags |= Tokenizer::Flags::JoinerAnnotate;
    if (joiner_new)
        flags |= Tokenizer::Flags::JoinerNew;
    if (spacer_annotate)
        flags |= Tokenizer::Flags::SpacerAnnotate;
    if (spacer_new)
        flags |= Tokenizer::Flags::SpacerNew;
    if (preserve_placeholders)
        flags |= Tokenizer::Flags::PreservePlaceholders;
    if (preserve_segmented_tokens)
        flags |= Tokenizer::Flags::PreserveSegmentedTokens;
    if (support_prior_joiners)
        flags |= Tokenizer::Flags::SupportPriorJoiners;
    if (segment_case)
        flags |= Tokenizer::Flags::SegmentCase;
    if (segment_numbers)
        flags |= Tokenizer::Flags::SegmentNumbers;
    if (segment_alphabet_change)
        flags |= Tokenizer::Flags::SegmentAlphabetChange;

    return flags;
}
