from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"unexpected Mozc source while patching {label}")
    return text.replace(old, new, 1)


def patch_checkout(checkout: Path) -> None:
    header = checkout / "src/prediction/realtime_decoder.h"
    text = header.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "  bool PushBackTopConversionResult(const ConversionRequest& request,\n"
        "                                   std::vector<Result>* results) const;\n",
        "  bool PushBackTopConversionResult(const ConversionRequest& request,\n"
        "                                   std::vector<Result>* results,\n"
        "                                   bool append_result) const;\n",
        "realtime decoder declaration",
    )
    header.write_text(text, encoding="utf-8", newline="\n")

    source = checkout / "src/prediction/realtime_decoder.cc"
    text = source.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "bool RealtimeDecoder::PushBackTopConversionResult(\n"
        "    const ConversionRequest& request, std::vector<Result>* results) const {\n",
        "bool RealtimeDecoder::PushBackTopConversionResult(\n"
        "    const ConversionRequest& request, std::vector<Result>* results,\n"
        "    bool append_result) const {\n",
        "realtime decoder definition",
    )
    text = replace_once(
        text,
        "  result.SetTypesAndTokenAttributes(REALTIME | REALTIME_TOP,\n"
        "                                    dictionary::Token::NONE);\n"
        "  result.attributes |= Attribute::NO_VARIANTS_EXPANSION;\n"
        "\n"
        "  results->emplace_back(std::move(result));\n",
        "  if (!append_result) {\n"
        "    // The converter was run only to let the AI rewriter prefetch its\n"
        "    // context and candidate vectors.  The immutable-converter results\n"
        "    // below remain the visible prediction list.\n"
        "    return true;\n"
        "  }\n"
        "\n"
        "  result.SetTypesAndTokenAttributes(REALTIME | REALTIME_TOP,\n"
        "                                    dictionary::Token::NONE);\n"
        "  result.attributes |= Attribute::NO_VARIANTS_EXPANSION;\n"
        "\n"
        "  results->emplace_back(std::move(result));\n",
        "realtime converter result suppression",
    )
    text = replace_once(
        text,
        "  if (request.options().use_actual_converter_for_realtime_conversion &&\n"
        "      request.request_type() != ConversionRequest::PARTIAL_SUGGESTION &&\n"
        "      request.request_type() != ConversionRequest::PARTIAL_PREDICTION) {\n"
        "    if (!PushBackTopConversionResult(request_for_realtime, &results)) {\n"
        "      LOG(WARNING) << \"Realtime conversion with converter failed\";\n"
        "    }\n"
        "  }\n",
        "  const bool can_run_actual_converter =\n"
        "      request.request_type() != ConversionRequest::PARTIAL_SUGGESTION &&\n"
        "      request.request_type() != ConversionRequest::PARTIAL_PREDICTION;\n"
        "  const bool suppressed_by_experiment = request.request()\n"
        "      .decoder_experiment_params()\n"
        "      .suppress_realtime_conversion_with_converter();\n"
        "  const bool run_for_ai_prefetch =\n"
        "      request.config().use_realtime_conversion() &&\n"
        "      !suppressed_by_experiment;\n"
        "  if (can_run_actual_converter &&\n"
        "      (request.options().use_actual_converter_for_realtime_conversion ||\n"
        "       run_for_ai_prefetch)) {\n"
        "    if (!PushBackTopConversionResult(\n"
        "            request_for_realtime, &results,\n"
        "            request.options().use_actual_converter_for_realtime_conversion)) {\n"
        "      LOG(WARNING) << \"Realtime conversion with converter failed\";\n"
        "    }\n"
        "  }\n",
        "AI prefetch realtime converter trigger",
    )
    source.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True, type=Path)
    args = parser.parse_args()
    patch_checkout(args.checkout.resolve())


if __name__ == "__main__":
    main()
