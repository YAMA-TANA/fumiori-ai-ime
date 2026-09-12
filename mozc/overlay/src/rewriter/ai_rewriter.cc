#include "rewriter/ai_rewriter.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "base/util.h"
#include "converter/attribute.h"
#include "converter/candidate.h"
#include "converter/segments.h"
#include "dictionary/dictionary_interface.h"
#include "rewriter/ai_ranker_client.h"

namespace mozc {

AiRewriter::AiRewriter(std::wstring pipe_name)
    : AiRewriter(nullptr, std::move(pipe_name)) {}

AiRewriter::AiRewriter(const dictionary::DictionaryInterface* dictionary,
                       std::wstring pipe_name)
    : dictionary_(dictionary), pipe_name_(std::move(pipe_name)) {}

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kMaxAiCandidateSurfaceLimit = 5;
constexpr size_t kNormalCandidateSurfaceLimit = 4;
constexpr size_t kSupplementalCandidateSurfaceLimit = 1;
constexpr size_t kMinInternalPhraseContextChars = 4;
constexpr double kBatchHighConfidence = 0.65;
constexpr double kRetryMinimumConfidence = 0.55;

int RemainingBudgetMs(const Clock::time_point& deadline) {
  const auto now = Clock::now();
  if (now >= deadline) return 0;
  return static_cast<int>(
      std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
          .count());
}

bool IsRerankableSegment(const converter::Segment& segment) {
  return segment.segment_type() == converter::Segment::FREE ||
         segment.segment_type() == converter::Segment::FIXED_BOUNDARY;
}

bool SegmentContainsSurface(const converter::Segment& segment,
                            absl::string_view value) {
  for (size_t i = 0; i < segment.candidates_size(); ++i) {
    if (segment.candidate(i).value == value) return true;
  }
  return false;
}

// These are deliberately small, high-value supplements.  Mozc's dictionary
// remains the source of truth for ordinary words; this table only repairs
// numeric/abbreviation spans that the converter commonly splits before the
// AI rewriter gets a chance to compare them as one phrase.
struct SupplementalSurfaceSpec {
  const char* reading;
  const char* surface;
};

const std::vector<SupplementalSurfaceSpec>& SupplementalSurfaceSpecs() {
  static const std::vector<SupplementalSurfaceSpec> kSpecs = {
      {"えーぴーあい", "API"},
      {"えーぴーあいれんけい", "API連携"},
      {"けんしん", "健診"},
      {"じゅうごにち", "15日"},
      {"にせんにじゅうろくねんど", "2026年度"},
      {"じゅっこ", "10個"},
      {"さんじ", "3時"},
      {"ごにん", "5人"},
      {"だいごい", "第5位"},
      {"いちい", "1位"},
  };
  return kSpecs;
}

std::vector<std::string> SupplementalSurfacesForReading(
    absl::string_view reading) {
  std::vector<std::string> result;
  size_t longest_bytes = 0;
  const char* selected_surface = nullptr;
  for (const SupplementalSurfaceSpec& spec : SupplementalSurfaceSpecs()) {
    const absl::string_view key(spec.reading);
    if (reading.size() < key.size() ||
        reading.substr(0, key.size()) != key || key.size() <= longest_bytes) {
      continue;
    }
    longest_bytes = key.size();
    selected_surface = spec.surface;
  }
  if (selected_surface != nullptr) {
    result.emplace_back(selected_surface);
    result.back().append(reading.substr(longest_bytes).data(),
                         reading.substr(longest_bytes).size());
  }
  return result;
}

size_t SupplementalPrefixChars(absl::string_view reading) {
  size_t longest_bytes = 0;
  for (const SupplementalSurfaceSpec& spec : SupplementalSurfaceSpecs()) {
    const absl::string_view key(spec.reading);
    if (reading.size() >= key.size() &&
        reading.substr(0, key.size()) == key) {
      longest_bytes = std::max(longest_bytes, key.size());
    }
  }
  if (longest_bytes == 0) return 0;
  return Util::CharsLen(reading.substr(0, longest_bytes));
}

void AddSupplementalCandidates(converter::Segment* segment) {
  if (segment == nullptr || segment->candidates_size() == 0) return;
  const std::vector<std::string> supplements =
      SupplementalSurfacesForReading(segment->key());
  for (const std::string& value : supplements) {
    if (SegmentContainsSurface(*segment, value)) continue;
    converter::Candidate* candidate = segment->add_candidate();
    *candidate = segment->candidate(0);
    candidate->key = std::string(segment->key());
    candidate->content_key = candidate->key;
    candidate->value = value;
    candidate->content_value = value;
    candidate->attributes |= converter::Attribute::SUPPLEMENTAL_MODEL |
                             converter::Attribute::NO_LEARNING;
    candidate->cost += 100000;
    candidate->wcost += 100000;
  }
}

std::string SegmentTopSurface(const converter::Segment& segment) {
  if (segment.candidates_size() > 0) {
    return std::string(segment.candidate(0).value);
  }
  return std::string(segment.key());
}

std::string FullReading(const Segments& segments) {
  std::string reading;
  for (const converter::Segment& segment : segments.conversion_segments()) {
    reading.append(segment.key().data(), segment.key().size());
  }
  return reading;
}

std::string ContextBeforeSegment(const Segments& segments, size_t index,
                                 absl::string_view document_prefix) {
  std::string context(document_prefix);
  for (size_t i = 0; i < index; ++i) {
    context.append(SegmentTopSurface(segments.conversion_segment(i)));
  }
  return context;
}

std::string ContextAfterSegment(const Segments& segments, size_t index,
                                absl::string_view document_suffix) {
  std::string context;
  for (size_t i = index + 1; i < segments.conversion_segments_size(); ++i) {
    context.append(SegmentTopSurface(segments.conversion_segment(i)));
  }
  context.append(document_suffix.data(), document_suffix.size());
  return context;
}

bool ParseCandidateId(absl::string_view id, size_t candidate_count,
                      size_t* index) {
  if (index == nullptr || id.size() < 2 || id[0] != 'c') return false;
  size_t parsed = 0;
  for (size_t pos = 1; pos < id.size(); ++pos) {
    const char c = id[pos];
    if (c < '0' || c > '9') return false;
    parsed = parsed * 10 + static_cast<size_t>(c - '0');
    if (parsed >= candidate_count) return false;
  }
  *index = parsed;
  return true;
}

std::vector<size_t> SelectDistinctCandidateSurfaces(
    const converter::Segment& segment) {
  std::vector<size_t> selected;
  std::vector<size_t> supplemental;
  std::vector<std::string> surfaces;
  selected.reserve(kMaxAiCandidateSurfaceLimit);
  supplemental.reserve(kSupplementalCandidateSurfaceLimit);
  surfaces.reserve(kMaxAiCandidateSurfaceLimit);
  for (size_t i = 0; i < segment.candidates_size(); ++i) {
    const std::string value(segment.candidate(i).value);
    if (std::find(surfaces.begin(), surfaces.end(), value) != surfaces.end()) {
      continue;
    }
    if (segment.candidate(i).attributes &
        converter::Attribute::SUPPLEMENTAL_MODEL) {
      supplemental.push_back(i);
      continue;
    }
    if (surfaces.size() < kNormalCandidateSurfaceLimit) {
      selected.push_back(i);
      surfaces.push_back(value);
    }
  }

  // Reserve one slot for a high-value numeric/abbreviation supplement when
  // one exists, but fill all five slots with ordinary candidates otherwise.
  if (!supplemental.empty() && selected.size() >= kMaxAiCandidateSurfaceLimit) {
    selected.pop_back();
    surfaces.pop_back();
  }
  for (const size_t index : supplemental) {
    if (selected.size() >= kMaxAiCandidateSurfaceLimit ||
        selected.size() >= kNormalCandidateSurfaceLimit +
                                kSupplementalCandidateSurfaceLimit) {
      break;
    }
    const std::string value(segment.candidate(index).value);
    if (std::find(surfaces.begin(), surfaces.end(), value) != surfaces.end()) {
      continue;
    }
    selected.push_back(index);
    surfaces.push_back(value);
  }

  // If no supplement was present, or a duplicate supplement did not consume
  // the reserved slot, use the remaining ordinary candidates up to five.
  for (size_t i = 0; i < segment.candidates_size() &&
                     selected.size() < kMaxAiCandidateSurfaceLimit; ++i) {
    const std::string value(segment.candidate(i).value);
    if (segment.candidate(i).attributes &
            converter::Attribute::SUPPLEMENTAL_MODEL ||
        std::find(surfaces.begin(), surfaces.end(), value) != surfaces.end()) {
      continue;
    }
    selected.push_back(i);
    surfaces.push_back(value);
  }
  return selected;
}

size_t SharedCandidateContextChars(absl::string_view left,
                                   absl::string_view right) {
  const std::vector<std::string> left_chars =
      Util::SplitStringToUtf8Chars(left);
  const std::vector<std::string> right_chars =
      Util::SplitStringToUtf8Chars(right);
  const size_t limit = std::min(left_chars.size(), right_chars.size());
  size_t prefix = 0;
  while (prefix < limit && left_chars[prefix] == right_chars[prefix]) {
    ++prefix;
  }
  size_t suffix = 0;
  while (suffix < limit - prefix &&
         left_chars[left_chars.size() - 1 - suffix] ==
             right_chars[right_chars.size() - 1 - suffix]) {
    ++suffix;
  }
  return prefix + suffix;
}

bool HasCandidateInternalPhraseContext(const converter::Segment& segment) {
  if (segment.candidates_size() < 2) return false;
  const std::vector<size_t> selected =
      SelectDistinctCandidateSurfaces(segment);
  if (selected.size() < 2) return false;
  const std::string baseline(segment.candidate(selected[0]).value);
  for (size_t position = 1; position < selected.size(); ++position) {
    if (SharedCandidateContextChars(
            baseline, segment.candidate(selected[position]).value) >=
        kMinInternalPhraseContextChars) {
      return true;
    }
  }
  return false;
}

bool ApplySelectedPermutation(
    converter::Segment* segment,
    const std::vector<ai_ranker::RankedCandidate>& ranked,
    const std::vector<size_t>& selected_indices) {
  if (segment == nullptr || ranked.size() != selected_indices.size()) {
    return false;
  }
  const size_t candidate_count = segment->candidates_size();
  std::vector<size_t> desired;
  desired.reserve(candidate_count);
  std::vector<bool> seen(candidate_count, false);
  for (size_t rank = 0; rank < ranked.size(); ++rank) {
    const std::string& id = ranked[rank].id;
    size_t index = 0;
    if (!ParseCandidateId(id, candidate_count, &index)) return false;
    if (seen[index] || ranked[rank].rank != static_cast<int>(rank + 1)) {
      return false;
    }
    if (std::find(selected_indices.begin(), selected_indices.end(), index) ==
        selected_indices.end()) {
      return false;
    }
    seen[index] = true;
    desired.push_back(index);
  }
  for (size_t index : selected_indices) {
    if (index >= candidate_count || !seen[index]) return false;
  }
  // Ranked representatives go first.  Duplicate surfaces and candidates below
  // the neural comparison window retain their original relative order.
  for (size_t i = 0; i < candidate_count; ++i) {
    if (!seen[i]) desired.push_back(i);
  }

  const bool top_promoted = !desired.empty() && desired[0] != 0;
  std::vector<size_t> current_ids(candidate_count);
  for (size_t i = 0; i < candidate_count; ++i) current_ids[i] = i;
  std::vector<std::pair<int, int>> moves;
  moves.reserve(candidate_count);
  for (size_t target = 0; target < candidate_count; ++target) {
    const auto current_it =
        std::find(current_ids.begin() + target, current_ids.end(),
                  desired[target]);
    if (current_it == current_ids.end()) return false;
    const size_t current = static_cast<size_t>(
        std::distance(current_ids.begin(), current_it));
    if (current != target) {
      moves.emplace_back(static_cast<int>(current),
                         static_cast<int>(target));
      const size_t moved = current_ids[current];
      current_ids.erase(current_ids.begin() + current);
      current_ids.insert(current_ids.begin() + target, moved);
    }
  }
  if (moves.empty()) return false;
  for (const auto [current, target] : moves) {
    segment->move_candidate(current, target);
  }
  if (top_promoted) {
    segment->mutable_candidate(0)->attributes |= converter::Attribute::RERANKED;
  }
  return true;
}

bool ApplyWinner(converter::Segment* segment,
                 const ai_ranker::BatchSegmentResult& result,
                 double minimum_confidence) {
  if (segment == nullptr || result.confidence < minimum_confidence) {
    return false;
  }
  size_t winner_index = 0;
  if (!ParseCandidateId(result.winner_id, segment->candidates_size(),
                        &winner_index) || winner_index == 0) {
    return false;
  }
  segment->move_candidate(static_cast<int>(winner_index), 0);
  segment->mutable_candidate(0)->attributes |= converter::Attribute::RERANKED;
  return true;
}

}  // namespace

int AiRewriter::capability(const ConversionRequest& request) const {
  if (request.options().skip_slow_rewriters ||
      request.options().used_in_predictor_realtime_conversion) {
    return RewriterInterface::NOT_AVAILABLE;
  }
  return RewriterInterface::CONVERSION;
}

std::optional<RewriterInterface::ResizeSegmentsRequest>
AiRewriter::CheckResizeSegmentsRequest(const ConversionRequest& request,
                                       const Segments& segments) const {
  if (request.options().skip_slow_rewriters ||
      request.options().used_in_predictor_realtime_conversion ||
      segments.resized()) {
    return std::nullopt;
  }
  const size_t count = segments.conversion_segments_size();
  if (count == 0 || count > 6 || dictionary_ == nullptr) {
    return std::nullopt;
  }

  // Boundary repair used to issue a separate AI request before the final
  // rerank.  That made one Space operation pay for multiple synchronous model
  // calls.  Keep deterministic supplemental-prefix resizing below, but leave
  // semantic boundary decisions to Mozc so the AI path stays one batch request
  // per conversion.
  const std::string reading = FullReading(segments);
  const size_t supplemental_prefix_chars = SupplementalPrefixChars(reading);
  if (supplemental_prefix_chars > 0 && supplemental_prefix_chars <= 12) {
    ResizeSegmentsRequest resize_request = {.segment_index = 0};
    resize_request.segment_sizes[0] =
        static_cast<uint8_t>(supplemental_prefix_chars);
    return resize_request;
  }
  return std::nullopt;
}

bool AiRewriter::Rewrite(const ConversionRequest& request,
                         Segments* segments) const {
  if (request.options().skip_slow_rewriters ||
      request.options().used_in_predictor_realtime_conversion) {
    return false;
  }
  if (segments == nullptr || segments->conversion_segments_size() == 0) {
    return false;
  }

  std::string preceding_text(request.context().preceding_text());
  if (preceding_text.empty()) {
    preceding_text = segments->history_value();
  }
  const std::string trailing_text(request.context().following_text());
  const bool has_in_composition_context =
      !trailing_text.empty() || segments->conversion_segments_size() > 1;
  const bool has_internal_phrase_context =
      segments->conversion_segments_size() == 1 &&
      HasCandidateInternalPhraseContext(segments->conversion_segment(0));
  // Preserve the conservative single-word/no-context behavior, but do not
  // disable AI for an entire sentence just because nothing was committed
  // before the current composition.  Later conversion segments, or the
  // competing whole-phrase surfaces of a long collapsed segment, provide real
  // context even when the host application exposes no surrounding document
  // text.
  if (preceding_text.empty() && !segments->resized() &&
      !has_in_composition_context && !has_internal_phrase_context) {
    return false;
  }

  size_t rerankable_segments = 0;
  for (size_t i = 0; i < segments->conversion_segments_size(); ++i) {
    const converter::Segment& seg = segments->conversion_segment(i);
    if (IsRerankableSegment(seg) && seg.candidates_size() >= 2) {
      ++rerankable_segments;
    }
  }
  if (rerankable_segments == 0) {
    return false;
  }

  const Clock::time_point deadline =
      Clock::now() + std::chrono::milliseconds(ai_ranker::kDefaultTimeoutMs);
  ai_ranker::Client client(pipe_name_);
  std::vector<ai_ranker::BatchSegmentInput> batch;
  std::vector<size_t> batch_segment_indices;
  batch.reserve(rerankable_segments);
  batch_segment_indices.reserve(rerankable_segments);
  for (size_t index = 0; index < segments->conversion_segments_size();
       ++index) {
    converter::Segment* segment = segments->mutable_conversion_segment(index);
    if (segment == nullptr || !IsRerankableSegment(*segment)) {
      continue;
    }
    if (segment->candidates_size() < 2) {
      continue;
    }

    AddSupplementalCandidates(segment);

    const std::vector<size_t> selected_indices =
        SelectDistinctCandidateSurfaces(*segment);
    if (selected_indices.size() < 2) continue;
    std::vector<ai_ranker::CandidateInput> input;
    input.reserve(selected_indices.size());
    for (const size_t candidate_index : selected_indices) {
      const converter::Candidate& candidate = segment->candidate(candidate_index);
      input.push_back({"c" + std::to_string(candidate_index), candidate.value,
                       static_cast<int>(input.size() + 1)});
    }
    batch.push_back({
        "s" + std::to_string(index),
        ContextBeforeSegment(*segments, index, preceding_text),
        ContextAfterSegment(*segments, index, trailing_text),
        std::string(segment->key().data(), segment->key().size()),
        std::move(input),
    });
    batch_segment_indices.push_back(index);
  }

  if (batch.empty()) return false;

  std::vector<ai_ranker::BatchSegmentResult> initial_results;
  if (!client.RankBatch(batch, RemainingBudgetMs(deadline), &initial_results)) {
    return false;
  }

  std::vector<size_t> retry_indices;
  retry_indices.reserve(batch.size());
  bool any_reordered = false;
  for (size_t batch_index = 0; batch_index < batch.size(); ++batch_index) {
    const auto result_it = std::find_if(
        initial_results.begin(), initial_results.end(), [&](const auto& result) {
          return result.id == batch[batch_index].id;
        });
    if (result_it == initial_results.end() ||
        result_it->confidence < kBatchHighConfidence) {
      retry_indices.push_back(batch_index);
      continue;
    }
    const size_t segment_index = batch_segment_indices[batch_index];
    any_reordered =
        ApplyWinner(segments->mutable_conversion_segment(segment_index),
                    *result_it, kBatchHighConfidence) || any_reordered;
  }

  // Only uncertain segments pay for a second pass.  They are rebuilt in
  // composition order so each later retry can see earlier winners.
  for (const size_t batch_index : retry_indices) {
    if (RemainingBudgetMs(deadline) <= 0) break;
    const size_t segment_index = batch_segment_indices[batch_index];
    converter::Segment* segment =
        segments->mutable_conversion_segment(segment_index);
    if (segment == nullptr) continue;
    const std::vector<size_t> selected_indices =
        SelectDistinctCandidateSurfaces(*segment);
    if (selected_indices.size() < 2) continue;
    std::vector<ai_ranker::CandidateInput> input;
    input.reserve(selected_indices.size());
    for (const size_t candidate_index : selected_indices) {
      const converter::Candidate& candidate = segment->candidate(candidate_index);
      input.push_back({"c" + std::to_string(candidate_index), candidate.value,
                       static_cast<int>(input.size() + 1)});
    }
    const std::vector<ai_ranker::BatchSegmentInput> retry = {{
        batch[batch_index].id,
        ContextBeforeSegment(*segments, segment_index, preceding_text),
        ContextAfterSegment(*segments, segment_index, trailing_text),
        std::string(segment->key().data(), segment->key().size()),
        std::move(input),
    }};
    std::vector<ai_ranker::BatchSegmentResult> retry_results;
    if (!client.RankBatch(retry, RemainingBudgetMs(deadline), &retry_results) ||
        retry_results.size() != 1) {
      continue;
    }
    any_reordered = ApplyWinner(segment, retry_results[0],
                                kRetryMinimumConfidence) || any_reordered;
  }

  return any_reordered;
}

}  // namespace mozc
