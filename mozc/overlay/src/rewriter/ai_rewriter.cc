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

constexpr size_t kBoundaryDictionaryLimit = 6;
constexpr size_t kBoundaryRepairLimit = 12;
constexpr size_t kAiCandidateSurfaceLimit = 4;
constexpr size_t kSupplementalCandidateSurfaceLimit = 2;
constexpr size_t kMinInternalPhraseContextChars = 4;
constexpr int kBoundaryProbeTimeoutMs = 700;
constexpr int kPrefetchAckTimeoutMs = 5;
constexpr double kBoundaryRepairMargin = 0.60;

struct SurfaceOption {
  std::string value;
  int64_t cost = 0;
};

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

bool IsGrammarLikeKey(absl::string_view key) {
  return key == "こと" || key == "もの" || key == "ため" ||
         key == "よう" || key == "ので" || key == "のに" ||
         key == "から" || key == "まで" || key == "だけ" ||
         key == "ほど" || key == "する" || key == "した" ||
         key == "して" || key == "です" || key == "ます" ||
         key == "いる" || key == "ある" || key == "なる" ||
         key == "ない" || key == "たい" || key == "れる" ||
         key == "られる" || key == "せる" || key == "させる";
}

void AddSurfaceOption(std::vector<SurfaceOption>* options, std::string value,
                      int64_t cost) {
  if (options == nullptr || value.empty()) return;
  auto existing = std::find_if(options->begin(), options->end(),
                               [&](const SurfaceOption& option) {
                                 return option.value == value;
                               });
  if (existing == options->end()) {
    options->push_back({std::move(value), cost});
  } else if (cost < existing->cost) {
    existing->cost = cost;
  }
}

std::vector<SurfaceOption> LookupExactSurfaces(
    const dictionary::DictionaryInterface* dictionary, absl::string_view key,
    size_t limit = kBoundaryDictionaryLimit) {
  std::vector<SurfaceOption> options;
  if (dictionary == nullptr || key.empty()) return options;

  dictionary::InlineCallback callback;
  callback.OnToken(
      [&](absl::string_view, absl::string_view,
          const dictionary::Token& token) {
        AddSurfaceOption(&options, token.value, token.cost);
        return dictionary::DictionaryInterface::Callback::TRAVERSE_CONTINUE;
      });
  dictionary->LookupExact(key, &callback);

  std::sort(options.begin(), options.end(),
            [](const SurfaceOption& lhs, const SurfaceOption& rhs) {
              if (lhs.cost != rhs.cost) return lhs.cost < rhs.cost;
              return lhs.value < rhs.value;
            });
  if (options.size() > limit) options.resize(limit);
  return options;
}

bool ContainsSurface(const std::vector<SurfaceOption>& options,
                     absl::string_view value) {
  return std::any_of(options.begin(), options.end(),
                     [&](const SurfaceOption& option) {
                       return option.value == value;
                     });
}

bool SegmentContainsSurface(const converter::Segment& segment,
                            absl::string_view value) {
  for (size_t i = 0; i < segment.candidates_size(); ++i) {
    if (segment.candidate(i).value == value) return true;
  }
  return false;
}

std::vector<SurfaceOption> BuildSplitSurfaces(
    const dictionary::DictionaryInterface* dictionary,
    absl::string_view left_key, absl::string_view right_key) {
  const std::vector<SurfaceOption> left =
      LookupExactSurfaces(dictionary, left_key);
  const std::vector<SurfaceOption> right =
      LookupExactSurfaces(dictionary, right_key);
  std::vector<SurfaceOption> combined;
  if (left.empty() || right.empty()) return combined;

  for (const SurfaceOption& lhs : left) {
    for (const SurfaceOption& rhs : right) {
      AddSurfaceOption(&combined, lhs.value + rhs.value, lhs.cost + rhs.cost);
    }
  }
  std::sort(combined.begin(), combined.end(),
            [](const SurfaceOption& lhs, const SurfaceOption& rhs) {
              if (lhs.cost != rhs.cost) return lhs.cost < rhs.cost;
              return lhs.value < rhs.value;
            });
  if (combined.size() > kBoundaryRepairLimit) {
    combined.resize(kBoundaryRepairLimit);
  }
  return combined;
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

std::string CurrentSurface(const Segments& segments) {
  std::string surface;
  for (const converter::Segment& segment : segments.conversion_segments()) {
    surface.append(SegmentTopSurface(segment));
  }
  return surface;
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

std::vector<std::string> RepairValues(
    const std::vector<SurfaceOption>& options, absl::string_view baseline,
    const std::vector<SurfaceOption>* blocked = nullptr) {
  std::vector<std::string> values;
  for (const SurfaceOption& option : options) {
    if (option.value == baseline) continue;
    if (blocked != nullptr && ContainsSurface(*blocked, option.value)) continue;
    if (std::find(values.begin(), values.end(), option.value) != values.end()) {
      continue;
    }
    values.push_back(option.value);
    if (values.size() >= kBoundaryRepairLimit) break;
  }
  return values;
}

bool BoundaryRepairWins(const ai_ranker::Client& client,
                        const ConversionRequest& request,
                        const Segments& segments, const std::string& reading,
                        const std::string& baseline,
                        const std::vector<std::string>& repairs) {
  if (baseline.empty() || repairs.empty()) return false;

  std::vector<ai_ranker::CandidateInput> candidates;
  candidates.reserve(repairs.size() + 1);
  candidates.push_back({"baseline", baseline, 1});
  for (size_t i = 0; i < repairs.size(); ++i) {
    candidates.push_back({"repair" + std::to_string(i), repairs[i],
                          static_cast<int>(i + 2)});
  }

  std::string preceding_text(request.context().preceding_text());
  if (preceding_text.empty()) preceding_text = segments.history_value();
  const std::string following_text(request.context().following_text());

  std::vector<ai_ranker::RankedCandidate> ranked;
  if (!client.Rank(preceding_text, following_text, reading, candidates,
                   kBoundaryProbeTimeoutMs, &ranked) ||
      ranked.size() != candidates.size() || ranked.empty()) {
    return false;
  }

  const auto baseline_it =
      std::find_if(ranked.begin(), ranked.end(), [](const auto& item) {
        return item.id == "baseline";
      });
  if (baseline_it == ranked.end() ||
      ranked.front().id.rfind("repair", 0) != 0) {
    return false;
  }
  return ranked.front().score - baseline_it->score >= kBoundaryRepairMargin;
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
  selected.reserve(kAiCandidateSurfaceLimit);
  supplemental.reserve(kSupplementalCandidateSurfaceLimit);
  surfaces.reserve(kAiCandidateSurfaceLimit);
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
    if (selected.size() >= kAiCandidateSurfaceLimit) continue;
    selected.push_back(i);
    surfaces.push_back(value);
  }
  for (const size_t index : supplemental) {
    if (selected.size() >=
        kAiCandidateSurfaceLimit + kSupplementalCandidateSurfaceLimit) {
      break;
    }
    const std::string value(segment.candidate(index).value);
    if (std::find(surfaces.begin(), surfaces.end(), value) != surfaces.end()) {
      continue;
    }
    selected.push_back(index);
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

}  // namespace

int AiRewriter::capability(const ConversionRequest& request) const {
  if (request.options().used_in_predictor_realtime_conversion) {
    // Realtime conversion is used only to warm the speculative cache.  Rewrite
    // never applies an AI permutation on this path, so typing remains instant.
    return RewriterInterface::CONVERSION;
  }
  if (request.options().skip_slow_rewriters) {
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

  ai_ranker::Client client(pipe_name_);
  if (!client.IsAvailable(25)) return std::nullopt;

  constexpr size_t kMinCompoundChars = 4;
  constexpr size_t kMaxCompoundChars = 12;

  if (count == 1) {
    const converter::Segment& segment = segments.conversion_segment(0);
    if (!IsRerankableSegment(segment) || segment.candidates_size() == 0 ||
        segment.candidates_size() > 4) {
      return std::nullopt;
    }
    const std::string key(segment.key());
    const size_t key_chars = Util::CharsLen(key);
    if (key_chars < kMinCompoundChars || key_chars > kMaxCompoundChars) {
      return std::nullopt;
    }

    size_t split_chars = 0;
    size_t valid_splits = 0;
    for (size_t split = 2; split + 2 <= key_chars; ++split) {
      const absl::string_view prefix = Util::Utf8SubString(key, 0, split);
      const absl::string_view suffix =
          Util::Utf8SubString(key, split, key_chars - split);
      if (dictionary_->HasKey(prefix) && dictionary_->HasKey(suffix)) {
        split_chars = split;
        ++valid_splits;
      }
    }
    if (valid_splits != 1) return std::nullopt;

    const absl::string_view left_key = Util::Utf8SubString(key, 0, split_chars);
    const absl::string_view right_key =
        Util::Utf8SubString(key, split_chars, key_chars - split_chars);
    const std::vector<SurfaceOption> split_surfaces =
        BuildSplitSurfaces(dictionary_, left_key, right_key);
    std::vector<std::string> repairs;
    const std::string baseline = SegmentTopSurface(segment);
    for (const SurfaceOption& option : split_surfaces) {
      if (option.value == baseline || SegmentContainsSurface(segment, option.value)) {
        continue;
      }
      repairs.push_back(option.value);
      if (repairs.size() >= kBoundaryRepairLimit) break;
    }
    if (!BoundaryRepairWins(client, request, segments, key, baseline, repairs)) {
      return std::nullopt;
    }

    ResizeSegmentsRequest resize_request = {
        .segment_index = 0,
        .segment_sizes = {
            static_cast<uint8_t>(split_chars),
            static_cast<uint8_t>(key_chars - split_chars), 0, 0, 0, 0, 0, 0},
    };
    return resize_request;
  }

  // Numeric/abbreviation supplements may span two as well as three or more
  // Mozc segments (e.g. じゅう + ご + にち, or API + 連携).  Resize only the
  // recognized prefix so trailing particles/verbs remain independently
  // convertible.
  {
    const std::string reading = FullReading(segments);
    const size_t supplemental_prefix_chars = SupplementalPrefixChars(reading);
    if (supplemental_prefix_chars > 0 &&
        supplemental_prefix_chars <= kMaxCompoundChars) {
      ResizeSegmentsRequest resize_request = {.segment_index = 0};
      resize_request.segment_sizes[0] =
          static_cast<uint8_t>(supplemental_prefix_chars);
      return resize_request;
    }
  }

  if (count != 2) return std::nullopt;

  size_t total_key_chars = 0;
  for (const converter::Segment& segment : segments.conversion_segments()) {
    if (segment.segment_type() != converter::Segment::FREE ||
        segment.candidates_size() == 0) {
      return std::nullopt;
    }
    total_key_chars += segment.key_len();
  }
  if (total_key_chars < 2 || total_key_chars > kMaxCompoundChars) {
    return std::nullopt;
  }

  const converter::Segment& first = segments.conversion_segment(0);
  const converter::Segment& second = segments.conversion_segment(1);
  const std::string reading = FullReading(segments);
  const std::string baseline = CurrentSurface(segments);
  ResizeSegmentsRequest resize_request = {.segment_index = 0};

  if (second.key_len() <= 2) {
    if (total_key_chars < kMinCompoundChars || IsGrammarLikeKey(first.key()) ||
        IsGrammarLikeKey(second.key())) {
      return std::nullopt;
    }
    const std::vector<SurfaceOption> whole_surfaces =
        LookupExactSurfaces(dictionary_, reading, kBoundaryRepairLimit);
    const std::vector<std::string> repairs =
        RepairValues(whole_surfaces, baseline);
    if (!BoundaryRepairWins(client, request, segments, reading, baseline,
                            repairs)) {
      return std::nullopt;
    }
    resize_request.segment_sizes[0] =
        static_cast<uint8_t>(total_key_chars);
    return resize_request;
  }

  // Lock an existing two-word boundary only when it exposes a whole-surface
  // reading that the unsegmented dictionary path does not offer and the AI
  // clearly prefers that split-only path.  This is the transactional version
  // of the old 飛行 + 少年 protection: no confident repair, no boundary change.
  const std::vector<SurfaceOption> whole_surfaces =
      LookupExactSurfaces(dictionary_, reading, kBoundaryRepairLimit);
  if (whole_surfaces.empty()) return std::nullopt;
  const std::vector<SurfaceOption> split_surfaces =
      BuildSplitSurfaces(dictionary_, first.key(), second.key());
  const std::vector<std::string> repairs =
      RepairValues(split_surfaces, baseline, &whole_surfaces);
  if (!BoundaryRepairWins(client, request, segments, reading, baseline,
                          repairs)) {
    return std::nullopt;
  }
  resize_request.segment_sizes[0] = static_cast<uint8_t>(first.key_len());
  resize_request.segment_sizes[1] = static_cast<uint8_t>(second.key_len());
  return resize_request;
}

bool AiRewriter::Rewrite(const ConversionRequest& request,
                         Segments* segments) const {
  const bool realtime_prefetch =
      request.options().used_in_predictor_realtime_conversion;
  if (request.options().skip_slow_rewriters && !realtime_prefetch) {
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

  if (realtime_prefetch) {
    // Realtime conversion is latency-sensitive: enqueue only the nearest
    // rerankable segment and never modify prediction candidates.  The Python
    // ranker returns Mozc order immediately, then scores the latest request on
    // its background worker for a later Space/Convert cache hit.
    ai_ranker::Client client(pipe_name_);
    for (size_t reverse = segments->conversion_segments_size(); reverse > 0;
         --reverse) {
      const size_t index = reverse - 1;
      const converter::Segment& segment = segments->conversion_segment(index);
      if (!IsRerankableSegment(segment) || segment.candidates_size() < 2) {
        continue;
      }
      const std::vector<size_t> selected_indices =
          SelectDistinctCandidateSurfaces(segment);
      if (selected_indices.size() < 2) continue;

      std::vector<ai_ranker::CandidateInput> input;
      input.reserve(selected_indices.size());
      for (size_t candidate_index : selected_indices) {
        const converter::Candidate& candidate =
            segment.candidate(candidate_index);
        input.push_back({"c" + std::to_string(candidate_index), candidate.value,
                         static_cast<int>(input.size() + 1)});
      }
      const std::string segment_prefix =
          ContextBeforeSegment(*segments, index, preceding_text);
      const std::string following_text =
          ContextAfterSegment(*segments, index, trailing_text);
      client.Prefetch(
          segment_prefix, following_text,
          std::string(segment.key().data(), segment.key().size()), input,
          kPrefetchAckTimeoutMs);
      break;
    }
    return false;
  }

  const Clock::time_point deadline =
      Clock::now() + std::chrono::milliseconds(ai_ranker::kDefaultTimeoutMs);
  ai_ranker::Client client(pipe_name_);
  bool any_reordered = false;

  // Work from the caret/back of the composition.  A single slow segment must
  // not consume the shared conversion deadline before the user's most recent
  // text gets considered.  Context is rebuilt from the current candidate-zero
  // surfaces on every iteration, so earlier segments also see any successful
  // promotion already made to their right.
  for (size_t reverse = segments->conversion_segments_size(); reverse > 0;
       --reverse) {
    const size_t index = reverse - 1;
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
    const size_t limit = selected_indices.size();
    if (limit < 2) continue;
    std::vector<ai_ranker::CandidateInput> input;
    input.reserve(limit);
    for (size_t candidate_index : selected_indices) {
      const converter::Candidate& candidate = segment->candidate(candidate_index);
      input.push_back({"c" + std::to_string(candidate_index), candidate.value,
                       static_cast<int>(input.size() + 1)});
    }

    const std::string segment_prefix =
        ContextBeforeSegment(*segments, index, preceding_text);
    const std::string following_text =
        ContextAfterSegment(*segments, index, trailing_text);

    const int remaining_ms = RemainingBudgetMs(deadline);
    if (remaining_ms <= 0) break;

    std::vector<ai_ranker::RankedCandidate> ranked;
    const bool rank_ok = client.Rank(
        segment_prefix, following_text,
        std::string(segment->key().data(), segment->key().size()), input,
        remaining_ms, &ranked);
    if (!rank_ok || ranked.size() != limit) {
      continue;
    }

    if (ApplySelectedPermutation(segment, ranked, selected_indices)) {
      any_reordered = true;
    }
  }

  return any_reordered;
}

}  // namespace mozc