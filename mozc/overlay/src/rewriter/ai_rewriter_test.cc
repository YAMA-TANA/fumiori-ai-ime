#include "rewriter/ai_rewriter.h"

#include <algorithm>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "converter/attribute.h"
#include "converter/candidate.h"
#include "converter/segments.h"
#include "dictionary/dictionary_interface.h"
#include "protocol/commands.pb.h"
#include "request/conversion_request.h"
#include "rewriter/rewriter_interface.h"
#include "testing/gunit.h"

namespace mozc {

namespace {

class CompoundDictionary final : public dictionary::DictionaryInterface {
 public:
  bool HasKey(absl::string_view key) const override {
    return key == "ひこう" || key == "しょうねん" ||
           key == "ひこうしょうねん" || key == "しゅせんりつ";
  }

  void LookupExact(absl::string_view key, Callback* callback) const override {
    if (callback == nullptr) return;
    auto emit = [&](absl::string_view value, int cost) {
      dictionary::Token token(key, value);
      token.cost = cost;
      callback->OnToken(key, key, token);
    };

    if (key == "ひこう") {
      emit("非行", 100);
      emit("飛行", 200);
    } else if (key == "しょうねん") {
      emit("少年", 100);
    } else if (key == "ひこうしょうねん") {
      emit("飛行少年", 100);
    } else if (key == "しゅせんりつ") {
      emit("主旋律", 100);
    }
  }
};

#ifdef _WIN32
class FakeRankerServer {
 public:
  FakeRankerServer(std::wstring pipe_name, std::string winner,
                   std::string required_fragment = std::string(),
                   size_t expected_requests = 1)
      : pipe_name_(std::move(pipe_name)),
        winner_(std::move(winner)),
        required_fragment_(std::move(required_fragment)),
        expected_requests_(expected_requests) {
    pipe_ = CreateNamedPipeW(pipe_name_.c_str(), PIPE_ACCESS_DUPLEX,
                             PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                             1, 65536, 65536, 0, nullptr);
    if (pipe_ != INVALID_HANDLE_VALUE) {
      thread_ = std::thread([this]() { Serve(); });
    }
  }

  ~FakeRankerServer() {
    if (thread_.joinable()) thread_.join();
    if (pipe_ != INVALID_HANDLE_VALUE) CloseHandle(pipe_);
  }

  bool valid() const { return pipe_ != INVALID_HANDLE_VALUE; }

  std::vector<std::string> readings() const {
    std::lock_guard<std::mutex> lock(readings_mutex_);
    return readings_;
  }

  std::vector<size_t> candidate_counts() const {
    std::lock_guard<std::mutex> lock(readings_mutex_);
    return candidate_counts_;
  }

 private:
  void Serve() {
    for (size_t request_index = 0; request_index < expected_requests_;
         ++request_index) {
      if (!ServeOne()) return;
    }
  }

  bool ServeOne() {
    const BOOL connected = ConnectNamedPipe(pipe_, nullptr);
    if (!connected && GetLastError() != ERROR_PIPE_CONNECTED) return false;

    std::string request;
    char buffer[65536];
    while (true) {
      DWORD read = 0;
      if (!ReadFile(pipe_, buffer, sizeof(buffer), &read, nullptr) || read == 0) {
        return false;
      }
      request.append(buffer, buffer + read);
      if (!request.empty() && request.back() == '\n') break;
      if (request.size() > 262144) return false;
    }

    const bool is_batch = request.find("\"segments\":[") != std::string::npos;
    if (!is_batch) {
      const std::string reading_marker = "\"read\":\"";
      const size_t reading_start = request.find(reading_marker);
      if (reading_start != std::string::npos) {
        const size_t value_start = reading_start + reading_marker.size();
        const size_t value_end = request.find('"', value_start);
        if (value_end != std::string::npos) {
          std::lock_guard<std::mutex> lock(readings_mutex_);
          readings_.push_back(request.substr(value_start, value_end - value_start));
        }
      }
    }

    const std::string request_marker = "\"request_id\":\"";
    const size_t id_start = request.find(request_marker);
    if (id_start == std::string::npos) return false;
    const size_t id_value_start = id_start + request_marker.size();
    const size_t id_end = request.find('"', id_value_start);
    if (id_end == std::string::npos) return false;
    const std::string request_id =
        request.substr(id_value_start, id_end - id_value_start);

    if (is_batch) {
      static const std::regex segment_header(
          "\\{\\\"id\\\":\\\"([A-Za-z0-9_.:-]+)\\\","
          "\\\"preceding_text\\\":\\\"[^\"]*\\\","
          "\\\"following_text\\\":\\\"[^\"]*\\\","
          "\\\"read\\\":\\\"([^\"]*)\\\","
          "\\\"candidates\\\":\\[");
      std::vector<std::string> segment_ids;
      std::vector<std::string> segment_readings;
      std::vector<std::vector<std::string>> segment_candidate_ids;
      size_t cursor = request.find('{', request.find("\"segments\":["));
      while (cursor != std::string::npos) {
        const std::string tail = request.substr(cursor);
        std::smatch match;
        if (!std::regex_search(tail, match, segment_header,
                               std::regex_constants::match_continuous)) {
          break;
        }
        const size_t candidate_start = cursor + match.length();
        const size_t candidate_end = request.find("]}", candidate_start);
        if (candidate_end == std::string::npos) return false;
        const std::string candidate_body =
            request.substr(candidate_start, candidate_end - candidate_start);
        static const std::regex candidate_regex(
            "\\{\\\"id\\\":\\\"([A-Za-z0-9_.:-]+)\\\",\\\"text\\\":");
        std::vector<std::string> ids;
        for (std::sregex_iterator it(candidate_body.begin(), candidate_body.end(),
                                     candidate_regex),
             end;
             it != end; ++it) {
          ids.push_back((*it)[1].str());
        }
        if (ids.empty()) return false;
        segment_ids.push_back(match[1].str());
        segment_readings.push_back(match[2].str());
        segment_candidate_ids.push_back(std::move(ids));
        cursor = request.find('{', candidate_end + 2);
      }
      if (segment_ids.empty()) return false;

      const bool winner_enabled =
          required_fragment_.empty() ||
          request.find(required_fragment_) != std::string::npos;
      {
        std::lock_guard<std::mutex> lock(readings_mutex_);
        for (size_t i = 0; i < segment_ids.size(); ++i) {
          readings_.push_back(segment_readings[i]);
          candidate_counts_.push_back(segment_candidate_ids[i].size());
        }
      }
      std::ostringstream response;
      response << "{\"request_id\":\"" << request_id
               << "\",\"segments\":[";
      for (size_t i = 0; i < segment_ids.size(); ++i) {
        if (i) response << ',';
        std::string winner = segment_candidate_ids[i].front();
        if (winner_enabled && std::find(segment_candidate_ids[i].begin(),
                                        segment_candidate_ids[i].end(), winner_) !=
                                 segment_candidate_ids[i].end()) {
          winner = winner_;
        }
        response << "{\"id\":\"" << segment_ids[i]
                 << "\",\"winner_id\":\"" << winner
                 << "\",\"confidence\":"
                 << (winner == winner_ && winner_enabled ? 0.95 : 0.60)
                 << '}';
      }
      response << "]}\n";
      const std::string payload = response.str();
      DWORD written = 0;
      WriteFile(pipe_, payload.data(), static_cast<DWORD>(payload.size()),
                &written, nullptr);
      FlushFileBuffers(pipe_);
      DisconnectNamedPipe(pipe_);
      return true;
    }

    static const std::regex candidate_regex(
        "\\{\\\"id\\\":\\\"([A-Za-z0-9_.:-]+)\\\",\\\"text\\\":");
    std::vector<std::string> ids;
    for (std::sregex_iterator it(request.begin(), request.end(),
                                 candidate_regex),
         end;
         it != end; ++it) {
      ids.push_back((*it)[1].str());
    }
    if (ids.empty()) return false;
    {
      std::lock_guard<std::mutex> lock(readings_mutex_);
      candidate_counts_.push_back(ids.size());
    }

    auto winner_it = std::find(ids.begin(), ids.end(), winner_);
    const bool winner_enabled =
        required_fragment_.empty() ||
        request.find(required_fragment_) != std::string::npos;
    if (winner_enabled && winner_it != ids.end()) {
      const std::string winner = *winner_it;
      ids.erase(winner_it);
      ids.insert(ids.begin(), winner);
    }

    std::ostringstream response;
    response << "{\"request_id\":\"" << request_id << "\",\"candidates\":[";
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i) response << ',';
      double score = -static_cast<double>(i + 1);
      if (winner_enabled && ids[i] == winner_) score = 10.0;
      if (ids[i] == "baseline" && winner_enabled && winner_ != "baseline") {
        score = 0.0;
      }
      response << "{\"id\":\"" << ids[i] << "\",\"score\":" << score
               << ",\"rank\":" << (i + 1) << '}';
    }
    response << "]}\n";
    const std::string payload = response.str();
    DWORD written = 0;
    WriteFile(pipe_, payload.data(), static_cast<DWORD>(payload.size()),
              &written, nullptr);
    FlushFileBuffers(pipe_);
    DisconnectNamedPipe(pipe_);
    return true;
  }

  std::wstring pipe_name_;
  std::string winner_;
  std::string required_fragment_;
  size_t expected_requests_;
  mutable std::mutex readings_mutex_;
  std::vector<std::string> readings_;
  std::vector<size_t> candidate_counts_;
  HANDLE pipe_ = INVALID_HANDLE_VALUE;
  std::thread thread_;
};
#endif

}  // namespace

TEST(AiRewriterTest, CapabilityIsConversionOnly) {
  AiRewriter rewriter(L"missing-ai-ime-pipe");
  const ConversionRequest request;
  EXPECT_EQ(rewriter.capability(request), RewriterInterface::CONVERSION);
}

TEST(AiRewriterTest, RealtimeConversionSkipsAiRanker) {
  ConversionRequest::Options options;
  options.request_type = ConversionRequest::CONVERSION;
  options.skip_slow_rewriters = true;
  const ConversionRequest request =
      ConversionRequestBuilder().SetOptions(std::move(options)).Build();

  AiRewriter rewriter(L"missing-ai-ime-pipe");
  EXPECT_EQ(rewriter.capability(request), RewriterInterface::NOT_AVAILABLE);

  Segments segments;
  Segment* segment = segments.add_segment();
  converter::Candidate* first = segment->add_candidate();
  first->value = "花";
  converter::Candidate* second = segment->add_candidate();
  second->value = "鼻";

  EXPECT_FALSE(rewriter.Rewrite(request, &segments));
  EXPECT_EQ(segments.segment(0).candidate(0).value, "花");
  EXPECT_EQ(segments.segment(0).candidate(1).value, "鼻");
}

TEST(AiRewriterTest, PredictorRealtimeMarkerSkipsAiRanker) {
  ConversionRequest::Options options;
  options.request_type = ConversionRequest::CONVERSION;
  options.used_in_predictor_realtime_conversion = true;
  const ConversionRequest request =
      ConversionRequestBuilder().SetOptions(std::move(options)).Build();

  AiRewriter rewriter(L"missing-ai-ime-pipe");
  EXPECT_EQ(rewriter.capability(request), RewriterInterface::NOT_AVAILABLE);

  Segments segments;
  Segment* segment = segments.add_segment();
  segment->add_candidate()->value = "花";
  segment->add_candidate()->value = "鼻";
  EXPECT_FALSE(rewriter.Rewrite(request, &segments));
}

#ifdef _WIN32
TEST(AiRewriterTest, SemanticBoundaryRepairDoesNotIssueAnAiRequest) {
  CompoundDictionary dictionary;

  Segments segments;
  Segment* first = segments.add_segment();
  first->set_key("しゅせん");
  first->add_candidate()->value = "主戦";
  Segment* second = segments.add_segment();
  second->set_key("りつ");
  second->add_candidate()->value = "率";

  commands::Context context;
  context.set_preceding_text("この曲の");
  const ConversionRequest request =
      ConversionRequestBuilder().SetContext(context).Build();
  AiRewriter rewriter(&dictionary, L"unused-ai-ime-pipe");
  EXPECT_FALSE(rewriter.CheckResizeSegmentsRequest(request, segments).has_value());
}

TEST(AiRewriterTest, SemanticBoundaryKeepsMozcBoundaryWithoutAi) {
  CompoundDictionary dictionary;

  Segments segments;
  Segment* first = segments.add_segment();
  first->set_key("しゅせん");
  first->add_candidate()->value = "主戦";
  Segment* second = segments.add_segment();
  second->set_key("りつ");
  second->add_candidate()->value = "率";

  const ConversionRequest request;
  AiRewriter rewriter(&dictionary, L"unused-ai-ime-pipe");
  EXPECT_FALSE(rewriter.CheckResizeSegmentsRequest(request, segments).has_value());
}

TEST(AiRewriterTest, AvailableRankerDoesNotMergeThreeSegmentPhrase) {
  Segments segments;
  Segment* first = segments.add_segment();
  first->set_key("わたしが");
  first->add_candidate()->value = "私が";
  Segment* second = segments.add_segment();
  second->set_key("する");
  second->add_candidate()->value = "する";
  Segment* third = segments.add_segment();
  third->set_key("こと");
  third->add_candidate()->value = "こと";

  CompoundDictionary dictionary;
  const ConversionRequest request;
  AiRewriter rewriter(&dictionary, L"missing-ai-ime-pipe");
  EXPECT_FALSE(rewriter.CheckResizeSegmentsRequest(request, segments).has_value());
}

TEST(AiRewriterTest, AvailableRankerDoesNotMergeGrammarSuffix) {
  Segments segments;
  Segment* first = segments.add_segment();
  first->set_key("わたしがする");
  first->add_candidate()->value = "私がする";
  Segment* second = segments.add_segment();
  second->set_key("こと");
  second->add_candidate()->value = "こと";

  CompoundDictionary dictionary;
  const ConversionRequest request;
  AiRewriter rewriter(&dictionary, L"missing-ai-ime-pipe");
  EXPECT_FALSE(rewriter.CheckResizeSegmentsRequest(request, segments).has_value());
}

TEST(AiRewriterTest, SemanticBoundaryDoesNotLockSplitWithAi) {
  CompoundDictionary dictionary;

  Segments segments;
  Segment* first = segments.add_segment();
  first->set_key("ひこう");
  first->add_candidate()->value = "飛行";
  Segment* second = segments.add_segment();
  second->set_key("しょうねん");
  second->add_candidate()->value = "少年";

  const ConversionRequest request;
  AiRewriter rewriter(&dictionary, L"unused-ai-ime-pipe");
  EXPECT_FALSE(rewriter.CheckResizeSegmentsRequest(request, segments).has_value());
}

TEST(AiRewriterTest, SemanticBoundaryDoesNotRestoreCollapsedCompoundWithAi) {
  CompoundDictionary dictionary;

  Segments segments;
  Segment* segment = segments.add_segment();
  segment->set_segment_type(converter::Segment::FIXED_BOUNDARY);
  segment->set_key("ひこうしょうねん");
  segment->add_candidate()->value = "飛行少年";
  segment->add_candidate()->value = "ひこうしょうねん";
  segment->add_candidate()->value = "ヒコウショウネン";

  const ConversionRequest request;
  AiRewriter rewriter(&dictionary, L"unused-ai-ime-pipe");
  EXPECT_FALSE(rewriter.CheckResizeSegmentsRequest(request, segments).has_value());
}

TEST(AiRewriterTest, RankerWinnerMovesToCandidateZero) {
  const std::wstring pipe_name =
      L"\\\\.\\pipe\\yamatana_ai_rewriter_rank_apply_test";
  FakeRankerServer server(pipe_name, "c1");
  ASSERT_TRUE(server.valid());

  Segments segments;
  Segment* segment = segments.add_segment();
  segment->set_key("はな");
  converter::Candidate* first = segment->add_candidate();
  first->key = "はな";
  first->value = "花";
  converter::Candidate* second = segment->add_candidate();
  second->key = "はな";
  second->value = "鼻";

  commands::Context context;
  context.set_preceding_text("象の長い");
  const ConversionRequest request =
      ConversionRequestBuilder().SetContext(context).Build();
  AiRewriter rewriter(pipe_name);

  EXPECT_TRUE(rewriter.Rewrite(request, &segments));
  ASSERT_EQ(segments.segment(0).candidates_size(), 2);
  EXPECT_EQ(segments.segment(0).candidate(0).value, "鼻");
  EXPECT_EQ(segments.segment(0).candidate(1).value, "花");
  EXPECT_NE(segments.segment(0).candidate(0).attributes &
                converter::Attribute::RERANKED,
            0);
}

TEST(AiRewriterTest, SendsAtMostFiveDistinctSurfacesToRanker) {
  const std::wstring pipe_name =
      L"\\\\.\\pipe\\yamatana_ai_rewriter_candidate_limit_test";
  FakeRankerServer server(pipe_name, "c8");
  ASSERT_TRUE(server.valid());

  Segments segments;
  Segment* segment = segments.add_segment();
  segment->set_key("こうほ");
  for (size_t i = 0; i < 10; ++i) {
    segment->add_candidate()->value = "候補" + std::to_string(i);
  }

  commands::Context context;
  context.set_preceding_text("この文の");
  const ConversionRequest request =
      ConversionRequestBuilder().SetContext(context).Build();
  AiRewriter rewriter(pipe_name);

  EXPECT_FALSE(rewriter.Rewrite(request, &segments));
  EXPECT_EQ(segments.segment(0).candidate(0).value, "候補0");
  const std::vector<size_t> counts = server.candidate_counts();
  ASSERT_EQ(counts.size(), 1);
  EXPECT_EQ(counts[0], 5);
}

TEST(AiRewriterTest, DuplicateSurfacesAreNotSentAsSeparateChoices) {
  const std::wstring pipe_name =
      L"\\\\.\\pipe\\yamatana_ai_rewriter_candidate_dedupe_test";
  FakeRankerServer server(pipe_name, "c2");
  ASSERT_TRUE(server.valid());

  Segments segments;
  Segment* segment = segments.add_segment();
  segment->set_key("はな");
  segment->add_candidate()->value = "花";
  segment->add_candidate()->value = "花";
  segment->add_candidate()->value = "鼻";
  segment->add_candidate()->value = "華";

  commands::Context context;
  context.set_preceding_text("彼の顔の");
  const ConversionRequest request =
      ConversionRequestBuilder().SetContext(context).Build();
  AiRewriter rewriter(pipe_name);

  EXPECT_TRUE(rewriter.Rewrite(request, &segments));
  EXPECT_EQ(segments.segment(0).candidate(0).value, "鼻");
  const std::vector<size_t> counts = server.candidate_counts();
  ASSERT_EQ(counts.size(), 1);
  EXPECT_EQ(counts[0], 3);
}

TEST(AiRewriterTest, SentenceStartUsesFollowingSegmentAsContext) {
  const std::wstring pipe_name =
      L"\\\\.\\pipe\\yamatana_ai_rewriter_sentence_start_test";
  FakeRankerServer server(pipe_name, "c1");
  ASSERT_TRUE(server.valid());

  Segments segments;
  Segment* first = segments.add_segment();
  first->set_key("はな");
  first->add_candidate()->value = "花";
  first->add_candidate()->value = "鼻";
  Segment* second = segments.add_segment();
  second->set_key("がながい");
  second->add_candidate()->value = "が長い";

  const ConversionRequest request;
  AiRewriter rewriter(pipe_name);

  EXPECT_TRUE(rewriter.Rewrite(request, &segments));
  EXPECT_EQ(segments.segment(0).candidate(0).value, "鼻");
  EXPECT_EQ(segments.segment(0).candidate(1).value, "花");
}

TEST(AiRewriterTest, FollowingSegmentsComeBeforeDocumentSuffix) {
  const std::wstring pipe_name =
      L"\\\\.\\pipe\\yamatana_ai_rewriter_suffix_order_test";
  FakeRankerServer server(
      pipe_name, "c1", "\"following_text\":\"後続文末\"");
  ASSERT_TRUE(server.valid());

  Segments segments;
  Segment* target = segments.add_segment();
  target->set_key("はな");
  target->add_candidate()->value = "鼻";
  target->add_candidate()->value = "花";
  Segment* later = segments.add_segment();
  later->set_key("こうぞく");
  later->add_candidate()->value = "後続";

  commands::Context context;
  context.set_following_text("文末");
  const ConversionRequest request =
      ConversionRequestBuilder().SetContext(context).Build();
  AiRewriter rewriter(pipe_name);

  EXPECT_TRUE(rewriter.Rewrite(request, &segments));
  EXPECT_EQ(segments.segment(0).candidate(0).value, "花");
}

TEST(AiRewriterTest, MultipleSegmentsAreRankedFromLeftToRight) {
  const std::wstring pipe_name =
      L"\\\\.\\pipe\\yamatana_ai_rewriter_left_to_right_test";
  FakeRankerServer server(pipe_name, "c0");
  ASSERT_TRUE(server.valid());

  Segments segments;
  Segment* left = segments.add_segment();
  left->set_key("ひだり");
  left->add_candidate()->value = "左";
  left->add_candidate()->value = "ヒダリ";
  Segment* right = segments.add_segment();
  right->set_key("みぎ");
  right->add_candidate()->value = "右";
  right->add_candidate()->value = "ミギ";

  const ConversionRequest request;
  AiRewriter rewriter(pipe_name);

  EXPECT_FALSE(rewriter.Rewrite(request, &segments));
  EXPECT_EQ(server.readings(),
            (std::vector<std::string>{"ひだり", "みぎ"}));
}

TEST(AiRewriterTest, LongCollapsedPhraseAtSentenceStartUsesAi) {
  const std::wstring pipe_name =
      L"\\\\.\\pipe\\yamatana_ai_rewriter_collapsed_phrase_test";
  FakeRankerServer server(pipe_name, "c1");
  ASSERT_TRUE(server.valid());

  Segments segments;
  Segment* segment = segments.add_segment();
  // The decision is based on shared phrase surfaces, not reading length.
  segment->set_key("はな");
  segment->add_candidate()->value = "庭には美しい鼻";
  segment->add_candidate()->value = "庭には美しい花";

  const ConversionRequest request;
  AiRewriter rewriter(pipe_name);

  EXPECT_TRUE(rewriter.Rewrite(request, &segments));
  EXPECT_EQ(segments.segment(0).candidate(0).value, "庭には美しい花");
  EXPECT_EQ(segments.segment(0).candidate(1).value, "庭には美しい鼻");
}

TEST(AiRewriterTest, UnchangedAiOrderDoesNotMarkReranked) {
  const std::wstring pipe_name =
      L"\\\\.\\pipe\\yamatana_ai_rewriter_same_order_test";
  FakeRankerServer server(pipe_name, "c0");
  ASSERT_TRUE(server.valid());

  Segments segments;
  Segment* segment = segments.add_segment();
  segment->set_key("はな");
  segment->add_candidate()->value = "花";
  segment->add_candidate()->value = "鼻";

  commands::Context context;
  context.set_preceding_text("庭の");
  const ConversionRequest request =
      ConversionRequestBuilder().SetContext(context).Build();
  AiRewriter rewriter(pipe_name);

  EXPECT_FALSE(rewriter.Rewrite(request, &segments));
  EXPECT_EQ(segments.segment(0).candidate(0).value, "花");
  EXPECT_EQ(segments.segment(0).candidate(0).attributes &
                converter::Attribute::RERANKED,
            0);
}
#endif

TEST(AiRewriterTest, NoContextPreservesMozcOrder) {
  Segments segments;
  Segment* segment = segments.add_segment();
  segment->set_key("hana");
  converter::Candidate* first = segment->add_candidate();
  first->key = "hana";
  first->value = "花";
  converter::Candidate* second = segment->add_candidate();
  second->key = "hana";
  second->value = "鼻";

  AiRewriter rewriter(L"missing-ai-ime-pipe");
  const ConversionRequest request;
  EXPECT_FALSE(rewriter.Rewrite(request, &segments));
  ASSERT_EQ(segments.segment(0).candidates_size(), 2);
  EXPECT_EQ(segments.segment(0).candidate(0).value, "花");
  EXPECT_EQ(segments.segment(0).candidate(1).value, "鼻");
}

TEST(AiRewriterTest, RankerFailureWithContextPreservesMozcOrder) {
  Segments segments;
  Segment* segment = segments.add_segment();
  segment->set_key("hana");
  converter::Candidate* first = segment->add_candidate();
  first->key = "hana";
  first->value = "花";
  converter::Candidate* second = segment->add_candidate();
  second->key = "hana";
  second->value = "鼻";

  commands::Context context;
  context.set_preceding_text("象の長い");
  const ConversionRequest request =
      ConversionRequestBuilder().SetContext(context).Build();
  AiRewriter rewriter(L"missing-ai-ime-pipe");
  EXPECT_FALSE(rewriter.Rewrite(request, &segments));
  ASSERT_EQ(segments.segment(0).candidates_size(), 2);
  EXPECT_EQ(segments.segment(0).candidate(0).value, "花");
  EXPECT_EQ(segments.segment(0).candidate(1).value, "鼻");
}

}  // namespace mozc
