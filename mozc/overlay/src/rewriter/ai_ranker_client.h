#ifndef MOZC_REWRITER_AI_RANKER_CLIENT_H_
#define MOZC_REWRITER_AI_RANKER_CLIENT_H_

#include <string>
#include <vector>

namespace mozc {
namespace ai_ranker {

// The entire explicit conversion path is intentionally bounded.  If the AI
// does not answer within this budget, Mozc keeps its original candidate order.
constexpr int kDefaultTimeoutMs = 400;

// A request candidate is deliberately a copy of Mozc's existing value.  The
// ranker can only return one of these IDs; it cannot create a replacement.
struct CandidateInput {
  std::string id;
  std::string value;
  int original_rank = 0;
};

struct RankedCandidate {
  std::string id;
  double score = 0.0;
  int rank = 0;
};

struct BatchSegmentInput {
  std::string id;
  std::string preceding_text;
  std::string following_text;
  std::string reading;
  std::vector<CandidateInput> candidates;
};

struct BatchSegmentResult {
  std::string id;
  std::string winner_id;
  double confidence = 0.0;
};

class Client {
 public:
  explicit Client(std::wstring pipe_name);

  // Returns whether a server pipe becomes available within the short probe
  // budget.  This does not connect to the pipe or start inference.
  bool IsAvailable(int timeout_ms) const;

  // Returns false on every transport, deadline, or schema error.  |ranked|
  // is not modified on failure, so callers can preserve Mozc's ordering.
  bool Rank(const std::string& preceding_text, const std::string& reading,
            const std::vector<CandidateInput>& candidates, int timeout_ms,
            std::vector<RankedCandidate>* ranked) const;

  // Overload that also forwards the text after the conversion point so the
  // ranker can disambiguate phrases such as 「庭には美しい●が咲く」.
  bool Rank(const std::string& preceding_text, const std::string& following_text,
            const std::string& reading,
            const std::vector<CandidateInput>& candidates, int timeout_ms,
            std::vector<RankedCandidate>* ranked) const;

  // Rank all conversion segments in one compact request.  The response only
  // contains one winner and confidence per segment.
  bool RankBatch(const std::vector<BatchSegmentInput>& segments,
                 int timeout_ms,
                 std::vector<BatchSegmentResult>* results) const;

 private:
  std::wstring pipe_name_;
};

}  // namespace ai_ranker
}  // namespace mozc

#endif  // MOZC_REWRITER_AI_RANKER_CLIENT_H_
