#ifndef MOZC_REWRITER_AI_RANKER_CLIENT_H_
#define MOZC_REWRITER_AI_RANKER_CLIENT_H_

#include <string>
#include <vector>

namespace mozc {
namespace ai_ranker {

// Explicit conversion may need a full CPU inference pass.  Typing-time
// prefetch requests only enqueue work and return immediately.
constexpr int kDefaultTimeoutMs = 2000;

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

  // Queue a typing-time speculative pass.  The server acknowledges immediately
  // with Mozc order while neural inference continues in its background worker.
  bool Prefetch(const std::string& preceding_text,
                const std::string& following_text,
                const std::string& reading,
                const std::vector<CandidateInput>& candidates,
                int timeout_ms) const;

 private:
  bool Request(const char* trigger,
               const std::string& preceding_text,
               const std::string& following_text,
               const std::string& reading,
               const std::vector<CandidateInput>& candidates,
               int timeout_ms,
               std::vector<RankedCandidate>* ranked) const;

  std::wstring pipe_name_;
};

}  // namespace ai_ranker
}  // namespace mozc

#endif  // MOZC_REWRITER_AI_RANKER_CLIENT_H_
