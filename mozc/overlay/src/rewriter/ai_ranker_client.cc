#include "rewriter/ai_ranker_client.h"

#ifdef _WIN32

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <regex>
#include <set>
#include <sstream>
#include <utility>

namespace mozc {
namespace ai_ranker {
namespace {

constexpr size_t kMaxResponseBytes = 262144;
constexpr size_t kMaxRequestBytes = 2097152;
constexpr size_t kMaxCandidates = 512;

bool EscapeJson(const std::string& value, std::string* out) {
  if (out == nullptr) return false;
  out->clear();
  out->reserve(value.size() + 8);
  for (const unsigned char c : value) {
    switch (c) {
      case '\\': *out += "\\\\"; break;
      case '"': *out += "\\\""; break;
      case '\n': *out += "\\n"; break;
      case '\r': *out += "\\r"; break;
      case '\t': *out += "\\t"; break;
      default:
        if (c < 0x20) return false;
        out->push_back(static_cast<char>(c));
    }
  }
  return true;
}

bool SafeId(const std::string& id) {
  if (id.empty() || id.size() > 128) return false;
  for (const unsigned char c : id) {
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '.' || c == ':' ||
          c == '-')) {
      return false;
    }
  }
  return true;
}

DWORD RemainingMs(ULONGLONG deadline) {
  const ULONGLONG now = GetTickCount64();
  if (now >= deadline) return 0;
  return static_cast<DWORD>(std::min<ULONGLONG>(deadline - now, INFINITE - 1));
}

bool WaitForPipeUntil(const std::wstring& pipe_name, ULONGLONG deadline) {
  while (const DWORD remaining_ms = RemainingMs(deadline)) {
    // Use short slices so a server that closes one pipe instance and creates
    // the next one cannot make the client fail with ERROR_FILE_NOT_FOUND.
    const DWORD slice_ms = std::min<DWORD>(remaining_ms, 50);
    if (WaitNamedPipeW(pipe_name.c_str(), slice_ms)) return true;
    const DWORD error = GetLastError();
    if (error != ERROR_FILE_NOT_FOUND && error != ERROR_PIPE_BUSY &&
        error != ERROR_SEM_TIMEOUT) {
      return false;
    }
    if (error == ERROR_FILE_NOT_FOUND) Sleep(1);
  }
  return false;
}

HANDLE OpenPipeUntil(const std::wstring& pipe_name, ULONGLONG deadline) {
  while (WaitForPipeUntil(pipe_name, deadline)) {
    HANDLE pipe =
        CreateFileW(pipe_name.c_str(), GENERIC_READ | GENERIC_WRITE, 0,
                    nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
    if (pipe != INVALID_HANDLE_VALUE) return pipe;
    const DWORD error = GetLastError();
    if (error != ERROR_FILE_NOT_FOUND && error != ERROR_PIPE_BUSY) break;
  }
  return INVALID_HANDLE_VALUE;
}

void CancelAndClose(HANDLE pipe, OVERLAPPED* overlapped) {
  // Keep stack OVERLAPPED storage alive until kernel cancellation completes.
  CancelIoEx(pipe, overlapped);
  WaitForSingleObject(overlapped->hEvent, INFINITE);
  CloseHandle(overlapped->hEvent);
}

bool WriteDeadline(HANDLE pipe, const std::string& payload,
                   ULONGLONG deadline) {
  DWORD offset = 0;
  while (offset < payload.size()) {
    OVERLAPPED overlapped{};
    overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!overlapped.hEvent) return false;
    DWORD written = 0;
    const BOOL ok = WriteFile(pipe, payload.data() + offset,
                              static_cast<DWORD>(payload.size() - offset),
                              &written, &overlapped);
    const bool pending = !ok && GetLastError() == ERROR_IO_PENDING;
    if (!ok && !pending) {
      CloseHandle(overlapped.hEvent);
      return false;
    }
    if (pending &&
        (RemainingMs(deadline) == 0 ||
         WaitForSingleObject(overlapped.hEvent, RemainingMs(deadline)) !=
             WAIT_OBJECT_0)) {
      CancelAndClose(pipe, &overlapped);
      return false;
    }
    const bool result =
        GetOverlappedResult(pipe, &overlapped, &written, FALSE) != FALSE;
    CloseHandle(overlapped.hEvent);
    if (!result || written == 0 || written > payload.size() - offset) {
      return false;
    }
    offset += written;
  }
  return true;
}

bool ReadDeadline(HANDLE pipe, std::string* output, ULONGLONG deadline) {
  if (output == nullptr) return false;
  output->clear();
  while (output->size() < kMaxResponseBytes) {
    OVERLAPPED overlapped{};
    overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!overlapped.hEvent) return false;
    char buffer[4096];
    DWORD read = 0;
    const BOOL ok = ReadFile(pipe, buffer, sizeof(buffer), &read, &overlapped);
    const bool pending = !ok && GetLastError() == ERROR_IO_PENDING;
    if (!ok && !pending) {
      CloseHandle(overlapped.hEvent);
      return false;
    }
    if (pending &&
        (RemainingMs(deadline) == 0 ||
         WaitForSingleObject(overlapped.hEvent, RemainingMs(deadline)) !=
             WAIT_OBJECT_0)) {
      CancelAndClose(pipe, &overlapped);
      return false;
    }
    const bool result =
        GetOverlappedResult(pipe, &overlapped, &read, FALSE) != FALSE;
    CloseHandle(overlapped.hEvent);
    if (!result || read == 0) return false;
    output->append(buffer, buffer + read);
    const size_t newline = output->find('\n');
    if (newline != std::string::npos) return newline + 1 == output->size();
  }
  return false;
}

bool ParseResponse(const std::string& response, const std::string& request_id,
                   const std::set<std::string>& allowed,
                   std::vector<RankedCandidate>* output) {
  if (output == nullptr || response.empty() || response.back() != '\n') {
    return false;
  }
  const std::string body = response.substr(0, response.size() - 1);
  const std::string prefix =
      "{\"request_id\":\"" + request_id + "\",\"candidates\":[";
  if (body.size() < prefix.size() + 2 ||
      body.compare(0, prefix.size(), prefix) != 0 ||
      body.substr(body.size() - 2) != "]}") {
    return false;
  }
  const std::string list =
      body.substr(prefix.size(), body.size() - prefix.size() - 2);
  // Strict JSON number grammar; reject NaN/Infinity, '+', and leading zeroes.
  static const std::regex item(
      "\\{\"id\":\"([A-Za-z0-9_.:-]+)\",\"score\":(-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?),\"rank\":([1-9][0-9]*)\\}");
  std::vector<RankedCandidate> parsed;
  size_t cursor = 0;
  while (cursor < list.size()) {
    std::smatch match;
    const std::string tail = list.substr(cursor);
    if (!std::regex_search(tail, match, item,
                           std::regex_constants::match_continuous)) {
      return false;
    }
    RankedCandidate candidate;
    candidate.id = match[1].str();
    try {
      candidate.score = std::stod(match[2].str());
      candidate.rank = std::stoi(match[3].str());
    } catch (...) {
      return false;
    }
    if (!std::isfinite(candidate.score) || !allowed.count(candidate.id) ||
        candidate.rank != static_cast<int>(parsed.size()) + 1 ||
        std::any_of(parsed.begin(), parsed.end(), [&](const auto& item) {
          return item.id == candidate.id;
        })) {
      return false;
    }
    parsed.push_back(std::move(candidate));
    cursor += static_cast<size_t>(match.length());
    if (cursor < list.size()) {
      if (list[cursor] != ',') return false;
      ++cursor;
    }
  }
  if (parsed.size() != allowed.size()) return false;
  *output = std::move(parsed);
  return true;
}

std::string NextRequestId() {
  static std::atomic<uint64_t> sequence{0};
  std::ostringstream id;
  id << "mozc-" << ++sequence;
  return id.str();
}

}  // namespace

Client::Client(std::wstring pipe_name) : pipe_name_(std::move(pipe_name)) {}

bool Client::IsAvailable(int timeout_ms) const {
  if (timeout_ms <= 0) return false;
  return WaitForPipeUntil(
      pipe_name_, GetTickCount64() + static_cast<ULONGLONG>(timeout_ms));
}

bool Client::Rank(const std::string& preceding_text, const std::string& reading,
                  const std::vector<CandidateInput>& candidates,
                  int timeout_ms,
                  std::vector<RankedCandidate>* ranked) const {
  return Request("explicit", preceding_text, std::string(), reading, candidates,
                 timeout_ms, ranked);
}

bool Client::Rank(const std::string& preceding_text,
                  const std::string& following_text,
                  const std::string& reading,
                  const std::vector<CandidateInput>& candidates,
                  int timeout_ms,
                  std::vector<RankedCandidate>* ranked) const {
  return Request("explicit", preceding_text, following_text, reading, candidates,
                 timeout_ms, ranked);
}

bool Client::Prefetch(const std::string& preceding_text,
                      const std::string& following_text,
                      const std::string& reading,
                      const std::vector<CandidateInput>& candidates,
                      int timeout_ms) const {
  std::vector<RankedCandidate> ignored;
  return Request("prefetch", preceding_text, following_text, reading, candidates,
                 timeout_ms, &ignored);
}

bool Client::Request(const char* trigger,
                     const std::string& preceding_text,
                     const std::string& following_text,
                     const std::string& reading,
                     const std::vector<CandidateInput>& candidates,
                     int timeout_ms,
                     std::vector<RankedCandidate>* ranked) const {
  const std::string trigger_value = trigger == nullptr ? "" : trigger;
  if ((trigger_value != "explicit" && trigger_value != "prefetch") ||
      ranked == nullptr || candidates.empty() ||
      candidates.size() > kMaxCandidates || timeout_ms <= 0 ||
      preceding_text.size() > 32768 || following_text.size() > 32768 ||
      reading.size() > 512) {
    return false;
  }
  // Keep a defensive upper bound for callers, but do not silently truncate
  // AiRewriter's explicit-conversion budget.  Prefetch callers pass a tiny
  // acknowledgement budget because the server returns Mozc order immediately.
  constexpr int kMaxTimeoutMs = 3000;
  const int budget_ms = std::min(timeout_ms, kMaxTimeoutMs);
  const std::string request_id = NextRequestId();
  std::ostringstream json;
  std::string escaped;
  if (!EscapeJson(preceding_text, &escaped)) return false;
  json << "{\"request_id\":\"" << request_id
       << "\",\"inference_trigger\":\"" << trigger_value
       << "\",\"preceding_text\":\"" << escaped
       << "\",\"following_text\":";
  if (!EscapeJson(following_text, &escaped)) return false;
  json << "\"" << escaped << "\",\"read\":";
  if (!EscapeJson(reading, &escaped)) return false;
  json << "\"" << escaped << "\",\"candidates\":[";
  std::set<std::string> allowed;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const CandidateInput& candidate = candidates[i];
    if (!SafeId(candidate.id) || candidate.value.size() > 4096 ||
        candidate.original_rank != static_cast<int>(i) + 1 ||
        !allowed.insert(candidate.id).second) {
      return false;
    }
    std::string escaped_id;
    std::string escaped_value;
    if (!EscapeJson(candidate.id, &escaped_id) ||
        !EscapeJson(candidate.value, &escaped_value)) {
      return false;
    }
    if (i) json << ',';
    json << "{\"id\":\"" << escaped_id << "\",\"text\":\""
         << escaped_value << "\",\"rank\":" << candidate.original_rank
         << '}';
  }
  json << "]}\n";
  const std::string payload = json.str();
  if (payload.size() > kMaxRequestBytes) return false;

  const ULONGLONG deadline =
      GetTickCount64() + static_cast<ULONGLONG>(budget_ms);
  HANDLE pipe = OpenPipeUntil(pipe_name_, deadline);
  if (pipe == INVALID_HANDLE_VALUE) return false;
  std::string response;
  bool ok = WriteDeadline(pipe, payload, deadline);
  if (ok) ok = ReadDeadline(pipe, &response, deadline);
  if (ok) ok = ParseResponse(response, request_id, allowed, ranked);
  CloseHandle(pipe);
  return ok;
}

}  // namespace ai_ranker
}  // namespace mozc

#else

namespace mozc {
namespace ai_ranker {
Client::Client(std::wstring) {}
bool Client::IsAvailable(int) const { return false; }
bool Client::Rank(const std::string&, const std::string&,
                  const std::vector<CandidateInput>&, int,
                  std::vector<RankedCandidate>*) const {
  return false;
}
bool Client::Rank(const std::string&, const std::string&,
                  const std::string&, const std::vector<CandidateInput>&, int,
                  std::vector<RankedCandidate>*) const {
  return false;
}
bool Client::Prefetch(const std::string&, const std::string&,
                      const std::string&,
                      const std::vector<CandidateInput>&, int) const {
  return false;
}
bool Client::Request(const char*, const std::string&, const std::string&,
                     const std::string&,
                     const std::vector<CandidateInput>&, int,
                     std::vector<RankedCandidate>*) const {
  return false;
}
}  // namespace ai_ranker
}  // namespace mozc

#endif
