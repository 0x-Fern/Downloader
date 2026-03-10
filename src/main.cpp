#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <curl/curl.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

// Metadata collected before any worker threads are created.
// The downloader needs to know both the total size and whether byte ranges are
// supported so it can decide between parallel mode and single-stream fallback.
struct DownloadMetadata {
    // Total file size reported by the server, or 0 when unknown.
    curl_off_t content_length = 0;
    // True only when the server explicitly supports ranged requests.
    bool accepts_ranges = false;
};

// One chunk assigned to one worker thread.
// Each task carries its own byte range and temporary file path so workers do
// not share output state.
struct ChunkTask {
    // Zero-based worker index used in logs.
    std::size_t index = 0;
    // Inclusive start byte for this chunk.
    curl_off_t start = 0;
    // Inclusive end byte for this chunk.
    curl_off_t end = 0;
    // False when the downloader falls back to a single full-file request.
    bool use_range = false;
    // Temporary file where this worker writes its bytes.
    fs::path temp_path;
};

// RAII wrapper for libcurl global initialization.
// This keeps process-wide setup and teardown paired even if an exception is
// thrown before main() reaches the end.
class CurlGlobalGuard {
  public:
    CurlGlobalGuard() {
        // libcurl requires one global init before any easy handles are used.
        const CURLcode result = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (result != CURLE_OK) {
            throw std::runtime_error("curl_global_init failed: " + std::string(curl_easy_strerror(result)));
        }
    }

    ~CurlGlobalGuard() {
        // Matching cleanup keeps the example disciplined and leak-free.
        curl_global_cleanup();
    }
};

// Serializes console writes from the progress thread and worker threads.
// Without this mutex, carriage-return progress updates and completion messages
// would frequently overlap and produce unreadable output.
class Console {
  public:
    void printStatus(const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Status output is rewritten in place so progress stays readable even
        // while worker threads are finishing in the background.
        std::cout << "\r\033[2K" << message << std::flush;
    }

    void printEvent(const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Clear the active progress line before printing a durable event.
        // Without the clear, thread completion messages and percentages tend
        // to interleave and make the demo look less professional.
        std::cout << "\r\033[2K" << message << '\n' << std::flush;
    }

    void finishLine(const std::string& message = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "\r\033[2K";
        if (!message.empty()) {
            std::cout << message;
        }
        std::cout << '\n' << std::flush;
    }

  private:
    std::mutex mutex_;
};

// Tracks total progress across every active worker.
// The only shared state here is a byte counter, so atomics are enough and no
// heavier synchronization is required.
class ProgressTracker {
  public:
    explicit ProgressTracker(curl_off_t total_bytes)
        : total_bytes_(total_bytes), started_at_(std::chrono::steady_clock::now()) {}

    void addBytes(curl_off_t delta) {
        if (delta > 0) {
            // Relaxed ordering is fine because this value is used only for the
            // UI, not for coordinating correctness-sensitive state.
            downloaded_bytes_.fetch_add(delta, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] curl_off_t downloadedBytes() const {
        return downloaded_bytes_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] curl_off_t totalBytes() const {
        return total_bytes_;
    }

    [[nodiscard]] double elapsedSeconds() const {
        const auto elapsed = std::chrono::steady_clock::now() - started_at_;
        return std::chrono::duration<double>(elapsed).count();
    }

  private:
    // Aggregate bytes downloaded by all workers.
    std::atomic<curl_off_t> downloaded_bytes_{0};
    // Expected total bytes, when the server revealed them.
    curl_off_t total_bytes_ = 0;
    // Transfer start time used to estimate throughput.
    std::chrono::steady_clock::time_point started_at_;
};

// Per-transfer state passed into libcurl's write callback.
struct StreamWriter {
    // Output file owned by one worker thread.
    std::ofstream stream;
    // Shared cancellation flag that lets one failure stop all workers.
    std::atomic_bool* cancel_requested = nullptr;
};

// Per-transfer state passed into libcurl's progress callback.
struct ChunkProgressContext {
    // Global tracker shared by all workers.
    ProgressTracker* tracker = nullptr;
    // Shared failure flag checked by every active request.
    std::atomic_bool* cancel_requested = nullptr;
    // Last cumulative byte count seen from libcurl for this request.
    curl_off_t last_reported = 0;
};

// Result of one probe attempt.
// The downloader may try more than one strategy before it knows enough to plan
// the transfer.
struct ProbeAttempt {
    DownloadMetadata metadata;
    CURLcode curl_result = CURLE_OK;
    long response_code = 0;
};

// Unique-pointer alias for libcurl easy handles.
// The custom deleter guarantees cleanup on every return path.
using CurlHandle = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>;

// Convert text to lowercase so header parsing can ignore case differences.
std::string toLower(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

// Trim surrounding whitespace from text read from headers or user input.
std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }

    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

// Parse a non-negative integer from a header fragment into curl's offset type.
// Invalid data is treated as "not present" instead of an exception because
// network inputs should be handled defensively.
std::optional<curl_off_t> parseCurlOffset(std::string_view text) {
    if (text.empty() || text == "*") {
        return std::nullopt;
    }

    curl_off_t value = 0;
    const auto [ptr, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || ptr != text.data() + text.size() || value < 0) {
        return std::nullopt;
    }

    return value;
}

// Replace filesystem-hostile characters in an inferred filename.
std::string sanitizeFilename(std::string filename) {
    for (char& ch : filename) {
        if (ch == '/' || ch == '\\' || ch == ':' || ch == '*' || ch == '?' || ch == '"' || ch == '<' || ch == '>' || ch == '|') {
            ch = '_';
        }
    }
    return filename.empty() ? "download.bin" : filename;
}

// Build a default output filename from the URL path.
std::string inferFilenameFromUrl(const std::string& url) {
    // Query parameters are stripped because they are not part of the actual
    // filename and often contain tokens or cache-busting data.
    const auto without_query = url.substr(0, url.find('?'));
    const auto slash = without_query.find_last_of('/');

    if (slash == std::string::npos || slash == without_query.size() - 1) {
        return "download.bin";
    }

    return sanitizeFilename(without_query.substr(slash + 1));
}

// Render a byte count in human-friendly units for the CLI.
std::string formatBytes(double bytes) {
    static constexpr const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    std::size_t suffix_index = 0;

    while (bytes >= 1024.0 && suffix_index + 1 < std::size(suffixes)) {
        // Keep scaling the value until it fits comfortably in the chosen unit.
        bytes /= 1024.0;
        ++suffix_index;
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(bytes < 10.0 && suffix_index > 0 ? 2 : 1) << bytes << ' ' << suffixes[suffix_index];
    return stream.str();
}

// Build the one-line progress string shown during the download.
std::string buildProgressLine(const ProgressTracker& tracker) {
    // Clamp the denominator so the first refresh cannot divide by zero.
    const double elapsed = std::max(tracker.elapsedSeconds(), 0.001);
    const double speed = static_cast<double>(tracker.downloadedBytes()) / elapsed;

    std::ostringstream stream;
    if (tracker.totalBytes() > 0) {
        // When the total size is known, percentage plus raw bytes is the most
        // useful status for a download demo.
        const double percent = (static_cast<double>(tracker.downloadedBytes()) * 100.0) / static_cast<double>(tracker.totalBytes());
        stream << "Downloading... " << std::fixed << std::setprecision(1) << std::min(percent, 100.0) << "% ("
               << formatBytes(static_cast<double>(tracker.downloadedBytes())) << " / "
               << formatBytes(static_cast<double>(tracker.totalBytes())) << ", "
               << formatBytes(speed) << "/s)";
    } else {
        // A true percentage requires a content length. If the server does not
        // provide one, showing bytes and speed is still more honest than
        // pretending we know how far along the transfer is.
        stream << "Downloading... " << formatBytes(static_cast<double>(tracker.downloadedBytes()))
               << " transferred (" << formatBytes(speed) << "/s)";
    }

    return stream.str();
}

// Write callback used by real download requests.
size_t writeToStream(char* data, size_t size, size_t count, void* userdata) {
    auto* writer = static_cast<StreamWriter*>(userdata);
    if (writer->cancel_requested->load(std::memory_order_relaxed)) {
        // Returning 0 asks libcurl to abort the transfer immediately.
        return 0;
    }

    const std::size_t total_bytes = size * count;
    writer->stream.write(data, static_cast<std::streamsize>(total_bytes));
    return writer->stream ? total_bytes : 0;
}

// Write callback used by metadata probes when the body should be ignored.
size_t discardBody(char*, size_t size, size_t count, void*) {
    return size * count;
}

// Progress callback that converts libcurl's per-request counters into updates on
// the shared global progress tracker.
int transferProgress(void* clientp, curl_off_t, curl_off_t downloaded_now, curl_off_t, curl_off_t) {
    auto* context = static_cast<ChunkProgressContext*>(clientp);

    if (context->cancel_requested->load(std::memory_order_relaxed)) {
        // Non-zero return values tell libcurl to cancel this request.
        return 1;
    }

    const curl_off_t delta = downloaded_now - context->last_reported;
    if (delta > 0) {
        context->tracker->addBytes(delta);
        context->last_reported = downloaded_now;
    }

    return 0;
}

// Header callback that extracts only the metadata relevant to chunk planning.
size_t captureHeaders(char* buffer, size_t size, size_t items, void* userdata) {
    auto* metadata = static_cast<DownloadMetadata*>(userdata);
    const std::string header = trim(std::string(buffer, size * items));
    const std::string lower = toLower(header);

    if (lower.rfind("accept-ranges:", 0) == 0 && lower.find("bytes") != std::string::npos) {
        metadata->accepts_ranges = true;
    }

    if (lower.rfind("content-range:", 0) == 0) {
        const auto slash = header.find('/');
        if (slash != std::string::npos) {
            const std::string total_text = trim(header.substr(slash + 1));
            if (const auto total = parseCurlOffset(total_text)) {
                // When a server replies to `Range: bytes=0-0`, the normal
                // content-length is only one byte. Parsing Content-Range lets
                // us recover the total file size without downloading the file.
                metadata->content_length = *total;
                metadata->accepts_ranges = true;
            }
        }
    }

    return size * items;
}

void configureCommonCurlOptions(CURL* curl, const std::string& url) {
    // These options are common to probes and real downloads, so centralizing
    // them keeps the behavior consistent across every request.
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "identity");
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "cpp-showcase-downloader/1.0");
}

// First probe strategy: try an HTTP HEAD request.
ProbeAttempt performHeadProbe(const std::string& url) {
    CurlHandle curl(curl_easy_init(), &curl_easy_cleanup);
    if (!curl) {
        throw std::runtime_error("Failed to create CURL handle while probing the URL.");
    }

    // The returned structure carries both protocol-level status and the parsed
    // metadata so the caller can decide whether a fallback probe is needed.
    ProbeAttempt attempt;
    configureCommonCurlOptions(curl.get(), url);
    curl_easy_setopt(curl.get(), CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, captureHeaders);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &attempt.metadata);

    // HEAD is the cheapest probe because it asks only for metadata. Some
    // servers reject HEAD though, so probeUrl() treats this as a first attempt
    // rather than a hard requirement.
    attempt.curl_result = curl_easy_perform(curl.get());
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &attempt.response_code);

    curl_off_t content_length = -1;
    curl_easy_getinfo(curl.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);
    // Some servers provide the total size directly in the HEAD response.
    attempt.metadata.content_length = std::max(attempt.metadata.content_length, std::max<curl_off_t>(0, content_length));
    return attempt;
}

// Second probe strategy: request only byte 0.
ProbeAttempt performRangeProbe(const std::string& url) {
    CurlHandle curl(curl_easy_init(), &curl_easy_cleanup);
    if (!curl) {
        throw std::runtime_error("Failed to create CURL handle while probing the URL.");
    }

    ProbeAttempt attempt;
    configureCommonCurlOptions(curl.get(), url);
    curl_easy_setopt(curl.get(), CURLOPT_RANGE, "0-0");
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, discardBody);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, captureHeaders);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &attempt.metadata);

    // Asking for only the first byte is a pragmatic fallback when HEAD is
    // blocked. A compliant server responds with 206 and a Content-Range header,
    // which is enough to infer both size and byte-range support.
    attempt.curl_result = curl_easy_perform(curl.get());
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &attempt.response_code);

    curl_off_t content_length = -1;
    curl_easy_getinfo(curl.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);
    if (attempt.metadata.content_length <= 0) {
        // This is weaker than Content-Range, but still worth using when it is
        // all the server reveals.
        attempt.metadata.content_length = std::max<curl_off_t>(0, content_length);
    }
    if (attempt.response_code == 206) {
        attempt.metadata.accepts_ranges = true;
    }

    return attempt;
}

// Normalize the raw probe information into the stricter invariants used by the
// rest of the downloader.
DownloadMetadata finalizeMetadata(DownloadMetadata metadata) {
    metadata.content_length = std::max<curl_off_t>(0, metadata.content_length);

    // Multiple workers are only useful when we can assign disjoint byte
    // ranges. If the server does not provide a reliable total size, the code
    // falls back to a single transfer instead of guessing chunk boundaries.
    metadata.accepts_ranges = metadata.accepts_ranges && metadata.content_length > 0;
    return metadata;
}

// Treat only transport-successful, HTTP-successful probes as usable.
[[nodiscard]] bool isSuccessfulResponse(const ProbeAttempt& attempt) {
    return attempt.curl_result == CURLE_OK && attempt.response_code >= 200 && attempt.response_code < 400;
}

// Turn a probe result into a user-facing error string.
// HTTP failures and transport failures need different wording because curl may
// succeed at the network level even when the server returns an error response.
std::string describeProbeFailure(std::string_view label, const ProbeAttempt& attempt) {
    std::ostringstream stream;
    stream << label << " ";

    if (attempt.curl_result != CURLE_OK) {
        stream << "failed: " << curl_easy_strerror(attempt.curl_result);
    } else if (attempt.response_code >= 400) {
        stream << "returned HTTP " << attempt.response_code;
    } else {
        stream << "did not provide enough metadata";
    }

    return stream.str();
}

// Decide which probe strategy to use and merge the metadata they discover.
DownloadMetadata probeUrl(const std::string& url) {
    const ProbeAttempt head_attempt = performHeadProbe(url);
    DownloadMetadata merged_metadata = head_attempt.metadata;

    // Some servers reject HEAD or omit key headers on HEAD, so a tiny ranged
    // GET is used as a recovery path.
    const bool needs_fallback_probe = !isSuccessfulResponse(head_attempt) || head_attempt.response_code == 405 ||
                                      merged_metadata.content_length <= 0 || !merged_metadata.accepts_ranges;

    if (needs_fallback_probe) {
        const ProbeAttempt range_attempt = performRangeProbe(url);
        if (isSuccessfulResponse(range_attempt)) {
            if (range_attempt.metadata.content_length > 0) {
                merged_metadata.content_length = range_attempt.metadata.content_length;
            }
            merged_metadata.accepts_ranges = merged_metadata.accepts_ranges || range_attempt.metadata.accepts_ranges;
            return finalizeMetadata(merged_metadata);
        }

        if (!isSuccessfulResponse(head_attempt)) {
            // If both probes failed, surface the fallback error because it was
            // the final attempt to recover a working plan.
            throw std::runtime_error("Failed to inspect remote file: " + describeProbeFailure("HEAD probe", head_attempt) + "; " +
                                     describeProbeFailure("range probe", range_attempt));
        }
    }

    if (!isSuccessfulResponse(head_attempt)) {
        throw std::runtime_error("Failed to inspect remote file: " + describeProbeFailure("HEAD probe", head_attempt));
    }

    return finalizeMetadata(merged_metadata);
}

// Split the file into one task per worker thread.
std::vector<ChunkTask> buildChunkPlan(const DownloadMetadata& metadata, std::size_t requested_threads, const fs::path& temp_dir) {
    std::vector<ChunkTask> tasks;

    if (!metadata.accepts_ranges || metadata.content_length <= 0) {
        // Without reliable chunk boundaries, the safest plan is a single worker
        // that downloads the whole file.
        tasks.push_back(ChunkTask{0, 0, 0, false, temp_dir / "part_0.bin"});
        return tasks;
    }

    // The worker count is capped so the program never spawns more threads than
    // there are bytes to fetch.
    const std::size_t thread_count = std::max<std::size_t>(
        1,
        std::min<std::size_t>(requested_threads, static_cast<std::size_t>(metadata.content_length)));

    const curl_off_t base_chunk_size = metadata.content_length / static_cast<curl_off_t>(thread_count);
    const curl_off_t remainder = metadata.content_length % static_cast<curl_off_t>(thread_count);

    curl_off_t cursor = 0;
    for (std::size_t index = 0; index < thread_count; ++index) {
        // Any remainder is distributed one byte at a time to the first chunks.
        // That keeps the work balanced while guaranteeing the chunks still
        // cover the file exactly once with no gaps and no overlaps.
        const curl_off_t chunk_size = base_chunk_size + (static_cast<curl_off_t>(index) < remainder ? 1 : 0);
        const curl_off_t start = cursor;
        const curl_off_t end = start + chunk_size - 1;

        tasks.push_back(ChunkTask{index, start, end, true, temp_dir / ("part_" + std::to_string(index) + ".bin")});
        cursor = end + 1;
    }

    return tasks;
}

void mergeChunks(const std::vector<ChunkTask>& tasks, const fs::path& output_file) {
    if (output_file.has_parent_path()) {
        // Create parent directories on demand so the caller can pass nested
        // output paths without preparing the directory tree first.
        fs::create_directories(output_file.parent_path());
    }

    std::ofstream output(output_file, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Failed to open output file: " + output_file.string());
    }

    std::vector<char> buffer(1 << 16);

    for (const ChunkTask& task : tasks) {
        // Chunks are merged in task order so the final file layout matches the
        // original byte order exactly.
        std::ifstream input(task.temp_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("Failed to open downloaded chunk: " + task.temp_path.string());
        }

        // Chunks are merged sequentially because that keeps correctness obvious:
        // thread coordination ends before file assembly starts, so the merge has
        // no race conditions and no shared file pointer to manage.
        while (input) {
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const auto bytes_read = input.gcount();
            if (bytes_read > 0) {
                output.write(buffer.data(), bytes_read);
            }
        }

        if (!output) {
            throw std::runtime_error("Failed while writing merged output file: " + output_file.string());
        }
    }
}

// Parse the CLI thread-count argument and reject malformed input early.
std::size_t parseThreadCount(const std::string& text) {
    std::size_t parsed_length = 0;
    unsigned long long value = 0;

    try {
        value = std::stoull(text, &parsed_length);
    } catch (...) {
        throw std::runtime_error("Thread count must be a positive integer.");
    }

    if (parsed_length != text.size() || value == 0 || value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("Thread count must be a positive integer.");
    }

    return static_cast<std::size_t>(value);
}

// Coordinator for the full download lifecycle.
class MultiThreadedDownloader {
  public:
    MultiThreadedDownloader(std::string url, std::size_t thread_count, fs::path output_path)
        : url_(std::move(url)), thread_count_(std::max<std::size_t>(1, thread_count)), output_path_(std::move(output_path)) {}

    void run() {
        // Probe before creating worker tasks so chunk planning is based on real
        // server capabilities instead of assumptions.
        metadata_ = probeUrl(url_);
        temp_directory_ = output_path_;
        temp_directory_ += ".parts";

        if (fs::exists(temp_directory_)) {
            // Remove leftovers from any previous interrupted run using the same
            // output path so stale chunks cannot be merged accidentally.
            fs::remove_all(temp_directory_);
        }
        fs::create_directories(temp_directory_);

        const std::vector<ChunkTask> tasks = buildChunkPlan(metadata_, thread_count_, temp_directory_);
        ProgressTracker tracker(metadata_.content_length);
        std::atomic_bool cancel_requested{false};
        std::atomic_bool stop_progress{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;

        console_.printEvent("Starting download with " + std::to_string(tasks.size()) + " thread(s).");
        if (tasks.size() == 1 && !metadata_.accepts_ranges) {
            console_.printEvent("Server does not support ranged downloads. Falling back to a single stream.");
        }

        // The progress display runs independently of the network callbacks so
        // the UI keeps updating at a predictable cadence.
        std::jthread progress_thread([&] {
            while (!stop_progress.load(std::memory_order_relaxed)) {
                console_.printStatus(buildProgressLine(tracker));
                std::this_thread::sleep_for(120ms);
            }
            console_.printStatus(buildProgressLine(tracker));
        });

        std::vector<std::thread> workers;
        workers.reserve(tasks.size());

        for (const ChunkTask& task : tasks) {
            workers.emplace_back([&, task] {
                try {
                    // Each worker owns its own CURL handle and temporary file.
                    // That avoids sharing non-thread-safe state and keeps the
                    // thread synchronization surface small.
                    downloadChunk(task, tracker, cancel_requested);
                    console_.printEvent("Thread " + std::to_string(task.index + 1) + " complete");
                } catch (...) {
                    // One bad chunk invalidates the whole final file, so the
                    // first failure asks every worker to shut down.
                    cancel_requested.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_error) {
                        first_error = std::current_exception();
                    }
                }
            });
        }

        for (std::thread& worker : workers) {
            worker.join();
        }

        stop_progress.store(true, std::memory_order_relaxed);
        progress_thread.join();
        console_.finishLine();

        if (first_error) {
            cleanupTemporaryFiles();
            std::rethrow_exception(first_error);
        }

        try {
            // Only merge after every worker has finished and every chunk file
            // has been fully flushed to disk.
            mergeChunks(tasks, output_path_);
        } catch (...) {
            cleanupTemporaryFiles();
            if (fs::exists(output_path_)) {
                // Remove the partial output file so failures never leave behind
                // something that looks valid but is only half-written.
                fs::remove(output_path_);
            }
            throw;
        }
        cleanupTemporaryFiles();
        console_.printEvent("File downloaded successfully: " + output_path_.string());
    }

  private:
    // Sanity-check the size of the chunk written to disk.
    void verifyChunkSize(const ChunkTask& task) const {
        if (!fs::exists(task.temp_path)) {
            throw std::runtime_error("Downloaded chunk is missing on disk: " + task.temp_path.string());
        }

        const curl_off_t actual_size = static_cast<curl_off_t>(fs::file_size(task.temp_path));
        const std::optional<curl_off_t> expected_size =
            task.use_range ? std::optional<curl_off_t>(task.end - task.start + 1)
                           : (metadata_.content_length > 0 ? std::optional<curl_off_t>(metadata_.content_length) : std::nullopt);

        if (expected_size && actual_size != *expected_size) {
            throw std::runtime_error("Chunk " + std::to_string(task.index + 1) + " size mismatch. Expected " +
                                     std::to_string(*expected_size) + " bytes but received " + std::to_string(actual_size) +
                                     " bytes.");
        }
    }

    // Download one chunk and store it in that chunk's temporary file.
    void downloadChunk(const ChunkTask& task, ProgressTracker& tracker, std::atomic_bool& cancel_requested) const {
        CurlHandle curl(curl_easy_init(), &curl_easy_cleanup);
        if (!curl) {
            throw std::runtime_error("Failed to create CURL handle for chunk " + std::to_string(task.index + 1));
        }

        // The writer object is owned entirely by this worker thread.
        StreamWriter writer;
        writer.cancel_requested = &cancel_requested;
        writer.stream.open(task.temp_path, std::ios::binary | std::ios::trunc);
        if (!writer.stream) {
            throw std::runtime_error("Failed to create temporary chunk file: " + task.temp_path.string());
        }

        // The progress context lets libcurl report progress without touching
        // any global variables.
        ChunkProgressContext progress_context{&tracker, &cancel_requested, 0};

        configureCommonCurlOptions(curl.get(), url_);
        curl_easy_setopt(curl.get(), CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, writeToStream);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &writer);
        curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, transferProgress);
        curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &progress_context);
        curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);

        if (task.use_range) {
            // Each worker gets an inclusive byte interval such as "0-1023".
            const std::string range = std::to_string(task.start) + "-" + std::to_string(task.end);
            curl_easy_setopt(curl.get(), CURLOPT_RANGE, range.c_str());
        }

        const CURLcode result = curl_easy_perform(curl.get());
        if (result != CURLE_OK) {
            throw std::runtime_error(
                "Chunk " + std::to_string(task.index + 1) + " failed: " + std::string(curl_easy_strerror(result)));
        }

        long response_code = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &response_code);

        if (task.use_range && response_code != 206) {
            // A downloader must distrust the network. If a server answers a
            // ranged request with 200 OK, it may have returned the whole file.
            // Treating that as success would silently corrupt the final merge.
            throw std::runtime_error("Server ignored range request for chunk " + std::to_string(task.index + 1));
        }

        writer.stream.flush();
        writer.stream.close();
        if (!writer.stream) {
            throw std::runtime_error("Failed to finalize chunk file: " + task.temp_path.string());
        }

        // Network APIs can report success even when the result is incomplete
        // because the transfer ended early but still produced a valid response
        // code. Verifying the on-disk byte count is a cheap last line of defense.
        verifyChunkSize(task);
    }

    // Remove the temporary chunk directory once it is no longer needed.
    void cleanupTemporaryFiles() const {
        if (fs::exists(temp_directory_)) {
            fs::remove_all(temp_directory_);
        }
    }

    // URL of the remote file.
    std::string url_;
    // Requested maximum worker count.
    std::size_t thread_count_;
    // Final destination path for the merged file.
    fs::path output_path_;
    // Probe result used to choose between ranged and single-stream download.
    DownloadMetadata metadata_;
    // Directory that stores partial chunk files during the transfer.
    fs::path temp_directory_;
    // Shared console serializer for all output.
    mutable Console console_;
};

// Print short usage text for CLI and interactive users.
void printUsage() {
    std::cout << "Usage: ./bin/downloader [url] [threads] [output-file]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        // Global libcurl initialization must happen before any easy handles are
        // created by the downloader.
        CurlGlobalGuard curl_guard;

        std::string url;
        std::size_t thread_count = 4;
        fs::path output_path;

        if (argc >= 2) {
            // Command-line arguments make the program easy to automate.
            url = argv[1];
        } else {
            // Interactive prompts make the demo easy to record and explain.
            std::cout << "Enter URL: ";
            std::getline(std::cin, url);
        }

        if (url.empty()) {
            printUsage();
            return 1;
        }

        if (argc >= 3) {
            thread_count = parseThreadCount(argv[2]);
        } else {
            std::cout << "Threads [default 4]: ";
            std::string input;
            std::getline(std::cin, input);
            if (!input.empty()) {
                thread_count = parseThreadCount(input);
            }
        }

        if (argc >= 4) {
            // Use the explicit output path when provided.
            output_path = argv[3];
        } else {
            // Otherwise derive a safe filename from the URL.
            output_path = inferFilenameFromUrl(url);
        }

        MultiThreadedDownloader downloader(url, thread_count, output_path);
        downloader.run();
        return 0;
    } catch (const std::exception& error) {
        // A single top-level error handler keeps failures readable for the user.
        std::cerr << "Download failed: " << error.what() << '\n';
        return 1;
    }
}
