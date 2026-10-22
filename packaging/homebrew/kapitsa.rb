# Homebrew formula for Kapitsa (by Nace AI, https://nace.ai), prepared for a public release and not published yet.
# It installs the release binary that packaging/release/build.sh makes (the same tarball install.sh downloads), so
# nothing compiles; `brew install --HEAD` builds main from source instead. At each release the workflow
# .github/workflows/release.yml sets version, url and sha256 and puts this file in the tap nace-ai/homebrew-tap as
# Formula/kapitsa.rb, so that `brew install nace-ai/tap/kapitsa` works.
class Kapitsa < Formula
  desc "Fast local Qwen inference for Apple silicon, with an OpenAI-compatible server"
  homepage "https://github.com/nace-ai/kapitsa"
  url "https://github.com/nace-ai/kapitsa/releases/download/v0.2.5/kapitsa-0.2.5-darwin-arm64.tar.gz"
  version "0.2.5" # Homebrew reads "64" from this file name on hosts other than GitHub
  sha256 "0000000000000000000000000000000000000000000000000000000000000000" # set by the release workflow
  license "Apache-2.0"

  head do
    url "https://github.com/nace-ai/kapitsa.git", branch: "main"
    depends_on "cmake" => :build
  end

  depends_on arch: :arm64
  depends_on macos: :sonoma

  def install
    if build.head?
      ENV.O3 # Homebrew would compile with -Os instead of the build's -O3; the benchmarks ran -O3
      system "cmake", "-S", ".", "-B", "build", *std_cmake_args
      system "cmake", "--build", "build", "--target", "kapitsa"
      bin.install "build/kapitsa"
    else
      bin.install "kapitsa"
    end
    pkgshare.install "LICENSE", "THIRD_PARTY_NOTICES"
    # `kapitsa bench compare` runs this harness; find_harness() looks for it in pkgshare, so a brew install needs no
    # checkout and no KAPITSA_HARNESS
    (pkgshare/"bench").install "bench/harness"
  end

  def caveats
    <<~EOS
      Start the best model for this Mac (the first run downloads it):
        kapitsa serve
      Then open http://127.0.0.1:8000 to chat, or use the API at http://127.0.0.1:8000/v1
      (OpenAI chat completions and Responses, Anthropic Messages).
      Or start a coding agent on it in one command: kapitsa claude, kapitsa codex, kapitsa opencode, kapitsa pi.
      Measure it on this Mac and add yours to the community results: kapitsa bench --submit
      Models are stored in ~/.cache/kapitsa/models (KAPITSA_HOME moves them).
    EOS
  end

  service do
    run [opt_bin/"kapitsa", "serve"]
    keep_alive successful_exit: false
    log_path var/"log/kapitsa.log"
    error_log_path var/"log/kapitsa.log"
  end

  test do
    assert_match "kapitsa", shell_output("#{bin}/kapitsa --version")
    assert_match "kapitsa opencode", shell_output("#{bin}/kapitsa --help")
    # the harness ships with the formula: bench compare finds it with no checkout and no environment variable
    assert_match "kapitsa bench compare", shell_output("#{bin}/kapitsa bench compare --help")
    # model names resolve offline
    ENV["KAPITSA_HOME"] = testpath.to_s
    assert_match "qwen3.5-4b", shell_output("#{bin}/kapitsa pull --dry-run --json 4b")
  end
end
