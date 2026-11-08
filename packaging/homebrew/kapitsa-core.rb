# Homebrew-core formula for Kapitsa (source build).
#
# The tap formula (nace-ai/tap/kapitsa) installs a pre-built binary and a bottled binary;
# homebrew-core requires a source build, so this formula builds from source with CMake, the
# same way the tap's `head do` block does. Both install the same `kapitsa` command, so
# `brew uninstall nace-ai/tap/kapitsa` before installing from homebrew-core.
#
# To submit: fork Homebrew/homebrew-core, add this as Formula/k/kapitsa.rb, open a PR.
# SHA256: curl -sL the tarball_url below, pipe to `shasum -a 256`.
class Kapitsa < Formula
  desc "Fast local LLM inference for Apple silicon, with an OpenAI-compatible server"
  homepage "https://github.com/nace-ai/kapitsa"
  url "https://github.com/nace-ai/kapitsa/archive/refs/tags/v0.2.31.tar.gz"
  sha256 "e70f48da1acfac82875a1fb1e4d323fd3f77de5f323338b71cf2eda19cca8539"
  license "Apache-2.0"

  livecheck do
    url :stable
    strategy :github_latest
  end

  depends_on "cmake" => :build
  depends_on arch: :arm64
  depends_on macos: :sonoma

  def install
    ENV.O3
    system "cmake", "-S", ".", "-B", "build", *std_cmake_args
    system "cmake", "--build", "build", "--target", "kapitsa", "systemone"

    bin.install "build/kapitsa"
    bin.install "build/systemone" => "kapitsa-systemone"
    bin.install "tools/drex_pointer.py" => "kapitsa-drex-pointer"
    pkgshare.install "LICENSE", "THIRD_PARTY_NOTICES"
    (pkgshare/"bench").install "bench/harness"
  end

  def caveats
    <<~EOS
      Start the best model for this Mac (the first run downloads it):
        kapitsa serve
      Then open http://127.0.0.1:8000 to chat, or point any OpenAI-compatible client at it.
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
    assert_match "usage: systemone", shell_output("#{bin}/kapitsa-systemone --help 2>&1")
    ENV["KAPITSA_HOME"] = testpath.to_s
    assert_match "qwen3.5-4b", shell_output("#{bin}/kapitsa pull --dry-run --json 4b")
  end
end
