# Homebrew formula for Kapitsa, prepared for a public release and not published anywhere yet.
# At release: tag vX.Y.Z, set `url` to that tag's tarball and `sha256` to its checksum
# (curl -sL <url> | shasum -a 256), then put this file in a tap (e.g. nace-ai/homebrew-tap as Formula/kapitsa.rb),
# so that `brew install nace-ai/tap/kapitsa` works. Until then, ./install.sh in a clone is the way to install.
class Kapitsa < Formula
  desc "Fast local Qwen inference for Apple silicon, with an OpenAI-compatible server"
  homepage "https://github.com/nace-ai/kapitsa"
  url "https://github.com/nace-ai/kapitsa/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000" # replace at release
  head "https://github.com/nace-ai/kapitsa.git", branch: "main"

  depends_on "cmake" => :build
  depends_on arch: :arm64
  depends_on macos: :sonoma

  def install
    system "cmake", "-S", ".", "-B", "build", *std_cmake_args
    system "cmake", "--build", "build", "--target", "kapitsa"
    bin.install "build/kapitsa"
  end

  def caveats
    <<~EOS
      Start the best model for this Mac (the first run downloads it):
        kapitsa serve
      Then open http://127.0.0.1:8000 to chat, or use the API at http://127.0.0.1:8000/v1.
      Models are stored in ~/.cache/kapitsa/models (KAPITSA_HOME moves them).
    EOS
  end

  test do
    assert_match "kapitsa", shell_output("#{bin}/kapitsa --version")
    # model names resolve offline
    ENV["KAPITSA_HOME"] = testpath.to_s
    assert_match "qwen3.5-4b", shell_output("#{bin}/kapitsa pull --dry-run --json 4b")
  end
end
