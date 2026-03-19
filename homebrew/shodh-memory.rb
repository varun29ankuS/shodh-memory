# Reference Homebrew formula for shodh-memory.
#
# To use:
#   1. Create a tap repo: github.com/varun29ankuS/homebrew-shodh-memory
#   2. Copy this file there as Formula/shodh-memory.rb
#   3. Update the version, URLs, and SHA256 hashes for each release
#
# Users install with:
#   brew tap varun29ankuS/shodh-memory
#   brew install shodh-memory

class ShodhMemory < Formula
  desc "Cognitive memory system for AI agents — local, private, neuroscience-inspired"
  homepage "https://github.com/varun29ankuS/shodh-memory"
  version "0.1.91"
  license "Apache-2.0"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/varun29ankuS/shodh-memory/releases/download/v#{version}/shodh-memory-macos-arm64.tar.gz"
      sha256 "15baa1cb6546fbd50e7e31d3865caf6b8a7d8188813179fbafe6707d839cd419"
    else
      url "https://github.com/varun29ankuS/shodh-memory/releases/download/v#{version}/shodh-memory-macos-x64.tar.gz"
      sha256 "6e4068f77f7abb5dc2cc3dd7ce56a276bf0a359b51e2ed04b923f6acdad6fad1"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/varun29ankuS/shodh-memory/releases/download/v#{version}/shodh-memory-linux-arm64.tar.gz"
      sha256 "d3a3dc2aedd853cebbbf82106e1d0039f071cfcfedfc84f6419577dc3d578ee3"
    else
      url "https://github.com/varun29ankuS/shodh-memory/releases/download/v#{version}/shodh-memory-linux-x64.tar.gz"
      sha256 "c07692e8f53d5b1515ae25e0035bc2db55ebf039e7051d3e877720b98b9718d9"
    end
  end

  def install
    bin.install "shodh"
    bin.install "shodh-memory-server"
    bin.install "shodh-tui"
    lib.install Dir["*.dylib"] if OS.mac?
    lib.install Dir["*.so*"] if OS.linux?
  end

  def post_install
    ohai "Run 'shodh init' to complete first-time setup"
  end

  def caveats
    <<~EOS
      Shodh-Memory has been installed. Get started:

        shodh init       # First-time setup (creates config, downloads AI model)
        shodh server     # Start the memory server
        shodh tui        # Launch the dashboard
        shodh status     # Check server health

      Claude Code integration:
        claude mcp add shodh-memory -- npx -y @shodh/memory-mcp

      Documentation: https://shodh-memory.com/docs
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/shodh version")
  end
end
