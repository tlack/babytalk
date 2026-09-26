defmodule TtsDemo.MixProject do
  use Mix.Project

  def project do
    [
      app: :tts_demo,
      version: "0.1.0",
      elixir: "~> 1.17",
      erlc_paths: ["src", "../../lib/babytalk/src"],
      deps: [{:exatomvm, git: "https://github.com/atomvm/exatomvm.git", runtime: false}],
      atomvm: [start: :tts_demo, flash_offset: 0xA90000]
    ]
  end

  def application, do: []
end
