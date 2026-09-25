defmodule SttServer.MixProject do
  use Mix.Project

  def project do
    [
      app: :stt_server,
      version: "0.1.0",
      elixir: "~> 1.17",
      erlc_paths: ["src", "../../lib/babytalk/src"],
      deps: [{:exatomvm, git: "https://github.com/atomvm/exatomvm.git", runtime: false}],
      atomvm: [start: :stt_server, flash_offset: 0x490000]
    ]
  end

  def application, do: []
end
