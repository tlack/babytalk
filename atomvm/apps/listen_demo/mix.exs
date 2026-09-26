defmodule ListenDemo.MixProject do
  use Mix.Project

  def project do
    [
      app: :listen_demo,
      version: "0.1.0",
      elixir: "~> 1.17",
      erlc_paths: ["src", "../../lib/babytalk/src"],
      deps: [{:exatomvm, git: "https://github.com/atomvm/exatomvm.git", runtime: false}],
      atomvm: [start: :listen_demo, flash_offset: 0xA90000]
    ]
  end

  def application, do: []
end
