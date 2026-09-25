defmodule M0Probe.MixProject do
  use Mix.Project

  def project do
    [
      app: :m0_probe,
      version: "0.1.0",
      elixir: "~> 1.17",
      erlc_paths: ["src", "../../lib/babytalk/src"],
      deps: [{:exatomvm, git: "https://github.com/atomvm/exatomvm.git", runtime: false}],
      atomvm: [start: :m0_probe, flash_offset: 0x390000]
    ]
  end

  def application, do: []
end
