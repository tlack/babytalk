defmodule MmrtTest.MixProject do
  use Mix.Project

  def project do
    [
      app: :mmrt_test,
      version: "0.1.0",
      elixir: "~> 1.17",
      erlc_paths: ["src", "../../lib/mmrt/src"],
      deps: [{:exatomvm, git: "https://github.com/atomvm/exatomvm.git", runtime: false}],
      atomvm: [start: :mmrt_test, flash_offset: 0x390000]
    ]
  end

  def application, do: []
end
