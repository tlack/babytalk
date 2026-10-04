defmodule SpeedBench.MixProject do
  use Mix.Project

  def project do
    [
      app: :speed_bench,
      version: "0.1.0",
      elixir: "~> 1.17",
      erlc_paths: ["src", "../../lib/babytalk/src"],
      deps: [
        {:exatomvm, git: "https://github.com/atomvm/exatomvm.git", runtime: false},
        # the AtomVM release's API, which recent exatomvm checks the app against
        {:atomvm, "~> 0.7.0-beta.0", runtime: false},
        # optional: mix atomvm.esp32.monitor and exatomvm's other esptool-based tasks
        {:pythonx, "~> 0.4.0", runtime: false}
      ],
      atomvm: [start: :speed_bench, flash_offset: 0xA90000]
    ]
  end

  def application, do: []
end
