defmodule ListenDemo.MixProject do
  use Mix.Project

  # The audio board is compiled in:  BABYTALK_BOARD=lilygo_t_lora_pager mix atomvm.packbeam
  # Unset: the firmware's default board.
  defp defines do
    case System.get_env("BABYTALK_BOARD") do
      val when val in [nil, ""] -> []
      val -> [{:d, :BOARD, String.to_atom(val)}]
    end
  end

  def project do
    [
      app: :listen_demo,
      version: "0.1.0",
      elixir: "~> 1.17",
      erlc_paths: ["src", "../../lib/babytalk/src"],
      erlc_options: defines(),
      deps: [
        {:exatomvm, git: "https://github.com/atomvm/exatomvm.git", runtime: false},
        # the AtomVM release's API, which recent exatomvm checks the app against
        {:atomvm, "~> 0.7.0-beta.0", runtime: false},
        # optional: mix atomvm.esp32.monitor and exatomvm's other esptool-based tasks
        {:pythonx, "~> 0.4.0", runtime: false}
      ],
      atomvm: [start: :listen_demo, flash_offset: 0xA90000]
    ]
  end

  def application, do: []
end
