defmodule MicStream.MixProject do
  use Mix.Project

  # The board and mic gain are compiled in (the app has no other input):
  #   BABYTALK_BOARD=lilygo_t_lora_pager BABYTALK_MIC_GAIN=4 mix atomvm.packbeam
  # Unset: the firmware's default board, at the codec's default gain.
  defp defines do
    for {var, macro} <- [{"BABYTALK_BOARD", :BOARD}, {"BABYTALK_MIC_GAIN", :MIC_GAIN}],
        val = System.get_env(var),
        val not in [nil, ""] do
      {:d, macro, if(macro == :MIC_GAIN, do: String.to_integer(val), else: String.to_atom(val))}
    end
  end

  def project do
    [
      app: :mic_stream,
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
      atomvm: [start: :mic_stream, flash_offset: 0xA90000]
    ]
  end

  def application, do: []
end
