defmodule VoiceCommands do
  @moduledoc """
  A small voice assistant in Elixir: say "wake up tomato face", wait for the chime, then
  "hello", "count to three", "how long have you been up" or "how much memory do you have".
  Anything else is repeated back.

      VoiceCommands.Supervisor (rest_for_one)
      |-- VoiceCommands.Responder   turns each command into a reply
      `-- BabyTalk.Listener         mic + speaker: wake phrase -> command -> reply

  The responder starts first, so it is there to receive the listener's first event; if it
  crashes, the listener restarts with it (rest_for_one).
  """

  # The wake phrase as the model tends to hear it, with its threshold
  # (tools/wake_phrases/wake_up_tomato_face.json, from export/kws.py)
  @spellings [
    "wake up tomato face", "wake up to mato face", "wake up to meato face",
    "wake up to meatto face", "wake up to meadow face", "wake up t martr face",
    "wake up to marto face from", "wake up to matle face", "wake up to midto face",
    "wake up tomato faceo", "wake up tomato face a", "wake up tomato face f"
  ]

  def start do
    # Without an `audio:` option the listener uses the board the firmware was built for (the
    # Waveshare ESP32-S3-CAM or ESP32-P4-WIFI6). Another board: audio: :waveshare_p4_wifi6,
    # or a map of your pins (atomvm/README.md, "Boards and pins").
    listener = %{
      notify: VoiceCommands.Responder,
      spellings: @spellings,
      threshold: -21.0,
      greeting: "Yes?"
    }

    children = [VoiceCommands.Responder, {BabyTalk.Listener, listener}]
    {:ok, _sup} = Supervisor.start_link(children, strategy: :rest_for_one, name: VoiceCommands.Supervisor)
    IO.puts("voice_commands: say \"wake up tomato face\"")
    Process.sleep(:infinity)
  end
end
