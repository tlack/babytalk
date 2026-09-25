defmodule BabyTalk do
  @moduledoc """
  On-device speech to text for AtomVM on the ESP32-S3: a thin wrapper over the Erlang
  `:babytalk` module (see `atomvm/lib/babytalk/src/babytalk.erl` for the details).

      {:ok, text, info} = BabyTalk.transcribe_sync(pcm16_mono_16khz, 10_000)
  """
  defdelegate transcribe(pcm), to: :babytalk
  defdelegate transcribe_sync(pcm, timeout), to: :babytalk
  defdelegate listen(chunk_ms), to: :babytalk
  defdelegate stop_listening(), to: :babytalk
  defdelegate record(secs), to: :babytalk
  defdelegate phrase(spellings), to: :babytalk
  defdelegate cache(bytes), to: :babytalk
  defdelegate info(), to: :babytalk
  defdelegate heap_info(), to: :babytalk
end
