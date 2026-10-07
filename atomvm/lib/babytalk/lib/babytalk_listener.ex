defmodule BabyTalk.Listener do
  @moduledoc """
  The voice front end as a GenServer: a thin wrapper over the Erlang `:babytalk_listener`
  module (see `atomvm/lib/babytalk/src/babytalk_listener.erl` for the options and events).

  It owns the mic and the speaker, listens for a wake phrase, records and transcribes the
  message after it, and speaks replies, with chimes between the phases. Events arrive at
  `notify` as `{:babytalk_listener, event}`, e.g. `{:command, text}`.

      {:ok, l} = BabyTalk.Listener.start_link(%{notify: self(), spellings: ["hey computer"]})
      receive do
        {:babytalk_listener, {:command, text}} -> BabyTalk.Listener.say(l, ["You said ", text])
      end
  """
  defdelegate start_link(opts), to: :babytalk_listener
  defdelegate start_link(name, opts), to: :babytalk_listener
  defdelegate stop(server), to: :babytalk_listener
  defdelegate say(server, text), to: :babytalk_listener
  defdelegate chime(server, notes), to: :babytalk_listener
  defdelegate chime(server, notes, volume), to: :babytalk_listener
  defdelegate ask(server, prompt, opts), to: :babytalk_listener
  defdelegate wake_on(server, spellings), to: :babytalk_listener
  defdelegate hold(server, ms), to: :babytalk_listener
  defdelegate sentences(text), to: :babytalk_listener

  @doc "A child spec, so the listener can go straight into a supervisor's children."
  def child_spec(opts) do
    %{id: __MODULE__, start: {:babytalk_listener, :start_link, [:babytalk_listener, opts]}}
  end
end
