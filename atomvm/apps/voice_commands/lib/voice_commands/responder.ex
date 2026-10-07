defmodule VoiceCommands.Responder do
  @moduledoc """
  Receives the listener's events and answers each command through it (the listener owns the
  speaker, and pauses listening while it talks so it doesn't hear itself).
  """
  use GenServer

  def start_link(_), do: GenServer.start_link(__MODULE__, nil, name: __MODULE__)

  @impl true
  def init(nil), do: {:ok, %{started: :erlang.monotonic_time(:second)}}

  # Defined here so `use GenServer` doesn't compile in its defaults, which call
  # :erlang.phash2/2, missing on AtomVM
  @impl true
  def handle_call(_request, _from, state), do: {:reply, :ok, state}

  @impl true
  def handle_cast(_request, state), do: {:noreply, state}

  @impl true
  def handle_info({:babytalk_listener, {:command, text}}, state) do
    IO.puts("command: " <> text)
    BabyTalk.Listener.say(:babytalk_listener, reply(text, state))
    {:noreply, state}
  end

  def handle_info({:babytalk_listener, {:wake, _text, score}}, state) do
    IO.puts("awake (score #{score})")
    {:noreply, state}
  end

  def handle_info({:babytalk_listener, _other}, state), do: {:noreply, state}

  # Transcripts are lower-case English with no punctuation. AtomVM's Elixir library has no
  # String module, so words are found with :binary.match.
  defp reply("", _state), do: "Sorry, I didn't catch that."

  defp reply(text, state) do
    cond do
      heard?(text, ["hello", "hi there"]) ->
        "Hello! Nice to hear you."

      heard?(text, ["count"]) ->
        "One. Two. Three."

      heard?(text, ["how long", "up time", "uptime"]) ->
        minutes = div(:erlang.monotonic_time(:second) - state.started, 60)
        "I've been listening for #{minutes} minutes."

      heard?(text, ["memory"]) ->
        kb = div(Keyword.get(BabyTalk.heap_info(), :psram_free), 1024)
        "I have #{kb} kilobytes of P S RAM free."

      true ->
        ["You said: ", text]
    end
  end

  defp heard?(text, words), do: Enum.any?(words, &(:binary.match(text, &1) != :nomatch))
end
