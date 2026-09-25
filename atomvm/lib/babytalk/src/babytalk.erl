%% BabyTalk: on-device speech to text for AtomVM on the ESP32-S3.
%%
%% Audio is 16 kHz mono signed 16-bit little-endian PCM in a binary. The model lives in the
%% "model" flash partition (an .mmrt image: int4 or int8 Citrinet-256). Inference takes about
%% 0.2 s per second of audio and runs on its own task, so transcribe/1 returns at once and
%% the result arrives as a message. One transcription at a time: others get {error, busy}.
-module(babytalk).
-export([transcribe/1, transcribe_sync/2, listen/1, stop_listening/0, record/1,
         say/1, say_sync/2, play/2, play/3, tones/2, rms/1, speak/1, speak/2,
         phrase/1, cache/1, info/0, heap_info/0]).
%% AtomVM binds NIFs only to external calls (Module:Fun), so NIFs wrapped here are exported
%% and called as ?MODULE:name_nif(...).
-export([transcribe_nif/1, listen_nif/1, say_nif/1, play_nif/3, tones_nif/2]).

-type info() :: [{score, float() | undefined} | {span, {integer(), integer()}} | {frames, integer()}
                 | {fe_ms | model_ms | dec_ms, float()}].
-export_type([info/0]).

%% Start transcribing Pcm. The caller later receives
%%   {babytalk, Ref, {ok, Text :: binary(), info()} | {error, Code :: integer()}}
%% Info's score is the wake phrase's (see phrase/1): 0.0 = the phrase is the best reading of
%% the audio, more negative = less likely; span is where it matched, in 80 ms frames.
-spec transcribe(binary()) -> {ok, reference()} | {error, busy | no_memory}.
transcribe(Pcm) -> ?MODULE:transcribe_nif(Pcm).

%% transcribe/1 and wait up to Timeout ms for the result.
-spec transcribe_sync(binary(), timeout()) -> {ok, binary(), info()} | {error, term()}.
transcribe_sync(Pcm, Timeout) ->
    case ?MODULE:transcribe_nif(Pcm) of
        {ok, Ref} ->
            receive {babytalk, Ref, Result} -> Result
            after Timeout -> {error, timeout}
            end;
        Error -> Error
    end.

transcribe_nif(_Pcm) -> erlang:nif_error(undefined).

%% Stream the microphone (Waveshare S3-CAM: ES7210, mic 1) to the caller, gapless, in
%% ChunkMs pieces of 16 kHz mono PCM, while transcriptions keep running:
%%   {babytalk_mic, Ref, Pcm}               every ChunkMs
%%   {babytalk_mic, Ref, stopped}           after stop_listening/0 (the last message)
%%   {babytalk_mic, Ref, {error, Code}}     on a capture failure (also the last message)
%% One listener at a time: others get {error, busy}.
-spec listen(pos_integer()) -> {ok, reference()} | {error, busy}.
listen(ChunkMs) -> ?MODULE:listen_nif(ChunkMs).

listen_nif(_ChunkMs) -> erlang:nif_error(undefined).

-spec stop_listening() -> ok.
stop_listening() -> erlang:nif_error(undefined).

%% Record Secs seconds (a number) from the microphone: 16 kHz mono PCM.
-spec record(number()) -> {ok, binary()} | {error, term()}.
record(Secs) ->
    Bytes = round(Secs * 32000) band (bnot 1),
    case listen(min(500, max(20, round(Secs * 1000)))) of
        {ok, Ref} -> collect(Ref, Bytes, 0, []);
        Error -> Error
    end.

collect(Ref, Want, Have, Acc) when Have >= Want ->
    ok = stop_listening(),
    drain(Ref),
    <<Pcm:Want/binary, _/binary>> = iolist_to_binary(lists:reverse(Acc)),
    {ok, Pcm};
collect(Ref, Want, Have, Acc) ->
    receive
        {babytalk_mic, Ref, Pcm} when is_binary(Pcm) -> collect(Ref, Want, Have + byte_size(Pcm), [Pcm | Acc]);
        {babytalk_mic, Ref, Other} -> {error, Other}
    after 5000 -> ok = stop_listening(), {error, timeout}
    end.

%% up to the stream's last message
drain(Ref) ->
    receive
        {babytalk_mic, Ref, Pcm} when is_binary(Pcm) -> drain(Ref);
        {babytalk_mic, Ref, _Last} -> ok
    after 5000 -> ok
    end.

%% Text to speech (sanoTTS, the en_us_e12nano voice): starts synthesizing Text (English;
%% a binary or string) and returns; the caller later receives
%%   {babytalk, Ref, {ok, Pcm24k :: binary(), Info} | {error, Code}}
%% with 24 kHz mono PCM and Info = [{rate, 24000}, {phonemes, N}, {g2p_ms, F}, {synth_ms, F}].
%% Shares the engine with transcription: one at a time ({error, busy}).
-spec say(iodata()) -> {ok, reference()} | {error, busy}.
say(Text) -> ?MODULE:say_nif(Text).

say_nif(_Text) -> erlang:nif_error(undefined).

-spec say_sync(iodata(), timeout()) -> {ok, binary(), list()} | {error, term()}.
say_sync(Text, Timeout) ->
    case say(Text) of
        {ok, Ref} -> receive {babytalk, Ref, R} -> R after Timeout -> {error, timeout} end;
        Error -> Error
    end.

%% Play mono PCM at Rate Hz on the speaker (Waveshare S3-CAM: ES8311 + NS4150B), Volume
%% 0..100 (DAC; 75 = 0 dB, default 85). Returns at once; then {babytalk_play, Ref, done |
%% {error, Code}}. The speaker and the microphone share one I2S port: {error, busy} while
%% listening (and listen/1 while playing).
-spec play(binary(), pos_integer()) -> {ok, reference()} | {error, busy | no_memory}.
play(Pcm, Rate) -> play(Pcm, Rate, 85).
-spec play(binary(), pos_integer(), 0..100) -> {ok, reference()} | {error, busy | no_memory}.
play(Pcm, Rate, Volume) -> ?MODULE:play_nif(Pcm, Rate, Volume).

play_nif(_Pcm, _Rate, _Volume) -> erlang:nif_error(undefined).

%% Play soft sine notes, e.g. a chime: [{Hz, Ms}] (Hz 0 = a rest; up to 16 notes, 3 s),
%% Volume 0..100. Made on the board; same messages and rules as play/3.
-spec tones([{non_neg_integer(), pos_integer()}], 0..100) -> {ok, reference()} | {error, busy | no_memory}.
tones(Notes, Volume) -> ?MODULE:tones_nif(Notes, Volume).

tones_nif(_Notes, _Volume) -> erlang:nif_error(undefined).

%% Root-mean-square level of 16-bit PCM (0..32768): speech vs silence.
-spec rms(binary()) -> non_neg_integer().
rms(_Pcm) -> erlang:nif_error(undefined).

%% say + play, waiting for both: the board says Text out loud.
-spec speak(iodata()) -> ok | {error, term()}.
speak(Text) -> speak(Text, 85).
-spec speak(iodata(), 0..100) -> ok | {error, term()}.
speak(Text, Volume) ->
    case say_sync(Text, 30000) of
        {ok, Pcm, Info} ->
            case play(Pcm, proplists:get_value(rate, Info), Volume) of
                {ok, Ref} -> receive {babytalk_play, Ref, done} -> ok; {babytalk_play, Ref, E} -> E
                             after 60000 -> {error, timeout} end;
                Error -> Error
            end;
        Error -> Error
    end.

%% Set the wake phrase to score each transcription against: one or more spellings of it
%% (binaries or strings; only letters count). [] clears it. Returns the number of token
%% sequences it expands to.
-spec phrase([binary() | string()]) -> {ok, non_neg_integer()} | {error, busy | no_memory}.
phrase(_Spellings) -> erlang:nif_error(undefined).

%% Keep up to Bytes of model weights in PSRAM (faster than flash: ~20 ms saved per MB per
%% transcription, at the cost of that much PSRAM). 0 releases. Returns the bytes cached.
-spec cache(non_neg_integer()) -> {ok, non_neg_integer()} | {error, term()}.
cache(_Bytes) -> erlang:nif_error(undefined).

%% Model and memory facts: model_bytes, cache_bytes, ops, int4_ops (0 until first use),
%% internal_free, psram_free.
-spec info() -> [{atom(), integer()}].
info() -> erlang:nif_error(undefined).

-spec heap_info() -> [{internal_free | internal_largest | psram_free | psram_largest, integer()}].
heap_info() -> erlang:nif_error(undefined).
