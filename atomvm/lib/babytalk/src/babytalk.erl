%% BabyTalk: on-device speech to text for AtomVM on the ESP32-S3.
%%
%% Audio is 16 kHz mono signed 16-bit little-endian PCM in a binary. The model lives in the
%% "model" flash partition (an .mmrt image: int4 or int8 Citrinet-256). Inference takes about
%% 0.2 s per second of audio and runs on its own task, so transcribe/1 returns at once and
%% the result arrives as a message. One transcription at a time: others get {error, busy}.
-module(babytalk).
-export([transcribe/1, transcribe_sync/2, phrase/1, cache/1, info/0, heap_info/0]).
%% AtomVM binds NIFs only to external calls (Module:Fun), so the NIF is exported and called
%% as ?MODULE:transcribe_nif/1.
-export([transcribe_nif/1]).

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
