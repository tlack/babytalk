%% BabyTalk: on-device speech to text and text to speech for AtomVM on the ESP32-S3 and ESP32-P4.
%%
%% Audio is 16 kHz mono signed 16-bit little-endian PCM in a binary. The model lives in the
%% "model" flash partition (an .mmrt image: int4 or int8 Citrinet-256). Inference takes about
%% 0.2 s per second of audio and runs on its own task, so transcribe/1 returns at once and
%% the result arrives as a message. One transcription at a time: others get {error, busy}.
-module(babytalk).
-export([transcribe/1, transcribe_sync/2, listen/1, stop_listening/0, record/1,
         say/1, say/2, say_sync/2, say_sync/3, play/2, play/3, tones/2, rms/1, speak/1, speak/2,
         phrase/1, cache/1, info/0, heap_info/0, reserve_at_boot/1,
         audio_config/0, audio_config/1, audio_preset/1, audio_presets/0]).
%% AtomVM binds NIFs only to external calls (Module:Fun), so NIFs wrapped here are exported
%% and called as ?MODULE:name_nif(...).
-export([transcribe_nif/1, listen_nif/1, say_nif/2, play_nif/3, tones_nif/2,
         audio_config_nif/1, audio_current_nif/0, audio_presets_nif/0]).

-type info() :: [{score, float() | undefined} | {span, {integer(), integer()}} | {frames, integer()}
                 | {fe_ms | model_ms | dec_ms, float()}].
-export_type([info/0, audio_config/0, switch/0]).

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

%% Stream the board's microphone (see audio_config/1) to the caller, gapless, in
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
    ok = ?MODULE:stop_listening(),
    drain(Ref),
    <<Pcm:Want/binary, _/binary>> = iolist_to_binary(lists:reverse(Acc)),
    {ok, Pcm};
collect(Ref, Want, Have, Acc) ->
    receive
        {babytalk_mic, Ref, Pcm} when is_binary(Pcm) -> collect(Ref, Want, Have + byte_size(Pcm), [Pcm | Acc]);
        {babytalk_mic, Ref, Other} -> {error, Other}
    after 5000 -> ok = ?MODULE:stop_listening(), {error, timeout}
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
%% Options: length_scale (1.0 = the voice's own pace; 1.1 = 10% slower, 0.5..2.0).
-spec say(iodata()) -> {ok, reference()} | {error, busy}.
say(Text) -> say(Text, #{}).
-spec say(iodata(), #{length_scale => float()}) -> {ok, reference()} | {error, busy}.
say(Text, Opts) -> ?MODULE:say_nif(Text, round(maps:get(length_scale, Opts, 1.0) * 1000)).

say_nif(_Text, _LengthPermille) -> erlang:nif_error(undefined).

-spec say_sync(iodata(), timeout()) -> {ok, binary(), list()} | {error, term()}.
say_sync(Text, Timeout) -> say_sync(Text, #{}, Timeout).
-spec say_sync(iodata(), map(), timeout()) -> {ok, binary(), list()} | {error, term()}.
say_sync(Text, Opts, Timeout) ->
    case say(Text, Opts) of
        {ok, Ref} -> receive {babytalk, Ref, R} -> R after Timeout -> {error, timeout} end;
        Error -> Error
    end.

%% Play mono PCM at Rate Hz on the board's speaker (see audio_config/1), Volume
%% 0..100 (DAC; 75 = 0 dB, each step ~0.5 dB; default 76 -- the speech peaks at 77% of full
%% scale, so much above ~80 clips in the DAC). Returns at once; then {babytalk_play, Ref, done |
%% {error, Code}}. The speaker and the microphone share one I2S port: {error, busy} while
%% listening (and listen/1 while playing).
-spec play(binary(), pos_integer()) -> {ok, reference()} | {error, busy | no_memory}.
play(Pcm, Rate) -> play(Pcm, Rate, 76).
-spec play(binary(), pos_integer(), 0..100) -> {ok, reference()} | {error, busy | no_memory}.
play(Pcm, Rate, Volume) -> ?MODULE:play_nif(Pcm, Rate, Volume).

play_nif(_Pcm, _Rate, _Volume) -> erlang:nif_error(undefined).

%% Play sine notes, e.g. a chime: [{Hz, Ms}] (Hz 0 = a rest; up to 24 notes, 6 s), Volume
%% 0..100. A note may also carry a Level (0..100 of its full loudness) and a Shape: flat (the
%% default: steady, 8 ms edges) or bloop (a soft rise, then a natural fall to near silence,
%% settling slightly in pitch -- for sounds meant to be ambient, not to call attention).
%% Made on the board; same messages and rules as play/3.
-type note() :: {non_neg_integer(), pos_integer()} | {non_neg_integer(), pos_integer(), 0..100}
              | {non_neg_integer(), pos_integer(), 0..100, flat | bloop}.
-spec tones([note()], 0..100) -> {ok, reference()} | {error, busy | no_memory}.
tones(Notes, Volume) -> ?MODULE:tones_nif(Notes, Volume).

tones_nif(_Notes, _Volume) -> erlang:nif_error(undefined).

%% Root-mean-square level of 16-bit PCM (0..32768): speech vs silence.
-spec rms(binary()) -> non_neg_integer().
rms(_Pcm) -> erlang:nif_error(undefined).

%% say + play, waiting for both: the board says Text out loud.
-spec speak(iodata()) -> ok | {error, term()}.
speak(Text) -> speak(Text, 76).
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

%% Memory facts; pool_bytes is the engines' internal-RAM block (0: not reserved at boot, so no
%% transcription or speech -- see reserve_at_boot/1).
-spec heap_info() -> [{internal_free | internal_largest | psram_free | psram_largest | pool_bytes, integer()}].
heap_info() -> erlang:nif_error(undefined).

%% Whether BabyTalk takes its 84.5 KB of internal RAM when the VM next starts (it has to be
%% early, before the heap fragments). Stored in NVS (namespace babytalk, key reserve); takes
%% effect after a restart; overrides the firmware's CONFIG_BABYTALK_RESERVE_AT_BOOT. Boards
%% short of internal RAM build with it off and let the app opt in.
-spec reserve_at_boot(boolean()) -> ok | {error, term()}.
reserve_at_boot(On) when is_boolean(On) ->
    esp:nvs_set_binary(babytalk, reserve, case On of true -> <<1>>; false -> <<0>> end).

%% ------------------------------------------------------------------ the board's audio pins
%% The microphone and speaker driver is configured at run time: one firmware per chip serves
%% any board built from the parts it knows --
%%   mic:      ES7210 (4-channel ADC) or ES8311 (its ADC), or none
%%   speaker:  ES8311 (its DAC), or none
%%   amp, power: a switch for the speaker amp (on only while playing) and for the audio
%%             power rail (on from bring-up): a GPIO, or a pin of an I2C IO expander --
%%             ch32v003 (Waveshare's, pins 0..7) or pca9555 (also TCA9555, XL9555; pins 0..15)
%% on any I2C and I2S pins. The firmware starts with its default board (Kconfig
%% BABYTALK_BOARD) but touches no pin before the first listen/play or audio_config/1, so
%% configure first, then use the mic and speaker. A board with other audio chips needs its
%% own C driver (CONFIG_BABYTALK_BOARD_CUSTOM; then audio_config/1 says {error, custom_board}).
%%
%%   babytalk:audio_config(waveshare_p4_wifi6)                          % a preset
%%   babytalk:audio_config(#{board => waveshare_p4_wifi6, amp => {gpio, 46}})   % preset + changes
%%   babytalk:audio_config(#{i2s => #{din => 21}})                       % change the board in use
%%   {ok, Map} = babytalk:audio_config()                                 % what's in force
%%
%% A change map is merged into the board in use (or into `board`'s preset), section by
%% section: #{i2s => #{din => 21}} keeps the other I2S pins. The full form:
%%   #{i2c => #{port => 0, sda => 8, scl => 7, speed_hz => 100000},
%%     i2s => #{port => 0, mclk => 10, bclk => 11, ws => 12, dout => 14, din => 13},
%%     mic => #{codec => es7210, address => 16#40, gain => default, slot => left},
%%     speaker => #{codec => es8311, address => 16#18},
%%     amp => {ch32v003, 16#24, 4},
%%     power => {ch32v003, 16#24, 6}}
%% Pins are GPIO numbers, or `none` (dout with no speaker, din with no mic, mclk if the
%% codec makes its own clock). gain is the codec's own step (ES7210 0..14, ES8311 0..7, 6 dB
%% each) or `default`; slot is the I2S slot the mic is on (left | right). An I2C bus that
%% something else already opened on that port (a camera, a display) is shared, and its pins
%% are whatever that driver set.
%%
%% audio_config/1 releases the old pins, brings the new board up and waits for it: ok, or
%%   {error, busy}                    the mic or speaker is in use (stop listening first)
%%   {error, {bad_value, Where}}      Where is [Section, Key] (e.g. [i2s, din]): not a valid
%%                                    pin/value on this chip
%%   {error, {unknown_key, Where}}    a typo, e.g. [i2s, dat]
%%   {error, {unknown_board, Name}}   no such preset (audio_presets/0 lists them)
%%   {error, i2c_bus | switch | mic_codec | speaker_codec}   bring-up failed there (a wrong
%%                                    pin or address: nothing answered)
%%   {error, custom_board}            this firmware has its own audio driver
-type switch() :: none | {gpio, non_neg_integer()} | {gpio, non_neg_integer(), active_low}
                | {ch32v003 | pca9555, 0..127, non_neg_integer()}
                | {ch32v003 | pca9555, 0..127, non_neg_integer(), active_low}.
-type audio_config() :: #{board => atom(), i2c => map(), i2s => map(), mic => map() | none,
                          speaker => map() | none, amp => switch(), power => switch()}.

-define(AUDIO_KEYS, [{i2c, [port, sda, scl, speed_hz]},
                     {i2s, [port, mclk, bclk, ws, dout, din]},
                     {mic, [codec, address, gain, slot]},
                     {speaker, [codec, address]},
                     {amp, switch}, {power, switch}]).

-spec audio_config() -> {ok, audio_config()} | {error, custom_board}.
audio_config() ->
    case ?MODULE:audio_current_nif() of
        undefined -> {error, custom_board};
        T -> {ok, from_tuple(T)}
    end.

-spec audio_config(atom() | audio_config()) -> ok | {error, term()}.
audio_config(Board) when is_atom(Board) -> audio_config(#{board => Board});
audio_config(Changes) when is_map(Changes) ->
    Base = case maps:find(board, Changes) of
               {ok, Board} -> audio_preset(Board);
               error -> audio_config()
           end,
    try
        {ok, Cfg} = Base,
        apply_audio(to_tuple(merge_audio(Cfg, maps:remove(board, Changes))))
    catch
        error:{badmatch, {error, _} = E} -> E;
        throw:E -> {error, E}
    end.

%% The boards this firmware knows by name.
-spec audio_presets() -> [atom()].
audio_presets() -> [Name || {Name, _} <- ?MODULE:audio_presets_nif()].

%% A preset's full configuration (a starting point for your own board's).
-spec audio_preset(atom()) -> {ok, audio_config()} | {error, {unknown_board, atom()}}.
audio_preset(Name) ->
    case lists:keyfind(Name, 1, ?MODULE:audio_presets_nif()) of
        {_, T} -> {ok, from_tuple(T)};
        false -> {error, {unknown_board, Name}}
    end.

audio_config_nif(_Tuple) -> erlang:nif_error(undefined).
audio_current_nif() -> erlang:nif_error(undefined).
audio_presets_nif() -> erlang:nif_error(undefined).

apply_audio(Tuple) ->
    case ?MODULE:audio_config_nif(Tuple) of
        {ok, Ref} ->
            receive
                {babytalk_audio, Ref, ok} -> ok;
                {babytalk_audio, Ref, {error, Code}} -> {error, init_error(Code)}
            after 5000 -> {error, timeout}
            end;
        {error, {bad_field, N}} -> {error, {bad_value, field_path(N)}};
        Error -> Error
    end.

init_error(-1) -> i2c_bus;
init_error(-2) -> switch;
init_error(-3) -> mic_codec;
init_error(-4) -> speaker_codec;
init_error(Code) -> Code.

merge_audio(Cfg, Changes) ->
    maps:fold(
        fun(Section, New, Acc) ->
            case lists:keyfind(Section, 1, ?AUDIO_KEYS) of
                false -> throw({unknown_key, [Section]});
                {_, switch} -> Acc#{Section => New};
                {_, Keys} ->
                    NewMap = case New of none -> #{codec => none}; _ when is_map(New) -> New;
                                 _ -> throw({bad_value, [Section]}) end,
                    [throw({unknown_key, [Section, K]}) || K <- maps:keys(NewMap), not lists:member(K, Keys)],
                    Acc#{Section => maps:merge(maps:get(Section, Acc), NewMap)}
            end
        end, Cfg, Changes).

%% The NIF's tuple: I2C port, SDA, SCL, Hz; I2S port, MCLK, BCLK, WS, DOUT, DIN; mic codec,
%% address, gain, slot; speaker codec, address; amp kind, address, pin, active_low; power the same
to_tuple(#{i2c := I2c, i2s := I2s, mic := Mic, speaker := Spk, amp := Amp, power := Power}) ->
    G = fun(Sec, M, K, Enc) -> enc(Enc, maps:get(K, M), [Sec, K]) end,
    list_to_tuple(
        [G(i2c, I2c, port, int), G(i2c, I2c, sda, pin), G(i2c, I2c, scl, pin), G(i2c, I2c, speed_hz, int),
         G(i2s, I2s, port, int), G(i2s, I2s, mclk, pin), G(i2s, I2s, bclk, pin), G(i2s, I2s, ws, pin),
         G(i2s, I2s, dout, pin), G(i2s, I2s, din, pin),
         G(mic, Mic, codec, codec), G(mic, Mic, address, int), G(mic, Mic, gain, gain), G(mic, Mic, slot, slot),
         G(speaker, Spk, codec, codec), G(speaker, Spk, address, int)]
        ++ switch_to(Amp, amp) ++ switch_to(Power, power)).

enc(int, V, _) when is_integer(V) -> V;
enc(pin, none, _) -> -1;
enc(pin, V, _) when is_integer(V), V >= 0 -> V;
enc(codec, none, _) -> 0;
enc(codec, es7210, _) -> 1;
enc(codec, es8311, _) -> 2;
enc(gain, default, _) -> -1;
enc(gain, V, _) when is_integer(V), V >= 0 -> V;
enc(slot, left, _) -> 0;
enc(slot, right, _) -> 1;
enc(_, _, Where) -> throw({bad_value, Where}).

switch_to(none, _) -> [0, 0, -1, 0];
switch_to({gpio, Pin}, _) when is_integer(Pin) -> [1, 0, Pin, 0];
switch_to({gpio, Pin, active_low}, _) when is_integer(Pin) -> [1, 0, Pin, 1];
switch_to({Exp, Addr, Pin}, W) -> switch_to({Exp, Addr, Pin, active_high}, W);
switch_to({Exp, Addr, Pin, Pol}, W) when is_integer(Addr), is_integer(Pin), (Pol =:= active_low orelse Pol =:= active_high) ->
    Kind = case Exp of ch32v003 -> 2; pca9555 -> 3; _ -> throw({bad_value, [W]}) end,
    [Kind, Addr, Pin, case Pol of active_low -> 1; active_high -> 0 end];
switch_to(_, W) -> throw({bad_value, [W]}).

from_tuple(T) ->
    [I2cPort, Sda, Scl, Hz, I2sPort, Mclk, Bclk, Ws, Dout, Din, MicC, MicA, MicG, MicS, SpkC, SpkA,
     AK, AA, AP, AL, PK, PA, PP, PL] = tuple_to_list(T),
    #{i2c => #{port => I2cPort, sda => Sda, scl => Scl, speed_hz => Hz},
      i2s => #{port => I2sPort, mclk => dec_pin(Mclk), bclk => Bclk, ws => Ws, dout => dec_pin(Dout),
               din => dec_pin(Din)},
      mic => #{codec => dec_codec(MicC), address => MicA, gain => case MicG of -1 -> default; G -> G end,
               slot => case MicS of 0 -> left; _ -> right end},
      speaker => #{codec => dec_codec(SpkC), address => SpkA},
      amp => switch_from([AK, AA, AP, AL]), power => switch_from([PK, PA, PP, PL])}.

dec_pin(-1) -> none;
dec_pin(P) -> P.

dec_codec(1) -> es7210;
dec_codec(2) -> es8311;
dec_codec(_) -> none.

switch_from([0, _, _, _]) -> none;
switch_from([1, _, Pin, 0]) -> {gpio, Pin};
switch_from([1, _, Pin, 1]) -> {gpio, Pin, active_low};
switch_from([K, Addr, Pin, Low]) ->
    Exp = case K of 2 -> ch32v003; _ -> pca9555 end,
    case Low of 1 -> {Exp, Addr, Pin, active_low}; _ -> {Exp, Addr, Pin} end.

%% the C side's field number (1-based, board_audio_check) -> [Section, Key]
field_path(N) when N >= 17, N =< 20 -> [amp];
field_path(N) when N >= 21 -> [power];
field_path(N) ->
    lists:nth(N, [[i2c, port], [i2c, sda], [i2c, scl], [i2c, speed_hz],
                  [i2s, port], [i2s, mclk], [i2s, bclk], [i2s, ws], [i2s, dout], [i2s, din],
                  [mic, codec], [mic, address], [mic, gain], [mic, slot],
                  [speaker, codec], [speaker, address]]).
