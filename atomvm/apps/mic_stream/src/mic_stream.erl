%% Streams the board's microphone to the computer over the USB console, nonstop, for
%% recording sessions (field/capture.py --board ... with a USB-streaming board): no WiFi, no
%% commands. The console carries text, so audio goes as lines:
%%
%%   BTMIC START <rate> <chunk_ms> <board>      once, and again after a restart
%%   A <seq> <base64 of chunk_ms of 16 kHz mono int16 LE PCM>
%%   BTMIC STAT <seq> <rms> <psram_free>        every 10 s
%%   BTMIC ERROR <what>                         then the stream restarts
%%
%% <seq> counts chunks from 0, so the reader can tell a lost line (the console drops output
%% when the computer stops reading) from silence. Other lines (ESP-IDF logs) are noise to it.
%% 16 kHz mono is 32 KB/s, 43 KB/s as base64: well within the chip's USB-Serial-JTAG.
-module(mic_stream).
-export([start/0]).

-define(CHUNK_MS, 100).
-define(RATE, 16000).

start() ->
    Board = configure(),
    stream(Board, 0).

-ifdef(BOARD).
-define(BOARD_CFG, #{board => ?BOARD}).
-else.
-define(BOARD_CFG, #{}).
-endif.
-ifdef(MIC_GAIN).
-define(GAIN_CFG, #{mic => #{gain => ?MIC_GAIN}}).
-else.
-define(GAIN_CFG, #{}).
-endif.

configure() ->
    Cfg = maps:merge(?BOARD_CFG, ?GAIN_CFG),
    case map_size(Cfg) of
        0 -> ok;
        _ -> case babytalk:audio_config(Cfg) of
                 ok -> ok;
                 Err -> line(["BTMIC ERROR audio_config ", io_lib:format("~p", [Err])])
             end
    end,
    maps:get(board, Cfg, default).

stream(Board, Seq) ->
    line(["BTMIC START ", integer_to_list(?RATE), " ", integer_to_list(?CHUNK_MS), " ",
          atom_to_list(Board)]),
    case babytalk:listen(?CHUNK_MS) of
        {ok, Ref} ->
            T0 = erlang:monotonic_time(millisecond),
            stream_loop(Board, Ref, Seq, T0);
        Err ->
            line(["BTMIC ERROR listen ", io_lib:format("~p", [Err])]),
            timer:sleep(1000),
            stream(Board, Seq)
    end.

stream_loop(Board, Ref, Seq, Tstat) ->
    receive
        {babytalk_mic, Ref, Pcm} when is_binary(Pcm) ->
            io:put_chars(["A ", integer_to_list(Seq), $\s, base64:encode(Pcm), $\n]),
            Now = erlang:monotonic_time(millisecond),
            case Now - Tstat >= 10000 of
                true ->
                    Heap = babytalk:heap_info(),
                    line(["BTMIC STAT ", integer_to_list(Seq), " ", integer_to_list(babytalk:rms(Pcm)),
                          " ", integer_to_list(proplists:get_value(psram_free, Heap, 0))]),
                    stream_loop(Board, Ref, Seq + 1, Now);
                false ->
                    stream_loop(Board, Ref, Seq + 1, Tstat)
            end;
        {babytalk_mic, Ref, {error, Code}} ->
            line(["BTMIC ERROR mic ", integer_to_list(Code)]),
            timer:sleep(500),
            stream(Board, Seq);
        {babytalk_mic, Ref, stopped} ->
            stream(Board, Seq)
    end.

line(Iodata) -> io:put_chars([Iodata, $\n]).
