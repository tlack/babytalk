%% Text to speech on the board, from Erlang: synthesize a few sentences, report speed
%% (RTF = synthesis time / audio time) and a SHA-256 of the PCM (to compare with the
%% MicroPython firmware), and say them out loud.
-module(tts_demo).
-export([start/0]).

-define(TEXTS, [<<"Hello from Erlang on the ESP32.">>,
                <<"The quick brown fox jumps over the lazy dog.">>,
                <<"I can understand what you say, and talk back, with no cloud at all.">>]).

start() ->
    io:format("tts_demo: ~p~n", [babytalk:heap_info()]),
    loop(1).

%% a few rounds, with pauses: time to film it
loop(0) -> io:format("tts_demo done ~p~n", [babytalk:heap_info()]), timer:sleep(infinity);
loop(N) -> lists:foreach(fun say/1, ?TEXTS), timer:sleep(3000), loop(N - 1).

say(Text) ->
    T0 = erlang:monotonic_time(millisecond),
    {ok, Pcm, Info} = babytalk:say_sync(Text, #{length_scale => 1.10}, 30000),
    Ms = erlang:monotonic_time(millisecond) - T0,
    Secs = byte_size(Pcm) / 2 / proplists:get_value(rate, Info),
    io:format("~s~n  ~.2f s of speech in ~p ms (RTF ~.2f) ~p~n  sha256 ~s~n",
              [Text, Secs, Ms, Ms / 1000 / Secs, Info, hex(crypto:hash(sha256, Pcm))]),
    {ok, Ref} = babytalk:play(Pcm, proplists:get_value(rate, Info)),
    receive {babytalk_play, Ref, R} -> io:format("  played: ~p~n", [R]) end,
    timer:sleep(1500).

hex(Bin) -> [io_lib:format("~2.16.0b", [B]) || <<B>> <= binary:part(Bin, 0, 8)].
