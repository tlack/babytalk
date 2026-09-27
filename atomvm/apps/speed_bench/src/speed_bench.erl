%% BabyTalk speed on the board: transcription at several lengths (after a warm-up run), and
%% speech synthesis. No WiFi, no camera: nothing else on the memory bus.
-module(speed_bench).
-export([start/0]).

start() ->
    timer:sleep(500),
    stt(warmup, 2),
    [stt(bench, S) || S <- [1, 2, 2, 4, 8, 10]],
    [tts(T) || T <- [<<"Hello. This is the E S P 32 P 4, speaking with BabyTalk.">>,
                     <<"The quick brown fox jumps over the lazy dog, and then it takes a nap in the warm afternoon sun.">>]],
    io:format("BENCH done~n"),
    timer:sleep(infinity).

stt(Tag, Secs) ->
    T0 = erlang:monotonic_time(millisecond),
    R = babytalk:transcribe_sync(binary:copy(<<0, 0>>, 16000 * Secs), 60000),
    io:format("BENCH ~p stt ~p s: ~p ms (model ~p)~n", [Tag, Secs, erlang:monotonic_time(millisecond) - T0,
              case R of {ok, _, I} -> round(proplists:get_value(model_ms, I)); _ -> R end]).

tts(Text) ->
    T0 = erlang:monotonic_time(millisecond),
    case babytalk:say_sync(Text, 60000) of
        {ok, Pcm, Info} ->
            Ms = erlang:monotonic_time(millisecond) - T0,
            Audio = byte_size(Pcm) * 1000 div (2 * proplists:get_value(rate, Info)),
            io:format("BENCH tts: ~p ms for ~p ms of audio (x~.2f real time) ~p~n", [Ms, Audio, Ms / Audio, Info]);
        E -> io:format("BENCH tts failed ~p~n", [E])
    end.
