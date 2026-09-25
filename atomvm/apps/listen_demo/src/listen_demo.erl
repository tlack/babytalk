%% Say "wake up tomato face", then a command: the board prints what it heard. All on the
%% ESP32-S3, in Erlang on AtomVM: the microphone streams while transcriptions run.
%%
%%   listen_demo_sup (rest_for_one)
%%   |-- listen_demo_printer   prints the listener's events
%%   `-- babytalk_listener     mic -> wake phrase -> command transcript
-module(listen_demo).
-export([start/0]).

start() ->
    io:format("listen_demo: ~p~n", [babytalk:heap_info()]),
    {ok, _Sup} = listen_demo_sup:start_link(),
    timer:sleep(infinity).
