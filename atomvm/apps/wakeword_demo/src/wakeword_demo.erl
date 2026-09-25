%% Teach the board your own wake phrase, then talk to it. All on the ESP32-S3, in Erlang:
%%
%%   board: (chime) "Please carefully say your wakeword, and then I'll remember it."
%%   you:   "hey pumpkin"
%%   board: "Did you say hey pumpkin?"               (listens 2 s)
%%   you:   "yes"
%%   board: "Wakeword stored. Repeat it to talk to me in the future."
%%   ... later ...
%%   you:   "hey pumpkin"
%%   board: "I'm here, how can I help?"
%%   you:   "turn on the lights"
%%   board: "You said: turn on the lights"
%%
%%   wakeword_demo_sup (rest_for_one)
%%   |-- babytalk_listener   mic + speaker; starts with no wake phrase
%%   `-- wakeword_demo_app   the dialog: enrolling -> confirming -> running
-module(wakeword_demo).
-export([start/0]).

start() ->
    io:format("wakeword_demo: ~p~n", [babytalk:heap_info()]),
    {ok, _Sup} = wakeword_demo_sup:start_link(),
    timer:sleep(infinity).
