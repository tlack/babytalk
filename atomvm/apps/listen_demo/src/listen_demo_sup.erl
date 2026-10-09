-module(listen_demo_sup).
-behaviour(supervisor).
-export([start_link/0, init/1]).

%% Wake phrase spellings and threshold from tools/wake_phrases/wake_up_tomato_face.json
%% (export/kws.py: 98% detection, 0 false wakes on 1189 negatives at -21).
-define(SPELLINGS, [<<"wake up tomato face">>, <<"wake up to mato face">>, <<"wake up to meato face">>,
                    <<"wake up to meatto face">>, <<"wake up to meadow face">>, <<"wake up t martr face">>,
                    <<"wake up to marto face from">>, <<"wake up to matle face">>, <<"wake up to midto face">>,
                    <<"wake up tomato faceo">>, <<"wake up tomato face a">>, <<"wake up tomato face f">>]).

%% BABYTALK_BOARD at build time (mix.exs) picks the audio board; else the firmware's default
-ifdef(BOARD).
-define(AUDIO, #{audio => ?BOARD}).
-else.
-define(AUDIO, #{}).
-endif.

start_link() -> supervisor:start_link({local, ?MODULE}, ?MODULE, []).

init([]) ->
    Listener = maps:merge(#{spellings => ?SPELLINGS, threshold => -21.0, notify => listen_demo_responder},
                          ?AUDIO),
    {ok, {#{strategy => rest_for_one, intensity => 5, period => 60},
          [#{id => responder, start => {listen_demo_responder, start_link, []}},
           #{id => listener, start => {babytalk_listener, start_link, [babytalk_listener, Listener]}}]}}.
