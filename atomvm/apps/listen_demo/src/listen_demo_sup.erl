-module(listen_demo_sup).
-behaviour(supervisor).
-export([start_link/0, init/1]).

%% Wake phrase spellings and threshold from tools/wake_phrases/wake_up_tomato_face.json
%% (export/kws.py: 98% detection, 0 false wakes on 1189 negatives at -21).
-define(SPELLINGS, [<<"wake up tomato face">>, <<"wake up to mato face">>, <<"wake up to meato face">>,
                    <<"wake up to meatto face">>, <<"wake up to meadow face">>, <<"wake up t martr face">>,
                    <<"wake up to marto face from">>, <<"wake up to matle face">>, <<"wake up to midto face">>,
                    <<"wake up tomato faceo">>, <<"wake up tomato face a">>, <<"wake up tomato face f">>]).

start_link() -> supervisor:start_link({local, ?MODULE}, ?MODULE, []).

init([]) ->
    Listener = #{spellings => ?SPELLINGS, threshold => -21.0, notify => listen_demo_printer},
    {ok, {#{strategy => rest_for_one, intensity => 5, period => 60},
          [#{id => printer, start => {listen_demo_printer, start_link, []}},
           #{id => listener, start => {babytalk_listener, start_link, [babytalk_listener, Listener]}}]}}.
