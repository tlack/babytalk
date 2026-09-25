-module(wakeword_demo_sup).
-behaviour(supervisor).
-export([start_link/0, init/1]).

start_link() -> supervisor:start_link({local, ?MODULE}, ?MODULE, []).

%% The listener first: the app talks to it as soon as it starts. If the listener dies, the
%% app restarts too (rest_for_one) and enrollment begins again.
init([]) ->
    Listener = #{spellings => [], notify => wakeword_demo_app,
                 greeting => <<"I'm here, how can I help?">>},
    {ok, {#{strategy => rest_for_one, intensity => 5, period => 60},
          [#{id => listener, start => {babytalk_listener, start_link, [babytalk_listener, Listener]}},
           #{id => app, start => {wakeword_demo_app, start_link, []}}]}}.
