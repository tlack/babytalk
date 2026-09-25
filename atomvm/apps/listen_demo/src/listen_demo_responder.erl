%% The listener's subscriber: prints what it hears and answers each command out loud,
%% through the listener (which owns the speaker as well as the mic).
-module(listen_demo_responder).
-behaviour(gen_server).
-export([start_link/0, init/1, handle_call/3, handle_cast/2, handle_info/2]).

start_link() -> gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

init([]) -> {ok, nil}.
handle_call(_R, _F, S) -> {reply, ok, S}.
handle_cast(_M, S) -> {noreply, S}.

handle_info({babytalk_listener, {heard, Text, Score}}, S) ->
    io:format("  heard ~p (score ~p)~n", [Text, Score]), {noreply, S};
handle_info({babytalk_listener, {wake, _Text, Score}}, S) ->
    io:format("AWAKE (score ~p) -- listening for a command~n", [Score]), {noreply, S};
handle_info({babytalk_listener, {command, Text}}, S) ->
    io:format("COMMAND ~p~n", [Text]),
    Reply = case Text of
                <<>> -> <<"Sorry, I didn't catch that.">>;
                _ -> [<<"You said: ">>, Text]
            end,
    babytalk_listener:say(babytalk_listener, Reply),
    {noreply, S};
handle_info({babytalk_listener, {said, Text}}, S) ->
    io:format("SAID ~p~n~n", [Text]), {noreply, S};
handle_info({babytalk_listener, Other}, S) ->
    io:format("  ~p~n", [Other]), {noreply, S}.
