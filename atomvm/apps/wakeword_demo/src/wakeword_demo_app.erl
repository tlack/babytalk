%% The dialog, driven by the listener's events.
%%   enrolling:  asked for the wake phrase; the answer is a candidate
%%   confirming: asked "Did you say ...?" (2 s answer); yes -> store it, else ask again
%%   running:    the wake loop; each command is repeated back
%% The phrase is stored as the model's own spelling of how you said it ("hey pumkin"):
%% the wake scorer compares what it hears against exactly that reading.
-module(wakeword_demo_app).
-behaviour(gen_server).
-export([start_link/0, init/1, handle_call/3, handle_cast/2, handle_info/2]).

-define(L, babytalk_listener).

start_link() -> gen_server:start_link({local, ?MODULE}, ?MODULE, [], []).

init([]) ->
    babytalk_listener:ask(?L, <<"Please carefully say your wakeword, and then I'll remember it.">>, #{}),
    {ok, enrolling}.

handle_call(_R, _F, S) -> {reply, ok, S}.
handle_cast(_M, S) -> {noreply, S}.

handle_info({babytalk_listener, {answer, Text}}, enrolling) ->
    io:format("candidate wake phrase: ~p~n", [Text]),
    case Text of
        <<>> ->
            babytalk_listener:ask(?L, <<"Sorry, I didn't catch that. Please say your wakeword.">>, #{}),
            {noreply, enrolling};
        _ ->
            babytalk_listener:ask(?L, [<<"Did you say ">>, Text, <<"?">>], #{ms => 2000}),
            {noreply, {confirming, Text}}
    end;
handle_info({babytalk_listener, {answer, Text}}, {confirming, Phrase}) ->
    io:format("confirmation: ~p~n", [Text]),
    case is_yes(Text) of
        true ->
            babytalk_listener:say(?L, <<"Wakeword stored. Repeat it to talk to me in the future.">>),
            babytalk_listener:wake_on(?L, [Phrase]),
            io:format("WAKE PHRASE ~p~n", [Phrase]),
            {noreply, {running, Phrase}};
        false ->
            babytalk_listener:ask(?L, <<"Okay, let's try again. Please say your wakeword.">>, #{}),
            {noreply, enrolling}
    end;
handle_info({babytalk_listener, {wake, _Text, Score}}, S) ->
    io:format("AWAKE (score ~p)~n", [Score]), {noreply, S};
handle_info({babytalk_listener, {command, Text}}, {running, _} = S) ->
    io:format("COMMAND ~p~n", [Text]),
    babytalk_listener:say(?L, case Text of
                                  <<>> -> <<"Sorry, I didn't catch that.">>;
                                  _ -> [<<"You said: ">>, Text]
                              end),
    {noreply, S};
handle_info({babytalk_listener, {heard, _, _}}, S) ->
    {noreply, S};                                   % every wake window: too chatty to print
handle_info({babytalk_listener, Event}, S) ->
    io:format("  ~p~n", [Event]), {noreply, S}.

%% "yes", "yeah", "yep", "yup", "sure", "correct", "right" -- or anything starting "ye"
is_yes(Text) ->
    Words = binary:split(Text, <<" ">>, [global]),
    lists:any(fun(W) ->
                  lists:member(W, [<<"yes">>, <<"yeah">>, <<"yep">>, <<"yup">>, <<"ya">>, <<"yea">>,
                                   <<"sure">>, <<"correct">>, <<"right">>])
                  orelse (byte_size(W) >= 2 andalso binary:part(W, 0, 2) =:= <<"ye">>)
              end, Words).
