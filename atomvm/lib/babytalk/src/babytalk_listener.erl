%% A wake-phrase listener as an OTP gen_server: owns the microphone stream, scores the last
%% few seconds against the wake phrase about once a second, and after a wake transcribes
%% the command that follows. Events go to a subscriber (pid or registered name):
%%
%%   {babytalk_listener, {heard, Text, Score}}     every scored window (Score: undefined
%%                                                 when no phrase is set)
%%   {babytalk_listener, {wake, Text, Score}}      the phrase was heard
%%   {babytalk_listener, {command, Text}}          what was said after it
%%
%% Put it under a supervisor: if the mic or the engine fails, it crashes and restarts clean.
%% Options (map): notify (required), spellings ([binary()] of the phrase, [] = transcribe
%% only), threshold (float, e.g. -21.0 from export/kws.py), window_ms (4000),
%% every_ms (1000), command_ms (4000: it starts when the wake is scored,
%% ~1 s after the phrase), chunk_ms (250).
-module(babytalk_listener).
-behaviour(gen_server).
-export([start_link/1, start_link/2, stop/1]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-record(st, {opts, mic, chunks = [], have = 0, % the window: newest chunk first, bytes
             scoring = none,                   % none | Ref of the transcription in flight
             last = 0,                         % ms of the last scoring start
             command = none}).                 % none | {Ref | collecting, [Chunk], Bytes}

start_link(Opts) -> gen_server:start_link(?MODULE, Opts, []).
start_link(Name, Opts) -> gen_server:start_link({local, Name}, ?MODULE, Opts, []).
stop(Server) -> gen_server:stop(Server).

init(Opts0) ->
    Opts = maps:merge(#{spellings => [], threshold => -21.0, window_ms => 4000,
                        every_ms => 1000, command_ms => 4000, chunk_ms => 250}, Opts0),
    {ok, Seqs} = babytalk:phrase(maps:get(spellings, Opts)),
    true = Seqs > 0 orelse maps:get(spellings, Opts) =:= [],
    {ok, Mic} = babytalk:listen(maps:get(chunk_ms, Opts)),
    {ok, #st{opts = Opts, mic = Mic, last = now_ms()}}.

handle_call(_Req, _From, St) -> {reply, {error, unknown_call}, St}.
handle_cast(_Msg, St) -> {noreply, St}.

%% microphone
handle_info({babytalk_mic, Mic, Pcm}, #st{mic = Mic} = St) when is_binary(Pcm) ->
    {noreply, maybe_score(command_audio(Pcm, window(Pcm, St)))};
handle_info({babytalk_mic, Mic, Other}, #st{mic = Mic}) ->
    exit({mic, Other});
%% scored window
handle_info({babytalk, Ref, Result}, #st{scoring = Ref} = St) ->
    {noreply, scored(Result, St#st{scoring = none})};
%% command transcript
handle_info({babytalk, Ref, Result}, #st{command = {Ref, _, _}} = St) ->
    case Result of
        {ok, Text, _} -> notify(St, {command, Text});
        {error, E} -> notify(St, {command_error, E})
    end,
    {noreply, St#st{command = none, chunks = [], have = 0, last = now_ms()}};
handle_info(_Stale, St) ->
    {noreply, St}.

terminate(_Reason, _St) ->
    babytalk:stop_listening().

%% ---------------------------------------------------------------------------- internals

%% keep the newest window_ms of audio
window(Pcm, #st{chunks = Cs, have = H, opts = #{window_ms := W}} = St) ->
    Max = W * 32,
    {Cs2, H2} = trim([Pcm | Cs], H + byte_size(Pcm), Max),
    St#st{chunks = Cs2, have = H2}.

trim(Cs, H, Max) when H =< Max -> {Cs, H};
trim(Cs, H, Max) ->
    [Oldest | RestRev] = lists:reverse(Cs),
    trim(lists:reverse(RestRev), H - byte_size(Oldest), Max).

%% after a wake: gather command_ms of audio, then transcribe it
command_audio(Pcm, #st{command = {collecting, Cs, B}, opts = #{command_ms := Ms}} = St) ->
    Cs2 = [Pcm | Cs], B2 = B + byte_size(Pcm),
    case B2 >= Ms * 32 of
        false -> St#st{command = {collecting, Cs2, B2}};
        true ->
            case babytalk:transcribe(iolist_to_binary(lists:reverse(Cs2))) of
                {ok, Ref} -> St#st{command = {Ref, [], 0}};
                {error, busy} -> St#st{command = {collecting, Cs2, B2}}  % retry next chunk
            end
    end;
command_audio(_Pcm, St) -> St.

maybe_score(#st{command = none, scoring = none, have = H, last = Last,
                opts = #{every_ms := E, window_ms := W}} = St) when H >= W * 32 div 2 ->
    Now = now_ms(),
    case Now - Last >= E andalso babytalk:transcribe(iolist_to_binary(lists:reverse(St#st.chunks))) of
        {ok, Ref} -> St#st{scoring = Ref, last = Now};
        _ -> St
    end;
maybe_score(St) -> St.

scored({ok, Text, Info}, #st{opts = #{threshold := Thr}} = St) ->
    Score = proplists:get_value(score, Info),
    notify(St, {heard, Text, Score}),
    case is_float(Score) andalso Score >= Thr of
        true ->
            notify(St, {wake, Text, Score}),
            St#st{command = {collecting, [], 0}, chunks = [], have = 0};
        false -> St
    end;
scored({error, E}, St) ->
    notify(St, {error, E}),
    St.

notify(#st{opts = #{notify := To}}, Event) -> To ! {babytalk_listener, Event}.

now_ms() -> erlang:monotonic_time(millisecond).
