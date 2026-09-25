%% A voice front end as an OTP gen_server: owns the microphone and the speaker (they share
%% one I2S port on the board), listens for a wake phrase, records the message after it until
%% the speaker stops talking, transcribes it, and speaks replies -- with short chimes so a
%% person can follow along:
%%
%%   ready chime     listening for the wake phrase (at start, and after each exchange)
%%   wake chime      heard it: listening for the message
%%   decoding chime  the message ended: transcribing it
%%
%% Events go to a subscriber (pid or registered name):
%%   {babytalk_listener, {heard, Text, Score}}   each scored wake window (Score: undefined
%%                                               when no phrase is set)
%%   {babytalk_listener, {wake, Text, Score}}    the phrase was heard
%%   {babytalk_listener, {message_end, silence | max_length, #{ms, floor, peak}}}
%%                                               the message stopped (levels: RMS)
%%   {babytalk_listener, {command, Text}}        the message after it
%%   {babytalk_listener, {said, Text}}           say/2 finished speaking Text
%% After a {command, _} the subscriber may answer with say/2 within reply_ms; then (or
%% right away if it doesn't) the ready chime plays and wake listening resumes.
%%
%% Put it under a supervisor: if the mic or the engine fails, it crashes and restarts clean.
%% Options (map): notify (required); spellings ([binary()], [] = no wake phrase);
%% threshold (-21.0, from export/kws.py); window_ms (4000) and every_ms (1000) for wake
%% scoring; silence_ms (700) of quiet that ends a message, max_message_ms (6000);
%% reply_ms (1500); chunk_ms (125); cues (true); cue_volume (70).
-module(babytalk_listener).
-behaviour(gen_server).
-export([start_link/1, start_link/2, stop/1, say/2]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(CUES, #{ready => [{587, 90}, {0, 30}, {880, 140}],      % D5 -> A5
                wake => [{880, 70}, {0, 20}, {1175, 90}],       % A5 -> D6, brighter
                decoding => [{784, 80}, {0, 20}, {523, 120}]}). % G5 -> C5, falling

-record(st, {opts,
             phase = idle,       % wake | message | idle (mic off: decoding, speaking, waiting)
             mic = none,         % none | {on, Ref} | {stopping, Ref}
             out = [],           % queued output: {cue, Name} | {say, Text}
             out_ref = none,     % none | {say | play, Ref, Item}
             then = wake,        % the phase to enter when the queue is empty
             win = [], win_bytes = 0,              % wake: the newest window_ms, newest first
             scoring = none, last = 0,             % wake: transcription in flight, its start
             msg = [], msg_bytes = 0, voiced = false, quiet = 0, peak = 0, levels = [],  % message being recorded
             floor = 300,        % noise floor (RMS), tracked while listening for the phrase
             decoding = none,    % none | Ref of the message transcription | {pending, Pcm}
             resume = 0}).       % token of the pending resume timer

start_link(Opts) -> gen_server:start_link(?MODULE, Opts, []).
start_link(Name, Opts) -> gen_server:start_link({local, Name}, ?MODULE, Opts, []).
stop(Server) -> gen_server:stop(Server).

%% Speak Text (iodata) through the speaker; listening pauses meanwhile.
say(Server, Text) -> gen_server:cast(Server, {say, iolist_to_binary(Text)}).

init(Opts0) ->
    Opts = maps:merge(#{spellings => [], threshold => -21.0, window_ms => 4000, every_ms => 1000,
                        silence_ms => 700, max_message_ms => 6000, reply_ms => 1500,
                        chunk_ms => 125, cues => true, cue_volume => 70}, Opts0),
    {ok, Seqs} = babytalk:phrase(maps:get(spellings, Opts)),
    true = Seqs > 0 orelse maps:get(spellings, Opts) =:= [],
    {ok, output([cue(ready)], wake, #st{opts = Opts})}.

handle_call(_Req, _From, St) -> {reply, {error, unknown_call}, St}.

%% a reply to a message (we are waiting after it): speak it, then the ready chime
handle_cast({say, Text}, #st{phase = idle, then = idle} = St) ->
    {noreply, output([{say, Text}, cue(ready)], wake, St#st{resume = St#st.resume + 1})};
handle_cast({say, Text}, #st{phase = idle} = St) ->
    {noreply, output([{say, Text}], St#st.then, St)};
handle_cast({say, Text}, #st{phase = Phase} = St) ->
    {noreply, output([{say, Text}], Phase, St)};
handle_cast(_Msg, St) -> {noreply, St}.

%% ---- microphone
handle_info({babytalk_mic, Ref, Pcm}, #st{mic = {on, Ref}} = St) when is_binary(Pcm) ->
    {noreply, heard_chunk(Pcm, St)};
handle_info({babytalk_mic, Ref, Pcm}, #st{mic = {stopping, Ref}} = St) when is_binary(Pcm) ->
    {noreply, St};
handle_info({babytalk_mic, Ref, stopped}, #st{mic = {stopping, Ref}} = St) ->
    {noreply, pump(St#st{mic = none})};
handle_info({babytalk_mic, _Ref, {error, E}}, _St) ->
    exit({mic, E});
%% ---- transcriptions
handle_info({babytalk, Ref, Result}, #st{scoring = Ref} = St) ->
    {noreply, scored(Result, St#st{scoring = none})};
handle_info({babytalk, Ref, Result}, #st{decoding = Ref} = St) ->
    Text = case Result of {ok, T, _} -> T; {error, _} -> <<>> end,
    notify(St, {command, Text}),
    Token = St#st.resume + 1,
    erlang:send_after(maps:get(reply_ms, St#st.opts), self(), {resume, Token}),
    {noreply, St#st{decoding = none, resume = Token}};
handle_info(retry_decode, #st{decoding = {pending, Pcm}} = St) ->
    {noreply, start_decoding(Pcm, St)};
%% ---- output
handle_info({babytalk, Ref, Result}, #st{out_ref = {say, Ref, {say, Text}}} = St) ->
    case Result of
        {ok, Pcm, Info} ->
            {ok, P} = babytalk:play(Pcm, proplists:get_value(rate, Info)),
            {noreply, St#st{out_ref = {play, P, {say, Text}}}};
        {error, E} ->
            notify(St, {say_error, E}),
            {noreply, pump(St#st{out_ref = none})}
    end;
handle_info({babytalk_play, Ref, Result}, #st{out_ref = {play, Ref, Item}} = St) ->
    case {Item, Result} of
        {{say, Text}, done} -> notify(St, {said, Text});
        {_, done} -> ok;
        {_, E} -> notify(St, {play_error, E})
    end,
    {noreply, pump(St#st{out_ref = none})};
handle_info(retry_output, St) ->
    {noreply, pump(St)};
%% no reply came after a message: chime and listen again
handle_info({resume, Token}, #st{resume = Token, phase = idle, out = [], out_ref = none} = St) ->
    {noreply, output([cue(ready)], wake, St)};
handle_info(_Stale, St) ->
    {noreply, St}.

terminate(_Reason, _St) ->
    babytalk:stop_listening().

%% ---------------------------------------------------------------------------- output queue

cue(Name) -> {cue, Name}.

%% Queue Items, then enter phase Then; the mic pauses while anything plays.
output(Items, Then, #st{opts = #{cues := false}} = St) ->
    output1([I || I <- Items, element(1, I) =/= cue], Then, St);
output(Items, Then, St) ->
    output1(Items, Then, St).

output1(Items, Then, #st{out = Out} = St) ->
    St2 = St#st{out = Out ++ Items, then = Then, phase = idle},
    case St2#st.mic of
        {on, Ref} -> ok = babytalk:stop_listening(), St2#st{mic = {stopping, Ref}};
        {stopping, _} -> St2;
        none -> pump(St2)
    end.

%% start the next queued item, or enter the next phase (waits for the mic to stop)
pump(#st{out_ref = R} = St) when R =/= none -> St;
pump(#st{mic = M} = St) when M =/= none -> St;
pump(#st{out = [], then = Then} = St) -> enter(Then, St);
pump(#st{out = [Item | Rest], opts = Opts} = St) ->
    Started = case Item of
                  {cue, Name} -> {play, babytalk:tones(maps:get(Name, ?CUES), maps:get(cue_volume, Opts))};
                  {say, Text} -> {say, babytalk:say(Text)}
              end,
    case Started of
        {Kind, {ok, Ref}} -> St#st{out = Rest, out_ref = {Kind, Ref, Item}};
        {_, {error, busy}} ->                   % a transcription is finishing
            erlang:send_after(50, self(), retry_output),
            St
    end.

enter(idle, St) ->
    St#st{phase = idle};
enter(Phase, #st{opts = #{chunk_ms := C}} = St) ->
    case babytalk:listen(C) of
        {ok, Ref} ->
            St#st{phase = Phase, mic = {on, Ref}, win = [], win_bytes = 0, last = now_ms(),
                  msg = [], msg_bytes = 0, voiced = false, quiet = 0, peak = 0, levels = []};
        {error, busy} ->
            erlang:send_after(50, self(), retry_output),
            St
    end.

%% ---------------------------------------------------------------------------- listening

heard_chunk(Pcm, #st{phase = wake} = St) ->
    maybe_score(track_floor(babytalk:rms(Pcm), window(Pcm, St)));
heard_chunk(Pcm, #st{phase = message} = St) ->
    message_chunk(Pcm, St);
heard_chunk(_Pcm, St) ->
    St.

%% noise floor: follows quiet levels at once, louder ones slowly
track_floor(R, #st{floor = F} = St) when R < F -> St#st{floor = max(50, R)};
track_floor(R, #st{floor = F} = St) -> St#st{floor = F + (R - F) div 50}.

%% keep the newest window_ms of audio
window(Pcm, #st{win = W, win_bytes = B, opts = #{window_ms := Ms}} = St) ->
    {W2, B2} = trim([Pcm | W], B + byte_size(Pcm), Ms * 32),
    St#st{win = W2, win_bytes = B2}.

trim(Cs, B, Max) when B =< Max -> {Cs, B};
trim(Cs, B, Max) ->
    [Oldest | RestRev] = lists:reverse(Cs),
    trim(lists:reverse(RestRev), B - byte_size(Oldest), Max).

%% score the window about every every_ms, once it holds at least half of window_ms
maybe_score(#st{scoring = none, win_bytes = B, last = Last,
                opts = #{every_ms := E, window_ms := W}} = St) when B >= W * 16 ->
    Now = now_ms(),
    case Now - Last >= E andalso babytalk:transcribe(iolist_to_binary(lists:reverse(St#st.win))) of
        {ok, Ref} -> St#st{scoring = Ref, last = Now};
        _ -> St
    end;
maybe_score(St) -> St.

scored({ok, Text, Info}, #st{phase = wake, opts = #{threshold := Thr}} = St) ->
    Score = proplists:get_value(score, Info),
    notify(St, {heard, Text, Score}),
    case is_float(Score) andalso Score >= Thr of
        true ->
            notify(St, {wake, Text, Score}),
            output([cue(wake)], message, St);
        false -> St
    end;
scored({ok, _, _}, St) -> St;                  % arrived after the phase ended
scored({error, E}, St) -> notify(St, {error, E}), St.

%% record until silence_ms of quiet after some speech, or max_message_ms
message_chunk(Pcm, #st{msg = M, msg_bytes = B, voiced = V, quiet = Q, floor = F,
                       opts = #{chunk_ms := C, silence_ms := S, max_message_ms := Max}} = St) ->
    R = babytalk:rms(Pcm),
    Loud = R > max(3 * F, 250),
    St2 = St#st{msg = [Pcm | M], msg_bytes = B + byte_size(Pcm), voiced = V orelse Loud,
                quiet = case Loud of true -> 0; false -> Q + C end, peak = max(R, St#st.peak), levels = [R | St#st.levels]},
    End = if St2#st.voiced andalso St2#st.quiet >= S -> silence;
             St2#st.msg_bytes >= Max * 32 -> max_length;
             true -> false
          end,
    case End of
        false -> St2;
        _ ->
            notify(St2, {message_end, End, #{ms => St2#st.msg_bytes div 32, floor => F, peak => St2#st.peak,
                                                        levels => lists:reverse(St2#st.levels)}}),
            Pcm2 = iolist_to_binary(lists:reverse(St2#st.msg)),
            start_decoding(Pcm2, output([cue(decoding)], idle, St2#st{msg = [], msg_bytes = 0}))
    end.

start_decoding(Pcm, St) ->
    case babytalk:transcribe(Pcm) of
        {ok, Ref} -> St#st{decoding = Ref};
        {error, busy} ->                       % a wake scoring is finishing
            erlang:send_after(50, self(), retry_decode),
            St#st{decoding = {pending, Pcm}}
    end.

notify(#st{opts = #{notify := To}}, Event) -> To ! {babytalk_listener, Event}.

now_ms() -> erlang:monotonic_time(millisecond).
