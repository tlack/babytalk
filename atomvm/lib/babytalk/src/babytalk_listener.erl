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
%%   {babytalk_listener, {message_end, silence | max_length | fixed_length, #{ms, floor, peak, levels}}}
%%                                               the message stopped (levels: RMS)
%%   {babytalk_listener, {command, Text}}        the message after it
%%   {babytalk_listener, {answer, Text}}         the answer to ask/3
%%   {babytalk_listener, {said, Text}}           say/2 finished speaking Text
%% After a {command, _} the subscriber may answer with say/2 within reply_ms; then (or
%% right away if it doesn't) the ready chime plays and wake listening resumes.
%%
%% Dialogs: ask/3 speaks a prompt, chimes, records the answer and reports {answer, Text}; the
%% listener then waits (mic off) for the next call. wake_on/2 sets the wake phrase and starts
%% the wake loop. With no spellings at start, the listener chimes and waits: an app can
%% enroll a wake phrase first (apps/wakeword_demo).
%%
%% Put it under a supervisor: if the mic or the engine fails, it crashes and restarts clean.
%% Options (map): notify (required); spellings ([binary()], [] = no wake phrase);
%% threshold (-21.0, from export/kws.py); window_ms (4000) and every_ms (1000) for wake
%% scoring; silence_ms (700) of quiet that ends a message, max_message_ms (6000);
%% reply_ms (1500); chunk_ms (125); cues (true); cue_volume (70); voice_volume (76);
%% length_scale (1.10: speech 10% slower than the voice's own pace, easier to follow);
%% greeting (none: said after the wake chime, e.g. <<"I'm here, how can I help?">>);
%% vad (true: only score a window when something louder than the room was heard since the
%% last score -- a transcription takes 0.3-1.5 s of both cores, and scoring silence every
%% second can keep the chip busy for good on slower boards); vad_factor (2: "louder" = this
%% times the noise floor, and at least vad_min, 200 RMS).
%%
%% The listener traps exits, so when its owner goes (a supervisor shutting down, a crash) its
%% terminate/2 still runs and the mic is released; otherwise the stream would keep the audio
%% claimed and the next listener would get {error, busy} forever.
-module(babytalk_listener).
-behaviour(gen_server).
-export([start_link/1, start_link/2, stop/1, say/2, chime/2, ask/3, wake_on/2, sentences/1]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2, terminate/2]).

-define(CUES, #{ready => [{587, 90}, {0, 30}, {880, 140}],      % D5 -> A5
                wake => [{880, 70}, {0, 20}, {1175, 90}],       % A5 -> D6, brighter
                decoding => [{784, 80}, {0, 20}, {523, 120}]}). % G5 -> C5, falling

-record(st, {opts,
             phase = idle,       % wake | message | idle (mic off: decoding, speaking, waiting)
             mic = none,         % none | {on, Ref} | {stopping, Ref}
             out = [],           % queued output: {cue, Name} | {tones, Notes} | {say, Piece, Said}
                                 % (Said: the whole text, on its last piece, else none)
             out_ref = none,     % none | {say | play, Ref, Item}
             then = wake,        % the phase to enter when the queue is empty
             win = [], win_bytes = 0,              % wake: the newest window_ms, newest first
             sound = false,      % wake: something above the room was heard since the last score (vad)
             scoring = none, last = 0,             % wake: transcription in flight, its start
             msg = [], msg_bytes = 0, voiced = false, quiet = 0, peak = 0, levels = [],  % message being recorded
             floor = 300,        % noise floor (RMS), tracked while listening for the phrase
             decoding = none,    % none | Ref of the message transcription | {pending, Pcm}
             resume = 0,         % token of the pending resume timer
             msg_kind = command, % command (after a wake) | answer (to ask/3)
             msg_fixed = none,   % none | ms: record exactly this long (ask/3's #{ms => _})
             phrase = none,      % none | spellings to install before the next wake phase
             awaiting_reply = false}).  % a command was reported: a say/2 is its reply

start_link(Opts) -> gen_server:start_link(?MODULE, Opts, []).
start_link(Name, Opts) -> gen_server:start_link({local, Name}, ?MODULE, Opts, []).
stop(Server) -> gen_server:stop(Server).

%% Speak Text (iodata) through the speaker; listening pauses meanwhile. Long text is said a
%% sentence or so at a time (sentences/1): synthesis takes about as long as the speech, so a
%% long answer said whole would be silent (and deaf) for that long first.
say(Server, Text) -> gen_server:cast(Server, {say, iolist_to_binary(Text)}).

%% Play a chime ([{Hz, Ms}], as babytalk:tones/2) at the cue volume, in turn with speech;
%% listening pauses meanwhile.
chime(Server, Notes) -> gen_server:cast(Server, {chime, Notes}).

%% Say Prompt, chime, record the answer (until silence, or exactly #{ms => Ms}) and report
%% {answer, Text}; then wait.
ask(Server, Prompt, Opts) -> gen_server:cast(Server, {ask, iolist_to_binary(Prompt), Opts}).

%% Listen for a new wake phrase (spellings: binaries or strings); starts the wake loop.
wake_on(Server, Spellings) -> gen_server:cast(Server, {wake_on, Spellings}).

init(Opts0) ->
    Opts = maps:merge(#{spellings => [], threshold => -21.0, window_ms => 4000, every_ms => 1000,
                        silence_ms => 700, max_message_ms => 6000, reply_ms => 1500,
                        chunk_ms => 125, cues => true, cue_volume => 70, voice_volume => 76,
                        length_scale => 1.10, greeting => none,
                        vad => true, vad_factor => 2, vad_min => 200}, Opts0),
    process_flag(trap_exit, true),
    St = #st{opts = Opts},
    case maps:get(spellings, Opts) of
        [] -> {ok, output([cue(ready)], idle, St)};              % wait for ask/3 or wake_on/2
        Spellings -> {ok, output([cue(ready)], wake, St#st{phrase = Spellings})}
    end.

handle_call(_Req, _From, St) -> {reply, {error, unknown_call}, St}.

%% a reply to a command: speak it, then the ready chime
handle_cast({say, Text}, #st{awaiting_reply = true} = St) ->
    {noreply, output(says(Text) ++ [cue(ready)], wake, St#st{awaiting_reply = false})};
handle_cast({say, Text}, #st{phase = idle} = St) ->
    {noreply, output(says(Text), St#st.then, St)};
handle_cast({say, Text}, #st{phase = Phase} = St) ->
    {noreply, output(says(Text), Phase, St)};
handle_cast({chime, Notes}, #st{phase = idle} = St) ->
    {noreply, output([{tones, Notes}], St#st.then, St)};
handle_cast({chime, Notes}, #st{phase = Phase} = St) ->
    {noreply, output([{tones, Notes}], Phase, St)};
handle_cast({ask, Prompt, Opts}, St) ->
    Then = {message, answer, maps:get(ms, Opts, none)},
    {noreply, output(says(Prompt) ++ [cue(wake)], Then, St#st{awaiting_reply = false})};
handle_cast({wake_on, Spellings}, St) ->
    {noreply, output([cue(ready)], wake, St#st{phrase = Spellings, awaiting_reply = false})};
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
handle_info({babytalk, Ref, Result}, #st{decoding = Ref, msg_kind = Kind} = St) ->
    Text = case Result of {ok, T, _} -> T; {error, _} -> <<>> end,
    notify(St, {Kind, Text}),
    case Kind of
        answer -> {noreply, St#st{decoding = none}};           % the app decides what's next
        command ->
            Token = St#st.resume + 1,
            erlang:send_after(maps:get(reply_ms, St#st.opts), self(), {resume, Token}),
            {noreply, St#st{decoding = none, resume = Token, awaiting_reply = true}}
    end;
handle_info(retry_decode, #st{decoding = {pending, Pcm}} = St) ->
    {noreply, start_decoding(Pcm, St)};
%% ---- output
handle_info({babytalk, Ref, Result}, #st{out_ref = {say, Ref, {say, _, _} = Item}} = St) ->
    case Result of
        {ok, Pcm, Info} ->
            {ok, P} = babytalk:play(Pcm, proplists:get_value(rate, Info), maps:get(voice_volume, St#st.opts)),
            {noreply, St#st{out_ref = {play, P, Item}}};
        {error, E} ->
            notify(St, {say_error, E}),
            {noreply, pump(St#st{out_ref = none})}
    end;
handle_info({babytalk_play, Ref, Result}, #st{out_ref = {play, Ref, Item}} = St) ->
    case {Item, Result} of
        {{say, _, none}, done} -> ok;                      % more of it to come
        {{say, _, Text}, done} -> notify(St, {said, Text});
        {_, done} -> ok;
        {_, E} -> notify(St, {play_error, E})
    end,
    {noreply, pump(St#st{out_ref = none})};
handle_info(retry_output, St) ->
    {noreply, pump(St)};
%% no reply came after a command: chime and listen again
handle_info({resume, Token}, #st{resume = Token, awaiting_reply = true, out = [], out_ref = none} = St) ->
    {noreply, output([cue(ready)], wake, St#st{awaiting_reply = false})};
handle_info(_Stale, St) ->
    {noreply, St}.

terminate(_Reason, _St) ->
    babytalk:stop_listening().

%% ---------------------------------------------------------------------------- output queue

cue(Name) -> {cue, Name}.

says(Text) ->
    case sentences(Text) of
        [] -> [];
        Ps -> [{say, P, none} || P <- lists:droplast(Ps)] ++ [{say, lists:last(Ps), Text}]
    end.

%% Text in pieces of at most ?PIECE bytes, cut after a sentence's end once a piece has some
%% length (short sentences travel together), else at a space
-define(PIECE, 160).
sentences(Text) ->
    pack(binary:split(Text, [<<" ">>, <<"\n">>], [global, trim_all]), <<>>, []).

pack([], <<>>, Acc) -> lists:reverse(Acc);
pack([], Cur, Acc) -> lists:reverse([Cur | Acc]);
pack([W | Ws], Cur, Acc) ->
    Next = case Cur of <<>> -> W; _ -> <<Cur/binary, " ", W/binary>> end,
    End = binary:last(W),
    if
        byte_size(Next) > ?PIECE, Cur =/= <<>> -> pack([W | Ws], <<>>, [Cur | Acc]);   % full: cut before W
        (End =:= $. orelse End =:= $? orelse End =:= $!), byte_size(Next) >= 60 -> pack(Ws, <<>>, [Next | Acc]);
        true -> pack(Ws, Next, Acc)
    end.

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
                  {tones, Notes} -> {play, babytalk:tones(Notes, maps:get(cue_volume, Opts))};
                  {say, Text, _} -> {say, babytalk:say(Text, #{length_scale => maps:get(length_scale, Opts)})}
              end,
    case Started of
        {Kind, {ok, Ref}} -> St#st{out = Rest, out_ref = {Kind, Ref, Item}};
        {_, {error, busy}} ->                   % a transcription is finishing
            erlang:send_after(50, self(), retry_output),
            St
    end.

enter(idle, St) ->
    St#st{phase = idle};
enter(wake, #st{phrase = Spellings} = St) when Spellings =/= none ->
    case babytalk:phrase(Spellings) of             % install the new wake phrase first
        {ok, _} -> enter(wake, St#st{phrase = none});
        {error, busy} -> erlang:send_after(50, self(), retry_output), St
    end;
enter({message, Kind, Fixed}, St) ->
    listen(message, St#st{msg_kind = Kind, msg_fixed = Fixed});
enter(wake, St) ->
    listen(wake, St).

listen(Phase, #st{opts = #{chunk_ms := C}} = St) ->
    case babytalk:listen(C) of
        {ok, Ref} ->
            St#st{phase = Phase, mic = {on, Ref}, win = [], win_bytes = 0, last = now_ms(),
                  msg = [], msg_bytes = 0, voiced = false, quiet = 0, peak = 0, levels = []};
        {error, busy} ->
            erlang:send_after(50, self(), retry_output),
            St
    end.

%% ---------------------------------------------------------------------------- listening

heard_chunk(Pcm, #st{phase = wake, floor = F, opts = O} = St) ->
    R = babytalk:rms(Pcm),
    Sound = St#st.sound orelse R > max(maps:get(vad_factor, O) * F, maps:get(vad_min, O)),
    maybe_score(track_floor(R, window(Pcm, St#st{sound = Sound})));
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

%% score the window about every every_ms, once it holds at least half of window_ms (and,
%% with vad, only if something was heard since the last score)
maybe_score(#st{sound = false, opts = #{vad := true}} = St) -> St;
maybe_score(#st{scoring = none, win_bytes = B, last = Last,
                opts = #{every_ms := E, window_ms := W}} = St) when B >= W * 16 ->
    Now = now_ms(),
    case Now - Last >= E andalso babytalk:transcribe(iolist_to_binary(lists:reverse(St#st.win))) of
        {ok, Ref} -> St#st{scoring = Ref, last = Now, sound = false};
        _ -> St
    end;
maybe_score(St) -> St.

scored({ok, Text, Info}, #st{phase = wake, opts = #{threshold := Thr}} = St) ->
    Score = proplists:get_value(score, Info),
    notify(St, {heard, Text, Score}),
    case is_float(Score) andalso Score >= Thr of
        true ->
            notify(St, {wake, Text, Score}),
            Greeting = case maps:get(greeting, St#st.opts) of none -> []; G -> says(iolist_to_binary(G)) end,
            output([cue(wake) | Greeting], {message, command, none}, St);
        false -> St
    end;
scored({ok, _, _}, St) -> St;                  % arrived after the phase ended
scored({error, E}, St) -> notify(St, {error, E}), St.

%% record until silence_ms of quiet after some speech, or max_message_ms (or exactly msg_fixed)
message_chunk(Pcm, #st{msg = M, msg_bytes = B, voiced = V, quiet = Q, msg_fixed = Fixed,
                       opts = #{chunk_ms := C, silence_ms := S, max_message_ms := Max}} = St0) ->
    R = babytalk:rms(Pcm),
    St = case V of false -> track_floor(R, St0); true -> St0 end,  % the quiet before speech
    F = St#st.floor,
    Loud = R > max(3 * F, 250),
    St2 = St#st{msg = [Pcm | M], msg_bytes = B + byte_size(Pcm), voiced = V orelse Loud,
                quiet = case Loud of true -> 0; false -> Q + C end, peak = max(R, St#st.peak), levels = [R | St#st.levels]},
    End = if Fixed =/= none -> (St2#st.msg_bytes >= Fixed * 32) andalso fixed_length;
             St2#st.voiced andalso St2#st.quiet >= S -> silence;
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
