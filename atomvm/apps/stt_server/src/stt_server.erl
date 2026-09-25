%% Speech to text over TCP, for testing babytalk from the laptop (atomvm/tools/stt_client.py):
%% the client sends a 4-byte big-endian length and that many bytes of 16 kHz mono int16 PCM;
%% the server replies with one tab-separated line: ok, Text, WallMs, Info (~w) -- or error, Reason.
%% WiFi credentials come from wifi_creds (generated, git-ignored).
-module(stt_server).
-export([start/0]).

-define(PORT, 5555).

start() ->
    io:format("babytalk: ~p~n", [babytalk:heap_info()]),
    {Ssid, Psk} = wifi_creds:get(),
    {ok, {Ip, _, _}} = network:wait_for_sta([{ssid, Ssid}, {psk, Psk}], 30000),
    {ok, L} = gen_tcp:listen(?PORT, [binary, {active, false}, {reuseaddr, true}]),
    io:format("stt_server listening on ~p:~p ~p~n", [Ip, ?PORT, babytalk:heap_info()]),
    accept(L).

%% one process per connection: a client that vanishes can't block the next one
accept(L) ->
    {ok, S} = gen_tcp:accept(L),
    Pid = spawn(fun() -> receive go -> serve(S, <<>>) end end),
    ok = gen_tcp:controlling_process(S, Pid),
    Pid ! go,
    accept(L).

%% Buf: bytes received but not consumed yet (recv returns whatever has arrived)
serve(S, Buf) ->
    case take(S, 4, Buf) of
        {ok, <<Len:32>>, Rest} -> clip(S, take(S, Len, Rest));
        _Closed -> gen_tcp:close(S)
    end.

%% (AtomVM's gen_tcp occasionally reports {error, closed} in the middle of a long upload --
%% about 1 in 35 clips of 100-330 KB here; the client then reconnects and resends)
clip(S, {ok, Pcm, Rest}) ->
    T0 = erlang:monotonic_time(millisecond),
    Result = babytalk:transcribe_sync(Pcm, 60000),
    Ms = erlang:monotonic_time(millisecond) - T0,
    io:format("~p in ~p ms, ~p~n", [Result, Ms, babytalk:heap_info()]),
    Line = case Result of
               {ok, Text, Info} -> [<<"ok\t">>, Text, $\t, integer_to_list(Ms), $\t, io_lib:format("~w", [Info])];
               Error -> [<<"error\t">>, io_lib:format("~w", [Error])]
           end,
    ok = gen_tcp:send(S, [Line, $\n]),
    serve(S, Rest);
clip(S, Error) ->
    io:format("upload failed: ~p~n", [Error]),
    gen_tcp:close(S).

%% the next N bytes, and what's left over (chunks gathered in a list: appending to a
%% binary copies on AtomVM)
take(S, N, Buf) -> take(S, N, [Buf], byte_size(Buf)).

take(_S, N, Acc, Have) when Have >= N ->
    <<Want:N/binary, Rest/binary>> = iolist_to_binary(lists:reverse(Acc)),
    {ok, Want, Rest};
take(S, N, Acc, Have) ->
    case gen_tcp:recv(S, 0) of
        {ok, B} -> take(S, N, [B | Acc], Have + byte_size(B));
        Error -> Error
    end.
