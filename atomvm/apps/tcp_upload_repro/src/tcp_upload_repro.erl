%% Repro: long uploads to a passive gen_tcp socket on AtomVM (ESP32-S3) occasionally end in
%% {error, closed} partway through. Stock AtomVM only (network + gen_tcp), no NIFs.
%%
%% Protocol: the client sends a 4-byte big-endian length and that many bytes; the server
%% replies "ok <Len>\n" and waits for the next upload on the same connection. On a recv
%% error it logs how far the upload got and closes. See README.md.
-module(tcp_upload_repro).
-export([start/0]).

-define(PORT, 5556).

start() ->
    {Ssid, Psk} = wifi_creds:get(),
    {ok, {Ip, _, _}} = network:wait_for_sta([{ssid, Ssid}, {psk, Psk}], 30000),
    {ok, L} = gen_tcp:listen(?PORT, [binary, {active, false}, {reuseaddr, true}]),
    io:format("tcp_upload_repro listening on ~p:~p, free heap ~p~n",
              [Ip, ?PORT, erlang:system_info(esp32_free_heap_size)]),
    accept(L, 1).

%% one process per connection
accept(L, Conn) ->
    {ok, S} = gen_tcp:accept(L),
    Pid = spawn(fun() -> receive go -> uploads(S, Conn, 1, <<>>) end end),
    ok = gen_tcp:controlling_process(S, Pid),
    Pid ! go,
    accept(L, Conn + 1).

uploads(S, Conn, N, Buf) ->
    case take(S, 4, Buf) of
        {ok, <<Len:32>>, Rest, _} ->
            T0 = erlang:monotonic_time(millisecond),
            case take(S, Len, Rest) of
                {ok, Data, Rest2, Chunks} ->
                    Ms = erlang:monotonic_time(millisecond) - T0,
                    io:format("conn ~p upload ~p: ~p bytes in ~p recv calls, ~p ms~n",
                              [Conn, N, byte_size(Data), Chunks, Ms]),
                    ok = gen_tcp:send(S, [<<"ok ">>, integer_to_list(Len), $\n]),
                    uploads(S, Conn, N + 1, Rest2);
                {error, Reason, Have, Chunks, SinceLastMs} ->
                    io:format("FAIL conn ~p upload ~p: ~p after ~p of ~p bytes (~p recv calls, "
                              "last data ~p ms before), free heap ~p~n",
                              [Conn, N, Reason, Have, Len, Chunks, SinceLastMs,
                               erlang:system_info(esp32_free_heap_size)]),
                    gen_tcp:close(S)
            end;
        {error, closed, 0, _, _} ->
            io:format("conn ~p: client closed after ~p uploads~n", [Conn, N - 1]),
            gen_tcp:close(S);
        {error, Reason, Have, _, _} ->
            io:format("FAIL conn ~p header of upload ~p: ~p after ~p bytes~n", [Conn, N, Reason, Have]),
            gen_tcp:close(S)
    end.

%% Exactly N bytes (recv(S, 0) returns whatever has arrived), plus leftovers.
%% Chunks are gathered in a list: appending to a binary copies on AtomVM.
take(S, N, Buf) ->
    take(S, N, [Buf], byte_size(Buf), 0, erlang:monotonic_time(millisecond)).

take(_S, N, Acc, Have, Chunks, _Last) when Have >= N ->
    <<Want:N/binary, Rest/binary>> = iolist_to_binary(lists:reverse(Acc)),
    {ok, Want, Rest, Chunks};
take(S, N, Acc, Have, Chunks, Last) ->
    case gen_tcp:recv(S, 0) of
        {ok, B} ->
            take(S, N, [B | Acc], Have + byte_size(B), Chunks + 1, erlang:monotonic_time(millisecond));
        {error, Reason} ->
            {error, Reason, Have, Chunks, erlang:monotonic_time(millisecond) - Last}
    end.
