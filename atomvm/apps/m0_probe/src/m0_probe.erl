%% M0 probe: how much internal RAM and PSRAM are left with AtomVM running, before and after
%% WiFi comes up. WiFi credentials come from wifi_creds (generated, git-ignored).
-module(m0_probe).
-export([start/0]).

start() ->
    report(boot),
    {Ssid, Psk} = wifi_creds:get(),
    case network:wait_for_sta([{ssid, Ssid}, {psk, Psk}], 30000) of
        {ok, {Ip, _Mask, _Gw}} -> io:format("wifi up, ip ~p~n", [Ip]);
        Error -> io:format("wifi failed: ~p~n", [Error])
    end,
    report(wifi),
    loop(5).

loop(0) -> ok;
loop(N) ->
    timer:sleep(3000),
    report(idle),
    loop(N - 1).

report(Tag) ->
    io:format("M0 ~p ~p esp32_free_heap_size=~p~n",
              [Tag, babytalk:heap_info(), erlang:system_info(esp32_free_heap_size)]).
