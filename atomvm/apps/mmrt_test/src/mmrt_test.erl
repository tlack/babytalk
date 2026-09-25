%% On-device check of the mmrt NIFs: every case in mmrt_vectors (generated on the host by
%% gen_vectors.py) must match bit for bit; then throughput, printed as GMAC/s.
-module(mmrt_test).
-export([start/0]).

start() ->
    Cases = mmrt_vectors:cases(),
    Fails = [Name || {Name, Fun, Args, Want} <- Cases, not check(Name, Fun, Args, Want)],
    io:format("mmrt: ~p/~p cases pass~n", [length(Cases) - length(Fails), length(Cases)]),
    errors(),
    bench(),
    io:format("mmrt_test done~n"),
    timer:sleep(infinity).

check(Name, Fun, Args, Want) ->
    Got = (catch apply(mmrt, Fun, [prep(A) || A <- Args])),
    case Got =:= Want of
        true -> true;
        false -> io:format("FAIL ~s: got ~p~n", [Name, Got]), false
    end.

prep({packed, RowMajor, R, C}) -> mmrt:pack(RowMajor, R, C);
prep(A) -> A.

%% Bad input raises badarg, oversized work raises too_big.
errors() ->
    M = mmrt:pack({int8, <<1:8/unit:8>>}, 1, 8),
    Expect = [{badarg, fun() -> mmrt:matvec(M, {int8, <<1, 2, 3>>}) end},
              {badarg, fun() -> mmrt:dot({int8, <<1>>}, {int8, <<1, 2>>}) end},
              {badarg, fun() -> mmrt:pack({int8, <<1, 2, 3>>}, 2, 2) end},
              {badarg, fun() -> mmrt:add({int8, <<1>>}, {int32, <<1:32>>}) end},
              {too_big, fun() -> mmrt:matmul(mmrt:pack({int8, <<0:(1024*1024)/unit:8>>}, 1024, 1024),
                                             {int8, <<0:(5*1024)/unit:8>>}, 0) end}],
    Bad = [W || {W, F} <- Expect, not raised(W, F)],
    io:format("mmrt: errors ~s~n", [case Bad of [] -> "ok"; _ -> io_lib:format("WRONG ~p", [Bad]) end]).

raised(W, F) -> try F(), false catch error:W -> true; _:_ -> false end.

bench() ->
    R = 256, C = 256,
    W8 = mmrt:pack({int8, bytes(R * C, 31)}, R, C),
    W4 = mmrt:pack(mmrt:to_int4({int8, bytes(R * C, 7)}), R, C),
    Big = mmrt:pack({int8, bytes(R * C, 127)}, R, C),       % forces the exact C path
    X1 = {int8, bytes(C, 15)}, X16 = {int8, bytes(16 * C, 15)}, XBig = {int8, bytes(C, 127)},
    run("matvec int8 256x256 (SIMD)", R * C, fun() -> mmrt:matvec(W8, X1, 8) end),
    run("matmul int8 256x256 T=16 (SIMD)", 16 * R * C, fun() -> mmrt:matmul(W8, X16, 8) end),
    run("matmul int4 256x256 T=16 (SIMD)", 16 * R * C, fun() -> mmrt:matmul(W4, X16, 8) end),
    run("matvec int8 256x256 (C, full-scale)", R * C, fun() -> mmrt:matvec(Big, XBig, 8) end),
    run("matvec int8 256x256 exact int32 (C)", R * C, fun() -> mmrt:matvec(W8, X1) end),
    {_, B} = X1,
    run("dot int8 x256 in C (NIF)", C, fun() -> mmrt:dot(X1, X1) end),
    L = [V - 256 * (V bsr 7) || V <- binary_to_list(B)],
    run("dot int8 x256 in pure Erlang (lists)", C, fun() -> erl_dot(L, L, 0) end),
    run("dot int8 x256 in pure Erlang (binary)", C, fun() -> erl_bdot(B, B, 0) end).

run(Name, Macs, F) ->
    F(),
    T0 = erlang:monotonic_time(microsecond),
    {N, Us} = timed(F, T0, 0),
    io:format("  ~-40s ~9.1f us/call  ~8.2f MMAC/s~n", [Name, Us / N, Macs * N / Us]).

%% call F in batches of 10 until half a second has passed
timed(F, T0, N) ->
    loop(F, 10),
    Us = erlang:monotonic_time(microsecond) - T0,
    if Us >= 500000 -> {N + 10, Us}; true -> timed(F, T0, N + 10) end.

loop(_, 0) -> ok;
loop(F, N) -> F(), loop(F, N - 1).

%% deterministic pseudo-random int8 bytes in -Amp..Amp: a 509-byte pattern, tiled natively
%% (building 64 KB element by element -- comprehension or list -- takes minutes on AtomVM)
bytes(N, Amp) ->
    Pat = list_to_binary([((I * 7919 + 13) rem (2 * Amp + 1) - Amp) band 255 || I <- lists:seq(1, 509)]),
    binary:part(binary:copy(Pat, N div 509 + 1), 0, N).

erl_dot([A | RA], [B | RB], Acc) -> erl_dot(RA, RB, Acc + A * B);
erl_dot([], [], Acc) -> Acc.

erl_bdot(<<A:8/signed, RA/binary>>, <<B:8/signed, RB/binary>>, Acc) -> erl_bdot(RA, RB, Acc + A * B);
erl_bdot(<<>>, <<>>, Acc) -> Acc.
