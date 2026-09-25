%% MMRT: int8/int4 vector kernels for AtomVM on the ESP32-S3 (PIE SIMD), over tagged binaries.
%%
%% Data:
%%   {int8, Bin}    one signed byte per element
%%   {int4, Bin}    signed nibbles (-8..7), low nibble first, two per byte
%%   {int32, Bin}   little-endian int32 (accumulators)
%%   {int8_m16 | cb4_m16, Rows, Cols, MaxRowL1, Bin}
%%                  a matrix packed by pack/3 for the SIMD kernel. Pack once, reuse. Bin is
%%                  an ordinary binary: keep it, send it, or build it on the host.
%%
%% matvec/matmul use the SIMD unit when the accumulators provably fit its 20-bit lanes
%% (MaxRowL1 * max|x| < 2^19) and an exact C path otherwise: the results are identical.
%% Calls run on the calling scheduler, so each one is capped at 4M multiply-adds (a few
%% ms); bigger calls raise too_big -- split the work.
-module(mmrt).
-export([pack/3, matvec/2, matvec/3, matmul/3, dot/2, add/2, relu/1, requant/2,
         argmax/1, top_k/2, to_int4/1, from_int4/1]).

-type int8() :: {int8, binary()}.
-type int4() :: {int4, binary()}.
-type int32() :: {int32, binary()}.
-type vec() :: int8() | int4() | int32().
-type matrix() :: {int8_m16 | cb4_m16, pos_integer(), pos_integer(), non_neg_integer(), binary()}.
-export_type([int8/0, int4/0, int32/0, vec/0, matrix/0]).

%% Pack a row-major Rows x Cols matrix: int8 -> int8_m16, int4 -> cb4_m16 (half the bytes).
-spec pack(int8() | int4(), pos_integer(), pos_integer()) -> matrix().
pack(_RowMajor, _Rows, _Cols) -> erlang:nif_error(undefined).

%% M . X, exact: Rows int32 accumulators.
-spec matvec(matrix(), int8()) -> int32().
matvec(_M, _X) -> erlang:nif_error(undefined).

%% sat8(round_half_up((M . X) >> Shift)): Rows int8 (Shift < 0 multiplies by 2^-Shift).
-spec matvec(matrix(), int8(), integer()) -> int8().
matvec(_M, _X, _Shift) -> erlang:nif_error(undefined).

%% X holds T rows of Cols; the result T rows of Rows, each as matvec/3. Weights are loaded
%% once for all T rows, so this is much faster than T matvec calls.
-spec matmul(matrix(), int8(), integer()) -> int8().
matmul(_M, _X, _Shift) -> erlang:nif_error(undefined).

%% Exact dot product of two vectors of the same length (any element types).
-spec dot(vec(), vec()) -> integer().
dot(_A, _B) -> erlang:nif_error(undefined).

%% Saturating elementwise sum, both int8 or both int32.
-spec add(int8(), int8()) -> int8(); (int32(), int32()) -> int32().
add(_A, _B) -> erlang:nif_error(undefined).

-spec relu(int8()) -> int8(); (int32()) -> int32().
relu(_V) -> erlang:nif_error(undefined).

%% int32 -> int8: sat8(round_half_up(V >> Shift)).
-spec requant(int32(), integer()) -> int8().
requant(_V, _Shift) -> erlang:nif_error(undefined).

%% 0-based index of the first largest element.
-spec argmax(vec()) -> non_neg_integer().
argmax(_V) -> erlang:nif_error(undefined).

%% The K largest elements as [{Index, Value}], largest first, ties by lower index.
-spec top_k(vec(), non_neg_integer()) -> [{non_neg_integer(), integer()}].
top_k(_V, _K) -> erlang:nif_error(undefined).

%% int8 -> int4 (clamped to -8..7); int4 -> int8 (an odd-length vector gains a trailing 0).
-spec to_int4(int8()) -> int4().
to_int4(_V) -> erlang:nif_error(undefined).
-spec from_int4(int4()) -> int8().
from_int4(_V) -> erlang:nif_error(undefined).
