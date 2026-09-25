%% BabyTalk: on-device speech for AtomVM on the ESP32-S3. The functions are NIFs implemented
%% in atomvm/components/atomvm_babytalk (C); these stubs only run if the firmware lacks them.
-module(babytalk).
-export([heap_info/0]).

%% Free and largest-block bytes of internal RAM and PSRAM.
-spec heap_info() -> [{internal_free | internal_largest | psram_free | psram_largest, non_neg_integer()}].
heap_info() ->
    erlang:nif_error(undefined).
