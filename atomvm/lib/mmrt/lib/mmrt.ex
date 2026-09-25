defmodule MMRT do
  @moduledoc """
  int8/int4 vector kernels for AtomVM on the ESP32-S3: a thin wrapper over the Erlang
  `:mmrt` module (see `atomvm/lib/mmrt/src/mmrt.erl` for the data formats and docs).

      w = MMRT.pack({:int8, weights}, 64, 128)
      {:int8, y} = MMRT.matvec(w, {:int8, x}, 7)
  """
  defdelegate pack(row_major, rows, cols), to: :mmrt
  defdelegate matvec(m, x), to: :mmrt
  defdelegate matvec(m, x, shift), to: :mmrt
  defdelegate matmul(m, x, shift), to: :mmrt
  defdelegate dot(a, b), to: :mmrt
  defdelegate add(a, b), to: :mmrt
  defdelegate relu(v), to: :mmrt
  defdelegate requant(v, shift), to: :mmrt
  defdelegate argmax(v), to: :mmrt
  defdelegate top_k(v, k), to: :mmrt
  defdelegate to_int4(v), to: :mmrt
  defdelegate from_int4(v), to: :mmrt
end
