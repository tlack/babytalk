defmodule TcpUploadRepro.MixProject do
  use Mix.Project

  def project do
    [
      app: :tcp_upload_repro,
      version: "0.1.0",
      elixir: "~> 1.17",
      erlc_paths: ["src"],
      deps: [{:exatomvm, git: "https://github.com/atomvm/exatomvm.git", runtime: false}],
      atomvm: [start: :tcp_upload_repro, flash_offset: 0xA90000]
    ]
  end

  def application, do: []
end
