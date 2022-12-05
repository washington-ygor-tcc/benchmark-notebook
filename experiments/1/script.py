import time
import numpy as np
import matplotlib.pyplot as plt

from benchmark.application import app, types
from benchmark.application.types import BenchmarkParams


def get_result(param):
    result = app.run_benchmark(param)

    time.sleep(5)

    return result


benchmark_types = [types.BenchmarkTypes.API, types.BenchmarkTypes.MSG]
batch_list = [2**i for i in range(13)]

bench_param_list = [
    BenchmarkParams(
        benchmark_type=bench_type,
        complexity_factor=100,
        memory_overhead=1,
        requests_number=4096,
        batch_size=batch_size,
        total_progress=True,
    )
    for bench_type in benchmark_types
    for batch_size in batch_list
]

results = [get_result(param) for param in bench_param_list]

elapsed_times = [[t.end - t.start for t in result.response_list] for result in results]


def get_requests_per_second(results):
    return [len(result.response_list) / result.elapsed_time for result in results]


def get_standard_deviation(elapsed_times):
    return [np.std(elapsed_time) for elapsed_time in elapsed_times]


def plot_requests_by_batch():
    fig, axs = plt.subplots(nrows=2, figsize=(12, 16))

    axs[0].boxplot(elapsed_times[: len(batch_list)], labels=batch_list)
    axs[0].set_title("API RPC", fontsize=10)
    axs[0].set_xlabel("Tamanho do lote de requisição")
    axs[0].set_ylabel("Tempo de resposta p/ cada requisição (s)")
    axs[0].grid(True)

    axs[1].boxplot(elapsed_times[len(batch_list) :], labels=batch_list)
    axs[1].set_title("Mensageria", fontsize=10)
    axs[1].set_xlabel("Tamanho do lote de requisição")
    axs[1].set_ylabel("Tempo de resposta p/ cada requisição (s)")
    axs[1].grid(True)

    fig.subplots_adjust(top=0.93, hspace=0.2, wspace=0.2)

    plt.savefig("plots/requests_by_batch.png")
    plt.show()


def plot_total_time_by_batch():
    plt.plot(
        batch_list,
        [result.elapsed_time for result in results[: len(batch_list)]],
        ":r",
        label="API RPC",
    )
    plt.plot(
        batch_list,
        [result.elapsed_time for result in results[len(batch_list) :]],
        "--b",
        label="Mensageria",
    )

    plt.legend()
    plt.xlabel("Tamanho do lote de requisição")
    plt.ylabel("Tempo total para realizar 4096 requisições (s)")

    plt.grid(True)
    plt.xscale("log", base=2)
    plt.xticks(batch_list)

    plt.savefig("plots/total_time_by_batch.png")
    plt.show()


def plot_req_per_second():
    plt.plot(
        batch_list,
        get_requests_per_second(results[: len(batch_list)]),
        ":r",
        label="API RPC",
    )
    plt.plot(
        batch_list,
        get_requests_per_second(results[len(batch_list) :]),
        "--b",
        label="Mensageria",
    )

    plt.legend()
    plt.xlabel("Tamanho do lote de requisição")
    plt.ylabel("Requisições por segundo")

    plt.grid(True)
    plt.xscale("log", base=2)
    plt.xticks(batch_list)

    plt.savefig("plots/req_per_second_by_batch.png")
    plt.show()


def plot_standard_deviation():
    plt.plot(
        batch_list,
        get_standard_deviation(elapsed_times[: len(batch_list)]),
        ":r",
        label="API RPC",
    )
    plt.plot(
        batch_list,
        get_standard_deviation(elapsed_times[len(batch_list) :]),
        "--b",
        label="Mensageria",
    )

    plt.legend()
    plt.xlabel("Tamanho do lote de requisição")
    plt.ylabel("Desvio padrão do tempo de resposta (s)")

    plt.grid(True)
    plt.xscale("log", base=2)
    plt.xticks(batch_list)

    plt.savefig("plots/standard_deviation_by_batch.png")
    plt.show()


plot_requests_by_batch()
plot_total_time_by_batch()
plot_req_per_second()
plot_standard_deviation()
