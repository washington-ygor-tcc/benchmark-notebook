import time
import numpy as np
import matplotlib.pyplot as plt

from benchmark.application import app, types
from benchmark.application.types import BenchmarkParams


def get_result(param):
    result = app.run_benchmark(param)

    time.sleep(3)

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
    axs[0].set_title("API", fontsize=10)
    axs[0].plot(
        axs[0].get_xticks(),
        [result.elapsed_time for result in results[: len(batch_list)]],
        "o--b",
        label="Tempo total",
    )
    axs[0].legend()

    axs[1].boxplot(elapsed_times[len(batch_list) :], labels=batch_list)
    axs[1].set_title("MSG", fontsize=10)
    axs[1].plot(
        axs[1].get_xticks(),
        [result.elapsed_time for result in results[len(batch_list) :]],
        "o--b",
        label="Tempo total",
    )
    axs[1].legend()

    fig.supxlabel("Tamanho do lote de requisição")
    fig.supylabel("Tempo de resposta p/ cada requisição")
    fig.suptitle("4096 requisições separadas por lotes")
    fig.subplots_adjust(top=0.93, hspace=0.2, wspace=0.2)

    plt.savefig("plots/requests_by_batch.png")
    plt.show()


def plot_req_per_second():
    plt.plot(
        batch_list, get_requests_per_second(results[: len(batch_list)]), label="API"
    )
    plt.plot(
        batch_list, get_requests_per_second(results[len(batch_list) :]), label="MSG"
    )

    plt.legend()
    plt.xlabel("Tamanho do lote de requisição")
    plt.ylabel("Req/s")
    plt.title("Requisições por segundo para cada tamanho de lote")

    plt.grid(True)
    plt.xscale("log", base=2)
    plt.xticks(batch_list)

    plt.savefig("plots/req_per_second_by_batch.png")
    plt.show()


def plot_standard_deviation():
    plt.plot(
        batch_list,
        get_standard_deviation(elapsed_times[: len(batch_list)]),
        label="API",
    )
    plt.plot(
        batch_list,
        get_standard_deviation(elapsed_times[len(batch_list) :]),
        label="MSG",
    )

    plt.legend()
    plt.xlabel("Tamanho do lote de requisição")
    plt.ylabel("Desvio padrão")
    plt.title("Desvio padrão observado para cada tamanho de lote")

    plt.grid(True)
    plt.xscale("log", base=2)
    plt.xticks(batch_list)

    plt.savefig("plots/standard_deviation_by_batch.png")
    plt.show()


plot_requests_by_batch()
plot_req_per_second()
plot_standard_deviation()
