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
requests_number_list = [1000 * (i + 1) for i in range(25)]

bench_param_list = [
    BenchmarkParams(
        benchmark_type=bench_type,
        complexity_factor=100,
        memory_overhead=0,
        requests_number=requests_number,
        batch_size=500,
        total_progress=True,
    )
    for requests_number in requests_number_list
    for bench_type in benchmark_types
]

results = [get_result(param) for param in bench_param_list]

elapsed_times = [[t.end - t.start for t in result.response_list] for result in results]


def get_requests_per_second(results):
    return [len(result.response_list) / result.elapsed_time for result in results]


def get_median(results):
    return [np.median(result) for result in results]


def get_standard_deviation(elapsed_times):
    return [np.std(elapsed_time) for elapsed_time in elapsed_times]


def plot_req_per_second():
    plt.plot(
        requests_number_list,
        get_requests_per_second(results[::2]),
        ":r",
        label="API RPC",
    )
    plt.plot(
        requests_number_list,
        get_requests_per_second(results[1::2]),
        "--b",
        label="Mensageria",
    )

    plt.legend()
    plt.xlabel("Quantidade de requisições")
    plt.ylabel("Requisições por segundo")

    plt.grid(True)

    plt.savefig("plots/req_per_second_by_requests_number.png")
    plt.show()


def plot_standard_deviation():
    plt.plot(
        requests_number_list,
        get_standard_deviation(elapsed_times[::2]),
        ":r",
        label="API RPC",
    )
    plt.plot(
        requests_number_list,
        get_standard_deviation(elapsed_times[1::2]),
        "--b",
        label="Mensageria",
    )

    plt.legend()
    plt.xlabel("Quantidade de requisições")
    plt.ylabel("Desvio padrão do tempo de resposta (s)")

    plt.grid(True)

    plt.savefig("plots/standard_deviation_by_requests_number.png")
    plt.show()


def plot_median():
    plt.plot(
        requests_number_list, get_median(elapsed_times[::2]), ":r", label="API RPC"
    )
    plt.plot(
        requests_number_list, get_median(elapsed_times[1::2]), "--b", label="Mensageria"
    )

    plt.legend()
    plt.xlabel("Quantidade de requisições")
    plt.ylabel("Média tempo de resposta (s)")

    plt.grid(True)

    plt.savefig("plots/median_by_requests_number.png")
    plt.show()


plot_median()
plot_req_per_second()
plot_standard_deviation()
