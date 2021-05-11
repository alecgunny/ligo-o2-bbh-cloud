import concurrent.futures
import multiprocessing as mp
import queue
import re
import time

import attr
import requests
from cloud_utils.utils import wait_for


_PACKAGE = "ligo-o2-bbh-cloud"
_PACKAGE_URL = f"https://github.com/alecgunny/{_PACKAGE}.git"
_RUN = f"./{_PACKAGE}/client/run.sh"


def _run_in_pool(fn, args, msg, exit_msg):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for a in args:
            if not isinstance(a, tuple):
                a = (a,)
            futures.append(executor.submit(fn, *a))

        results = []

        def _callback():
            for f in futures:
                try:
                    result = f.result(timeout=1e-3)
                    results.append(result)
                    futures.remove(f)
                except concurrent.futures.TimeoutError:
                    continue
            return len(futures) == 0

        wait_for(_callback, msg, exit_msg)
        return results


def configure_vm(vm):
    vm.wait_until_ready(verbose=False)

    cmds = [f"git clone -q {_PACKAGE_URL}", f"{_RUN} install", f"{_RUN} create"]
    for cmd in cmds:
        _, err = vm.run(cmd)
        if err:
            raise RuntimeError(f"Command {cmd} failed with message {err}")


def configure_vms_parallel(vms):
    _run_in_pool(
        configure_vm,
        vms,
        msg="Waiting for VMs to configure",
        exit_msg="Configured all VMs"
    )


@attr.s(auto_attribs=True)
class RunParallel:
    model_name: str
    model_version: int
    generation_rate: float
    sequence_id: int
    bucket_name: str
    kernel_stride: float
    sample_rate: float = 4000
    chunk_size: float = 1024

    @property
    def command(self):
        command = f"{_RUN} run"
        for a in self.__attrs_attrs__:
            if a.name != "sequence_id":
                command += " --{} {}".format(
                    a.name.replace("_", "-"), self.__dict__[a.name]
                )
        return command + (
            " --url {ip}:8001 --sequence-id {sequence_id} --prefix {prefix}"
        )

    def run_on_vm(self, vm, fname, ip, sequence_id):
        command = self.command.format(
            ip=ip,
            sequence_id=sequence_id,
            prefix=fname
        )
        out, err = vm.run(command)

        # parse out framecpp stderr info
        lines = []
        for line in err.split("\n"):
            if line and not line.startswith("Loading: Fr"):
                lines.append(line)

        err = "\n".join(lines)
        if err:
            raise RuntimeError(err)

        fname = fname.split("/")[-1].replace("gwf", "npy")
        vm.get(f"{_PACKAGE}/client/outputs.npy", fname)

    def __call__(self, vms, files, ips):
        seq_ids = [self.sequence_id + i for i in range(len(vms))]
        args = zip(vms, files, ips, seq_ids)

        _run_in_pool(
            self.run_on_vm,
            args,
            "Waiting for tasks to complete",
            "All tasks completed"
        )


_hexes = "[0-9a-f]"
_gpu_id_pattern = "-".join([_hexes + f"{{{i}}}" for i in [8, 4, 4, 4, 12]])
_res = [
    re.compile('(?<=nv_inference_)[a-z_]+(?=_duration_us{gpu_uuid="GPU-)'),
    re.compile(f'(?<=gpu_uuid="GPU-){_gpu_id_pattern}(?=")'),
    re.compile(f'(?<=model=")[a-z_]+(?=",version=)'),
    re.compile("(?<=} )[0-9.]+$")
]


class ServerMonitor(mp.Process):
    def __init__(self, ips, filename, models):
        self.ips = ips
        self.filename = filename
        self.models = models

        self.header = (
            "step,ip,gpu_id,model,process,time (us),interval,count,utilization"
        )
        self._last_times = {}
        self._counts = {}
        self._times = {}
        self._steps = {ip: 0 for ip in self.ips}

        self._stop_event = mp.Event()
        self._error_q = mp.Queue()
        super().__init__()

    def _get_data_for_ip(self, ip):
        response = requests.get(f"http://{ip}:8002/metrics")
        response.raise_for_status()

        request_time = time.time()
        try:
            last_time = self._last_times[ip]
            interval = request_time - last_time
        except KeyError:
            pass
        finally:
            self._last_times[ip] = request_time

        data, counts, utilizations = "", {}, {}
        models_to_update = []
        rows = response.content.decode().split("\n")

        # start by collecting the number of new inference
        # counts and the GPU utilization
        for row in rows:
            if row.startswith("nv_inference_exec_count"):
                try:
                    gpu_id, model, value = [
                        r.search(row).group(0) for r in _res[1:]
                    ]
                except AttributeError:
                    continue

                if model in self.models:
                    value = int(float(value))
                    try:
                        count = value - self._counts[(ip, gpu_id, model)]
                        if count > 0:
                            # if no new inferences were registered, we
                            # won't need to update this model below
                            models_to_update.append((gpu_id, model))
                        counts[(ip, gpu_id, model)] = count
                    except KeyError:
                        # we haven't recorded this model before, so
                        # there's no need to update it
                        pass
                    finally:
                        # add or update the number of inference counts
                        # for this model on this GPU
                        self._counts[(ip, gpu_id, model)] = value

            elif row.startswith("nv_gpu_utilization"):
                try:
                    gpu_id, value = [r.search(row).group(0) for r in _res[1::2]]
                except AttributeError:
                    continue
                utilizations[gpu_id] = value

        for row in rows:
            try:
                process, gpu_id, model, value = [
                    r.search(row).group(0) for r in _res
                ]
            except AttributeError:
                continue

            value = float(value)
            index = (ip, process, gpu_id, model)

            if (gpu_id, model) not in models_to_update:
                # we don't need to record a row of data for this
                # GPU/model combination, either because this is
                # the first loop or because we didn't record any
                # new inferences during this interval
                if model in self.models and index not in self._times:
                    # we don't have a duration for this process
                    # on this node/GPU/model combo, so create one
                    self._times[index] = value
                continue

            delta = value - self._times[index]
            self._times[index] = value
            utilization = utilizations[gpu_id]
            count = counts[(ip, gpu_id, model)]

            data += "\n" + ",".join([
                ip,
                str(self._steps[ip]),
                gpu_id,
                model,
                process,
                str(delta),
                str(interval),
                str(count),
                utilization
            ])
        return data

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def check(self):
        try:
            e = self._error_q.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("Error in monitor: " + e)

    def run(self):
        f = open(self.filename, "w")
        f.write(self.header)
        try:
            while not self.stopped:
                for ip in self.ips:
                    data = self._get_data_for_ip(ip)
                    if data:
                        f.write(data)
                        self._steps[ip] += 1
        except Exception as e:
            f.close()
            self._error_q.put(str(e))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_args):
        self.stop()
        self.join(1)
        try:
            self.close()
        except ValueError:
            self.terminate()
