import concurrent.futures
import logging
import multiprocessing as mp
import os
import queue
import re
import threading
import time
import typing
from zlib import adler32
from functools import lru_cache

import attr
import requests
from cloud_utils.utils import wait_for


_PACKAGE = "ligo-o2-bbh-cloud"
_PACKAGE_URL = f"https://github.com/alecgunny/{_PACKAGE}.git"
_RUN = f"./{_PACKAGE}/client/run.sh"
_MODELS = [
    "gwe2e", "snapshotter", "deepclean_l", "deepclean_h", "postproc", "bbh"
]
logging.getLogger("paramiko").setLevel(logging.CRITICAL)


def parse_blob_fname(name):
    # TODO: copied and pasted from client/frame_reader.py
    # How can we make this more general AND make it available
    # to both the client and this host code without making
    # a full subrepo for it?
    name = name.replace(".gwf", "")
    timestamp, length = tuple(map(int, name.split("-")[-2:]))
    return timestamp, length


def __run_in_pool(fn, args, msg, exit_msg, max_workers=None):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    with executor:
        futures = []
        for a in args:
            if not isinstance(a, tuple):
                a = (a,)
            futures.append(executor.submit(fn, *a))
        time.sleep(3)

        results = []
        start_time = time.time()

        def _callback():
            done, _ = concurrent.futures.wait(futures, timeout=1e-3)
            for f in done:
                results.append(f.result())
                futures.remove(f)

            num_done = len(results)
            num_pending = len([f for f in futures if not f.running()])
            num_running = len([f for f in futures if f.running()])
            elapsed = time.time() - start_time

            status = " Pending={}, Running={}, Completed={} after {} s".format(
                num_pending, num_running, num_done, int(elapsed)
            )
            return msg + status, len(futures) == 0

        wait_for(_callback, msg, exit_msg)
        return results


def _run_in_pool(fn, args, msg, exit_msg, max_workers=None):
    q = queue.Queue()

    def wrapper(*args):
        try:
            result = fn(*args)
            q.put(result)
        except Exception as e:
            q.put(e)

    threads = []
    for arg in args:
        if not isinstance(arg, tuple):
            arg = (arg,)
        t = threading.Thread(target=wrapper, args=arg)
        t.start()
        threads.append(t)

    results = []

    def _callback():
        try:
            result = q.get_nowait()
            if isinstance(result, Exception):
                raise result
            results.append(result)
        except queue.Empty:
            pass
        return not any([t.is_alive() for t in threads])

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
        exit_msg="Configured all VMs",
        max_workers=min(32, len(vms))
    )


@attr.s(auto_attribs=True)
class RunParallel:
    model_name: str
    model_version: int
    generation_rate: float
    sequence_id: int
    bucket_name: str
    kernel_stride: float
    length: float
    sample_rate: float = 4000
    chunk_size: float = 1024
    output_dir: typing.Optional[str] = None

    def __attrs_post_init__(self):
        self._start_time = None

    @property
    def command(self):
        command = f"{_RUN} run"
        for a in self.__attrs_attrs__:
            if a.name not in ("sequence_id", "output_dir"):
                command += " --{} {}".format(
                    a.name.replace("_", "-"), self.__dict__[a.name]
                )
        if self._start_time is not None:
            command += f" --start-time {self._start_time}"

        return command + (
            " --url {ip}:8001 --sequence-id {sequence_id} --t0 {t0}"
        )

    def run_on_vm(self, vm, t0, ip, sequence_id):
        command = self.command.format(ip=ip, sequence_id=sequence_id, t0=t0)
        out, err = vm.run(command)

        # parse out framecpp stderr info
        err_re = re.compile("^Loading: Fr.+$", re.MULTILINE)
        err = err_re.sub("", err).strip()
        if err:
            raise RuntimeError(err)

        fname = f"{t0}-{self.length}.npy"
        if self.output_dir is not None:
            fname = os.path.join(self.output_dir, fname)
        vm.get(f"{_PACKAGE}/client/outputs.npy", fname)

    def __call__(self, vms, t0s, ips):
        seq_ids = [self.sequence_id + i for i in range(len(vms))]
        args = zip(vms, t0s, ips, seq_ids)

        self._start_time = time.time() + 20
        _run_in_pool(
            self.run_on_vm,
            args,
            "Waiting for tasks to complete",
            "All tasks completed",
            max_workers=min(32, len(vms))
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
    def __init__(self, ips, filename):
        self.ips = ips
        self.filename = filename

        self.header = (
            "ip,step,gpu_id,model,process,time (us),interval,count,utilization"
        )
        self._last_times = {}
        self._counts = {}
        self._times = {}
        self._steps = {ip: 0 for ip in self.ips}

        self._stop_event = mp.Event()
        self._error_q = mp.Queue()
        super().__init__()

    def _get_data_for_ip(self, ip, step):
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

                if model in _MODELS:
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
                if model in _MODELS and index not in self._times:
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
                str(step),
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

        lock = threading.Lock()

        def target(ip):
            step = 0
            try:
                while not self.stopped:
                    data = self._get_data_for_ip(ip, step)
                    if data:
                        with lock:
                            f.write(data)
                        step += 1
            except Exception:
                self.stop()
                raise

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(target, ip) for ip in self.ips]

        try:
            while len(futures) > 0:
                done, futures = concurrent.futures.wait(futures, timeout=1e-2)
                for future in done:
                    exc = future.exception()
                    if exc is not None:
                        raise exc
        except Exception as e:
            self._error_q.put(str(e))
        finally:
            f.close()

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


@attr.s(auto_attribs=True, frozen=True)
class InstanceConfig:
    deepclean_h: int
    deepclean_l: int
    postproc: int
    bbh: int

    def __str__(self):
        string = "{"
        for a in self.__attrs_attrs__:
            value = self.__dict__[a.name]
            string += f"\n\t{a.name}: {value},"
        string = string[:-1]
        return string + "\n}"


@attr.s(auto_attribs=True, frozen=True)
class RunConfig:
    num_nodes: int
    gpus_per_node: int
    clients_per_node: int
    instance_config: InstanceConfig
    vcpus_per_gpu: int
    kernel_stride: float
    generation_rate: float

    @lru_cache(None)
    def _make_string(self):
        string = "{"
        for a in self.__attrs_attrs__:
            value = self.__dict__[a.name]
            if a.type is float:
                value = float(value)
            value = str(value)

            if "\n" in value:
                lines = value.split("\n")
                lines = lines[:1] + ["\t" + line for line in lines[1:]]
                value = "\n".join(lines)
            string += f"\n\t{a.name}: {value},"
        string = string[:-1]
        return string + "\n}"

    @property
    def total_clients(self):
        return self.clients_per_node * self.num_nodes

    def __str__(self):
        return f"RunConfig {self.id} " + self._make_string()

    @property
    def id(self):
        string = self._make_string()
        return hex(adler32(string.encode())).split("x")[1]
