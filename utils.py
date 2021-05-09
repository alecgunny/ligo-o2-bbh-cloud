import concurrent.futures

import attr
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
