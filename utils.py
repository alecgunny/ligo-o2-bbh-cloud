from multiprocessing import dummy as mp
from queue import Empty

from stillwater.utils import ExceptionWrapper


def configure_vm(vm):
    package = "ligo-o2-bbh-cloud"
    cmds = [
        f"git clone -q https://github.com/alecgunny/{package}.git",
        f"./{package}/client/init.sh install",
        f"./{package}/client/init.sh create",
    ]

    for cmd in cmds:
        our, err = vm.run(cmd)
        if err:
            raise RuntimeError(
                f"Command {cmd} failed with message {err}"
            )


def configure_vms_parallel(vms):
    error_q = mp.Queue()

    def f(vm):
        try:
            configure_vm(vm)
        except Exception as e:
            error_q.put(ExceptionWrapper(e))

    with mp.Pool(len(vms)) as pool:
        pool.map(configure_vm, vms, chunksize=1)
        pool.join()

    try:
        e = error_q.get_nowait()
    except Empty:
        return
    else:
        e.reraise()
