from multiprocessing import dummy as mp
from queue import Empty

import attr


_PACKAGE = "ligo-o2-bbh-cloud"


def configure_vm(vm):
    cmds = [
        f"git clone -q https://github.com/alecgunny/{_PACKAGE}.git",
        f"./{_PACKAGE}/client/init.sh install",
        f"./{_PACKAGE}/client/init.sh create",
    ]

    for cmd in cmds:
        _, err = vm.run(cmd)
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
            error_q.put(str(e))

    with mp.Pool(len(vms)) as pool:
        pool.map(f, vms, chunksize=1)
    pool.join()

    try:
        e = error_q.get_nowait()
    except Empty:
        return
    else:
        raise RuntimeError(e)


@attr.s(auto_attribs=True)
class RunParallel:
    url: str
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
        command = f"./{_PACKAGE}/client/init.sh run"
        for a in self.__attrs_attrs__:
            if a.name == "sequence_id":
                continue

            command += " --" + a.name.replace("_", "-")
            command += " " + str(self.__dict__[a.name])
        command += " --sequence-id {sequence_id}"
        command += " --prefix {prefix}"
        return command

    def __call__(self, vms, files):
        error_q = mp.Queue()

        def f(vm, _file, i):
            try:
                command = self.command.format(
                    sequence_id=self.sequence_id + i,
                    prefix=_file
                )
                _, err = vm.run(command)
                if err:
                    raise RuntimeError(err)

                fname = _file.split("/")[-1].replace("gwf", "npy")
                vm.get(f"{_PACKAGE}/client/outputs.npy", fname)
            except Exception as e:
                error_q.put(str(e))

        args = zip(vms, files, range(len(vms)))
        with mp.Pool(len(vms)) as pool:
            pool.starmap(f, args, chunksize=1)
        pool.join()

        try:
            e = error_q.get_nowait()
        except Empty:
            return
        else:
            raise RuntimeError(e)
