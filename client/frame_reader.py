import multiprocessing as mp
import os
import queue
import threading
import time
import typing
import yaml
from functools import wraps

import attr
import numpy as np
from google.cloud import storage
from gwpy.timeseries import TimeSeriesDict
from requests import HTTPError

from stillwater.utils import ExceptionWrapper, Package


def _catch_wrap_and_put(f):
    """
    function decorator for running async
    targets for either threads or processes
    in a try-catch that passes exceptions
    to parent processes if they occur
    """
    @wraps(f)
    def wrapper(q, *args, **kwargs):
        try:
            stuff = f(q, *args, **kwargs)
        except Exception as e:
            q.put(ExceptionWrapper(e))
        else:
            return stuff
    return wrapper


def _eager_get(q, p):
    """
    wrapper function for trying to get items
    out of a queue `q` for a child thread/process
    `p`. If `p` passes along an error, reraise it.
    Otherwise if there's nothing to get out of `q`
    and `p` isn't alive (which we can take to mean
    that nothing more is going to be put into `q`),
    go ahead and raise a StopIteration
    """
    while True:
        try:
            item = q.get_nowait()
            break
        except queue.Empty:
            if not p.is_alive():
                raise StopIteration
            continue
    if isinstance(item, ExceptionWrapper):
        item.reraise()
    return item


class _RaisedFromParent(Exception):
    """
    Dummmy exception for breaking out of outer while loop
    below when in the while loop checking for q fullness
    """
    pass


@_catch_wrap_and_put
def _download_and_write_frames(q, blobs, stop_event):
    for blob in blobs:
        if not blob.name.endswith(".gwf"):
            continue

        # check first if we even need to do this
        if stop_event.is_set():
            break
        blob_bytes = blob.download_as_bytes()

        # TODO: how to we make this general?
        fname = blob.name.split("/")[-1]
        timestamp = fname.split("-")[2]

        # check again here because things could have
        # changed in the time it took us to download
        # and this could be as time consuming
        if stop_event.is_set():
            break
        with open(timestamp + ".gwf", "wb") as f:
            f.write(blob_bytes)
        q.put(int(timestamp))


@_catch_wrap_and_put
def read_frames(
    q,
    stop_event,
    bucket_name,
    sample_rate,
    channels,
    segment_length,
    prefix=None
):
    client = storage.Client()
    try:
        bucket = client.get_bucket(bucket_name)
    except HTTPError as e:
        if e.code == 404:
            raise ValueError(f"Couldn't find bucket {bucket_name}")
        content = yaml.safe_load(e.response.content.decode("utf-8"))
        raise RuntimeError(
            f"Encountered HTTPError with code {e.code} and message: "
            + content["error"]["message"]
        )

    # thread the download so that we can iterate
    # through it in parallel. This is a super IO
    # bound operation so this shouldn't slow things
    # down (I think...)
    blobs = bucket.list_blobs(prefix=prefix)
    loader_q = queue.Queue()
    loader_event = threading.Event()
    loader = threading.Thread(
        target=_download_and_write_frames, args=(loader_q, blobs, loader_event)
    )
    loader.start()
    time.sleep(0.1)

    try:
        while True:
            timestamp = _eager_get(loader_q, loader)
            print(timestamp)

            fname = str(timestamp) + ".gwf"
            start = timestamp + 0
            while start < timestamp + 4096:
                end = min(
                    start + segment_length,
                    timestamp + 4096
                )
                timeseries = TimeSeriesDict.read(
                    fname,
                    channels=channels,
                    format="gwf",
                    start=start,
                    end=end
                )
                timeseries.resample(sample_rate)
                frame = np.stack(
                    [timeseries[channel].value for channel in channels]
                )

                try:
                    while q.full():
                        if stop_event.is_set():
                            raise _RaisedFromParent
                except _RaisedFromParent:
                    break
                q.put(frame)
                start += 4096
            os.remove(fname)
    finally:
        loader_event.set()
        loader.join(10)


@attr.s(auto_attribs=True)
class GCPFrameDataGenerator:
    bucket_name: str
    sample_rate: float
    channels: typing.List[str]
    kernel_stride: float
    segment_length: float
    generation_rate: typing.Optional[float] = None
    prefix: typing.Optional[str] = None

    def __attrs_post_init__(self):
        self._last_time = time.time()
        self._frame = None
        self._idx = 0
        self._step = int(self.kernel_stride * self.sample_rate)

    def __iter__(self):
        self._q = mp.Queue(maxsize=100)
        self._stop_event = mp.Event()
        self._frame_reader = mp.Process(
            target=read_frames,
            args=(
                self._q,
                self._stop_event,
                self.bucket_name,
                self.sample_rate,
                self.channels,
                self.segment_length,
                self.prefix
            )
        )
        self._frame_reader.start()
        return self

    def __next__(self):
        self._idx += 1
        if (
            self._frame is None or
            (self._idx + 1) * self._step > self._frame.shape[1]
        ):
            frame = _eager_get(self._q, self._frame_reader)
            if self._idx * self._step < self._frame.shape[1]:
                leftover = self._frame[:, self._idx * self._step:]
                frame = np.concatenate([leftover, frame], axis=1)

            self._frame = frame
            self._idx = 0

        if self.generation_rate is not None:
            while (
                (time.time() - self._last_time) < 
                (1. / self.generation_rate - 2e-4)
            ):
                time.sleep(1e-6)

        x = self._frame[self._idx * self._step: (self._idx + 1) * self._step]
        package = Package(x=x, t0=time.time())
        self._last_time = package.t0
        return package

    def stop(self):
        self._stop_event.set()
        if self._frame_reader.is_alive():
            self._frame_reader.join(0.5)
            try:
                self._frame_reader.close()
            except ValueError:
                self._frame_reader.terminate()


if __name__ == "__main__":
    channels = """
        H1:GDS-CALIB_F_CC_NOGATE
        H1:GDS-CALIB_STATE_VECTOR
        H1:GDS-CALIB_SRC_Q_INVERSE
        H1:GDS-CALIB_KAPPA_TST_REAL_NOGATE
        H1:GDS-CALIB_F_CC
        H1:GDS-CALIB_KAPPA_TST_IMAGINARY
        H1:IDQ-PGLITCH_OVL_32_2048
        H1:GDS-CALIB_KAPPA_PU_REAL_NOGATE
        H1:IDQ-LOGLIKE_OVL_32_2048
        H1:GDS-CALIB_KAPPA_TST_REAL
        H1:ODC-MASTER_CHANNEL_OUT_DQ
        H1:GDS-CALIB_F_S_NOGATE
        H1:GDS-CALIB_KAPPA_C
        H1:IDQ-FAP_OVL_32_2048
        H1:GDS-CALIB_SRC_Q_INVERSE_NOGATE
        H1:GDS-CALIB_KAPPA_PU_REAL
        H1:IDQ-EFF_OVL_32_2048
        H1:GDS-CALIB_KAPPA_TST_IMAGINARY_NOGATE
        H1:GDS-CALIB_KAPPA_PU_IMAGINARY
        H1:GDS-CALIB_KAPPA_PU_IMAGINARY_NOGATE
        H1:GDS-GATED_DQVECTOR
        """.split("\n")
    channels = [i.strip() for i in channels if i]

    dg = GCPFrameDataGenerator(
        bucket_name="ligo-o2",
        sample_rate=4000,
        channels=channels,
        kernel_stride=0.1,
        generation_rate=1000,
        segment_length=16,
        prefix="archive/frames/O2/hoft_C02/H1/H-H1_HOFT_C02-11854/H-H1_HOFT_C02-118540"
    )

    start_time = time.time()
    try:
        dg = iter(dg)
        n = 0
        while True:
            x = next(dg)
            n += 1

            throughput = n / (time.time() - start_time)
            msg = f"Output rate: {throughput:0.1f}"
            print(msg, end="\r", flush=True)

            if n == 10000:
                break
    finally:
        dg.stop()
