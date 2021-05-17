import multiprocessing as mp
import os
import queue
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


def _eager_put(q, x, event):
    """
    quick wrapper to avoid waiting on
    q.put if the q is full in case we
    raise an exception from the parent
    process. Won't work if you're worried
    about race conditions but we're not
    """
    while True:
        try:
            q.put_nowait(x)
            return True
        except queue.Full:
            if event.is_set():
                return False
    return True


def _parse_blob_fname(name):
    name = name.replace(".gwf", "")
    timestamp, length = tuple(map(int, name.split("-")[-2:]))
    return timestamp, length


@_catch_wrap_and_put
def _download_and_write_frames(q, blobs, stop_event, name):
    for blob in blobs:
        if not blob.name.endswith(".gwf"):
            continue

        # check first if we even need to do this
        if stop_event.is_set():
            break
        blob_bytes = blob.download_as_bytes()

        fname = blob.name.split("/")[-1]
        timestamp, length = _parse_blob_fname(fname)

        # check again here because things could have
        # changed in the time it took us to download
        # and this could be as time consuming
        if stop_event.is_set():
            break

        fname = "-".join(name.replace("/", "-"), timestamp, length) + ".gwf"
        with open(fname, "wb") as f:
            f.write(blob_bytes)
        q.put(fname)


@_catch_wrap_and_put
def read_frames(
    q,
    stop_event,
    bucket_name,
    t0,
    length,
    sample_rate,
    channels,
    chunk_size,
    name,
    fnames=None
):
    client = storage.Client()
    try:
        bucket = client.get_bucket(bucket_name)
    except HTTPError as e:
        if e.code == 404:
            raise ValueError(f"Couldn't find bucket {bucket_name}")
        content = yaml.safe_load(e.response.content.decode("utf-8"))
        raise RuntimeError(
            f"Encountered HTTPError with code {e.code} "
            "and message: {}".format(content["error"]["message"])
        )

    # filter out the relevant blobs up front
    def _blob_filter(blob):
        fname = blob.name.replace(".gwf", "").split("/")[-1]
        split = fname.split("-")
        tstamp, frame_length = tuple(map(int, split[-2:]))
        return (tstamp + frame_length) >= t0 and tstamp < (t0 + length)

    blobs = list(filter(_blob_filter, bucket.list_blobs()))

    # run the download/write in a separate process
    loader_q = mp.Queue(maxsize=2)
    loader_event = mp.Event()
    loader = mp.Process(
        target=_download_and_write_frames,
        args=(loader_q, blobs, loader_event, name)
    )
    loader.start()
    time.sleep(0.1)

    try:
        while not stop_event.is_set():
            fname = _eager_get(loader_q, loader)
            timestamp, frame_length = _parse_blob_fname(fname)

            start = max(timestamp, t0)
            end = min(timestamp + frame_length, t0 + length)
            while start < end:
                if stop_event.is_set():
                    break

                timeseries = TimeSeriesDict.read(
                    fname,
                    channels=list(set(channels)),
                    format="gwf",
                    start=start,
                    end=min(start + chunk_size, end)
                )

                timeseries.resample(sample_rate)
                frame = np.stack(
                    [timeseries[channel].value for channel in channels]
                ).astype("float32")

                if not _eager_put(q, frame, stop_event):
                    break
                start += chunk_size
            os.remove(fname)
    finally:
        loader_event.set()
        loader.join(10)
        try:
            loader.close()
        except ValueError:
            loader.terminate()
            time.sleep(0.1)
            loader.close()


@attr.s(auto_attribs=True)
class GCPFrameDataGenerator:
    bucket_name: str
    t0: float
    length: float
    sample_rate: float
    channels: typing.List[str]
    kernel_stride: float
    chunk_size: float
    generation_rate: typing.Optional[float] = None
    name: typing.Optional[str] = None

    def __attrs_post_init__(self):
        self._last_time = time.time()
        self._frame = None
        self._idx = 0
        self._step = int(self.kernel_stride * self.sample_rate)

        self._q = mp.Queue(maxsize=100)
        self._stop_event = mp.Event()

    def __iter__(self):
        self._frame_reader = mp.Process(
            target=read_frames,
            args=(
                self._q,
                self._stop_event,
                self.bucket_name,
                self.t0,
                self.length,
                self.sample_rate,
                self.channels,
                self.chunk_size,
                self.name
            )
        )
        self._frame_reader.start()
        return self

    def __next__(self):
        self._idx += 1
        start = self._idx * self._step
        stop = (self._idx + 1) * self._step

        # if we're about to exhaust the current
        # frame, try to get another from the queue
        if self._frame is None or stop > self._frame.shape[1]:
            frame = _eager_get(self._q, self._frame_reader)

            # check if we have any data left from the old frame
            # and if so tack it to the start of the new frame
            if self._frame is not None and stop < self._frame.shape[1]:
                leftover = self._frame[:, start:]
                frame = np.concatenate([leftover, frame], axis=1)

            # reset the frame and index and update
            # the start and stop to match
            self._frame, self._idx = frame, 0
            start, stop = 0, self._step

        # pause a beat if we have a throttle
        if self.generation_rate is not None:
            while (
                (time.time() - self._last_time) < 
                (1. / self.generation_rate - 2e-4)
            ):
                time.sleep(1e-6)
            self._last_time = time.time()

        # create a package from the
        # current slice of the frame
        x = self._frame[:, start:stop]
        package = Package(x=x, t0=time.time())

        self._last_time = package.t0
        return package

    def stop(self):
        self._stop_event.set()
        try:
            if self._frame_reader.is_alive():
                self._frame_reader.join(0.5)
                try:
                    self._frame_reader.close()
                except ValueError:
                    self._frame_reader.terminate()
        except AttributeError:
            pass


@attr.s(auto_attribs=True)
class DualDetectorDataGenerator:
    hanford: GCPFrameDataGenerator
    livingston: GCPFrameDataGenerator
    name: str

    def __attrs_post_init__(self):
        self._iterators = None

    def __iter__(self):
        self._iterators = []
        for detector in [self.hanford, self.livingston]:
            self._iterators.append(iter(detector))
        return self

    def __next__(self):
        if self._iterators is None:
            raise ValueError("Iterators not initialized")

        packages = {}
        strains, t0 = [], 0
        for detector in self._iterators:
            try:
                package = next(detector)
            except StopIteration:
                self._iterators = None
                raise

            strains.append(package.x[0])
            t0 += package.t0

            package.x = package.x[1:]
            packages[detector.name] = package

        strain = np.stack(strains)
        packages[self.name] = Package(x=strain, t0=t0 / 2)
        return packages
