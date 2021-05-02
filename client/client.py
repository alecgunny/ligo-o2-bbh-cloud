import argparse
import inspect
import re
import time
import typing

import numpy as np
from stillwater import ExceptionWrapper, ThreadedMultiStreamInferenceClient

from frame_reader import GCPFrameDataGenerator, DualDetectorDataGenerator
from channels import channels


def main(
    url: str,
    model_name: str,
    model_version: int,
    generation_rate: float,
    sequence_id: int,
    bucket_name: str,
    kernel_stride: float,
    sample_rate: float = 4000,
    chunk_size: float = 1024,
    prefix: typing.Optional[str] = None
):
    client = ThreadedMultiStreamInferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        qps_limit=generation_rate,
        name="client"
    )

    sources = {}
    for state_name, shape in client.states.items():
        try:
            name = re.search("(?<=deepclean_)[hl]", state_name).group(0)
            if name == "h":
                name = "hanford"
            else:
                name = "livingston"
        except AttributeError:
            # not for deepclean, so this is what we'll name
            # the dualdetector generator for the strain channel
            sources["name"] = state_name

        sources[name] = GCPFrameDataGenerator(
            bucket_name=bucket_name,
            sample_rate=sample_rate,
            channels=channels[name],
            kernel_stride=kernel_stride,
            chunk_size=chunk_size,
            generation_rate=generation_rate,
            prefix=prefix,
            name=state_name
        )
    source = DualDetectorDataGenerator(**sources)
    pipe = client.add_data_source(
        source, str(sequence_id), sequence_id
    )

    source.start()
    client.start()
    try:
        outputs = []
        while True:
            if not pipe.poll():
                continue
            package = pipe.recv()
            try:
                if isinstance(package, ExceptionWrapper):
                    package.reraise()
            except StopIteration:
                break
            outputs.append(package.x)
    finally:
        client.stop()

        source.join(10)
        client.join(10)
        try:
            client.close()
        except ValueError:
            client.terminate()
            time.sleep(0.1)
            client.close()

        outputs = np.array(outputs)
        np.save("outputs.npy", outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name, param in inspect.signature(main).parameters.items():
        annotation = param.annotation
        try:
            type_ = annotation.__args__[0]
        except AttributeError:
            type_ = annotation

        name = name.replace("_", "-")
        if param.default is inspect._empty:
            parser.add_argument(
                f"--{name}",
                type=type_,
                required=True
            )
        else:
            parser.add_argument(
                f"--{name}",
                type=type_,
                default=param.default
            )

    flags = parser.parse_args()
    main(**vars(flags))
