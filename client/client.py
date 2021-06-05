import logging
import re
import time
import typing

import numpy as np
import typeo
from stillwater import ExceptionWrapper, StreamingInferenceClient

from frame_reader import GCPFrameDataGenerator, DualDetectorDataGenerator
from channels import channels


def main(
    url: str,
    model_name: str,
    model_version: int,
    generation_rate: float,
    sequence_id: int,
    bucket_name: str,
    t0: float,
    length: float,
    kernel_stride: float,
    sample_rate: float = 4000,
    chunk_size: float = 1024,
    start_time: typing.Optional[float] = None
):
    client = StreamingInferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        qps_limit=generation_rate,
        name="client"
    )

    # create sources for each one of the detectors
    sources = {}
    for state_name, shape in client.states.items():
        try:
            # get the detector name from the state name
            name = re.search("(?<=deepclean_)[hl]", state_name).group(0)
            name = "hanford" if name == "h" else "livingston"
        except AttributeError:
            # not for deepclean, so this is what we'll name
            # the dualdetector generator for the strain channel
            sources["name"] = state_name
            continue

        sources[name] = GCPFrameDataGenerator(
            bucket_name=bucket_name,
            t0=t0,
            length=length,
            sample_rate=sample_rate,
            channels=channels[name],
            kernel_stride=kernel_stride,
            chunk_size=chunk_size,
            generation_rate=generation_rate,
            name=state_name
        )

    # combine sources into one that splits out strain
    # channels from each detector and combines them
    # add this multi-data source to the client
    source = DualDetectorDataGenerator(**sources)
    pipe = client.add_data_source(source, str(sequence_id), sequence_id)

    while time.time() < start_time:
        time.sleep(1e-3)

    client.start()
    timeout, stopped = 1, False
    try:
        outputs = []
        logging.info("Starting client")
        while True:
            tick = time.time()
            while (time.time() - tick) < timeout:
                if pipe.poll():
                    package = pipe.recv()
                    break
            else:
                if stopped:
                    logging.info("Timed out, breaking")
                    break
                continue

            try:
                if isinstance(package, ExceptionWrapper):
                    package.reraise()
            except StopIteration:
                logging.info("StopIteration raised")
                stopped = True
                continue

            outputs.append(package["prob"].x[0, 0])

    finally:
        client.stop()

        for name, source in sources.items():
            if name != "name":
                source.stop()

        client.join(10)
        try:
            client.close()
        except ValueError:
            client.terminate()
            time.sleep(0.1)
            client.close()

        outputs = np.array(outputs)
        logging.info("Saving {} predictions".format(len(outputs)))
        np.save("outputs.npy", outputs)


if __name__ == "__main__":
    parser = typeo.make_parser(main)
    parser.add_argument(
        "--log-file",
        type=str,
        default="client.log"
    )
    flags = vars(parser.parse_args())
    logging.basicConfig(filename=flags.pop("log_file"), level=logging.INFO)

    main(**flags)
