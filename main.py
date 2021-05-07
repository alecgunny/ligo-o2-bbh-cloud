import argparse
import inspect
import os
import re
import time

import cloud_utils as cloud
from google.cloud import container_v1 as container
from google.cloud.storage import Client as StorageClient

import gcp
import utils


def main(
    service_account_key_file: str,
    ssh_key_file: str,
    username: str,
    project: str,
    zone: str,
    cluster_name: str,
    data_bucket_name: str,
    model_repo_bucket_name: str,
    num_nodes: int,
    gpus_per_node: int,
    instances_per_gpu: int,
    vcpus_per_gpu: int,
    kernel_stride: float,
    generation_rate: float
):
    cluster_manager = cloud.GKEClusterManager(
        project=project, zone=zone, credentials=service_account_key_file
    )

    # first figure out how many clients we'll need based
    # on the number of files that we have to go through
    storage_client = StorageClient.from_service_account_json(
        service_account_key_file
    )
    data_bucket = storage_client.get_bucket(data_bucket_name)
    blobs = data_bucket.list_blobs(
        prefix="archive/frames/O2/hoft_C02/H1/H-H1_HOFT_C02-11854"
    )
    blob_names = [blob.name for blob in blobs]
    num_blobs = len(blob_names)
    clients_per_node = (num_blobs - 1) // num_nodes + 1

    # set up the Triton snapshotter config so that the
    # appropriate number of snapshot instances are available
    # on each node
    streams_per_gpu = (clients_per_node - 1) // gpus_per_node + 1
    model_repo_bucket = storage_client.get_bucket(model_repo_bucket_name)
    model_prefix = f"kernel-stride-{kernel_stride}_"
    for blob in model_repo_bucket.list_blobs(prefix=model_prefix):
        if not blob.name.endswith("config.pbtxt"):
            continue

        model_name = re.search(
            f"(?<={model_prefix}).+(?=/config.pbtxt)", blob.name
        ).group(0)
        if model_name == "snapshotter":
            count = streams_per_gpu
        else:
            count = instances_per_gpu

        config_str = blob.download_as_bytes().decode()
        config_str = re.sub(
            "\n  count: [0-9]+\n", f"\n  count: {count}\n", config_str
        )

        blob_name = blob.name
        blob.delete()
        blob = model_repo_bucket.blob(blob_name)
        blob.upload_from_string(config_str, content_type="application/octet-stream")

    client_connection = gcp.VMConnection(username, ssh_key_file)
    client_manager = gcp.ClientVMManager(
        project=project,
        zone=zone,
        prefix="o2-client",
        service_account_key_file=service_account_key_file,
        connection=client_connection
    )

    cluster_resource = container.Cluster(
        name=cluster_name,
        node_pools=[container.NodePool(
            name="default-pool",
            initial_node_count=2,
            config=container.NodeConfig()
        )]
    )
    with cluster_manager.manage_resource(cluster_resource) as cluster:
        cluster.deploy_gpu_drivers()

        vcpus_per_node = vcpus_per_gpu * gpus_per_node
        node_pool_config = cloud.create_gpu_node_pool_config(
            vcpus=vcpus_per_node,
            gpus=gpus_per_node,
            gpu_type="t4",
        )
        node_pool_resource = container.NodePool(
            name="tritonserver-t4-pool",
            initial_node_count=num_nodes,
            config=node_pool_config
        )
        with cluster.manage_resource(node_pool_resource):
            values = {
                "gpus": gpus_per_node,
                "tag": "20.11",
                "vcpus": vcpus_per_node - 1,
                "repo": model_repo_bucket_name,
            }
            deploy_file = os.path.join("apps", "tritonserver.yaml")

            # deploy all the triton server instances
            # on to the nodes in our node pool
            for i in range(num_nodes):
                values["name"] = f"tritonserver-{i}"
                with cloud.deploy_file(deploy_file, values=values) as f:
                    cluster.deploy(f)

                # spin up client vms for each instance
                for j in range(clients_per_node):
                    idx = i * clients_per_node + j
                    if idx < num_blobs:
                        client_manager.create_instance(idx, 8)
            utils.configure_vms_parallel(client_manager.instances)

            # now wait for all the deployments to come online
            # and grab the associated IPs of their load balancers
            server_ips = []
            for i in range(num_nodes):
                cluster.k8s_client.wait_for_deployment(name=f"tritonserver-{i}")
                ip = cluster.k8s_client.wait_for_service(name=f"tritonserver-{i}")
                for j in range(clients_per_node):
                    server_ips.append(ip)

            runner = utils.RunParallel(
                model_name=f"kernel-stride-{kernel_stride}_gwe2e",
                model_version=1,
                generation_rate=1000,
                sequence_id=1001,
                bucket_name=data_bucket_name,
                kernel_stride=kernel_stride,
                sample_rate=4000,
                chunk_size=256
            )

            start_time = time.time()
            runner(client_manager.instances, blob_names, server_ips)
            end_time = time.time()

            delta = end_time - start_time
            throughput = 4096 * num_blobs / (kernel_stride * delta)
            print(
                "Predicted on {} s worth of data in "
                "{:0.1f} s, throughput {0.2f}".format(
                    4096 * num_blobs, delta, throughput)
            )


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
