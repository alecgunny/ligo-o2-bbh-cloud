import os
import re
import time
import typing

import cloud_utils as cloud
import typeo
from google.cloud import container_v1 as container
from google.cloud.storage import Client as StorageClient

import gcp
import utils


def update_model_configs(
    client: StorageClient,
    model_repo_bucket_name: str,
    streams_per_gpu: int,
    instances_per_gpu: typeo.MaybeDict(int)
):
    model_repo_bucket = client.get_bucket(model_repo_bucket_name)
    for blob in model_repo_bucket.list_blobs():
        if not blob.name.endswith("config.pbtxt"):
            continue

        model_name = blob.name.split("/")[0]
        if model_name == "gwe2e":
            continue
        elif model_name == "snapshotter":
            count = streams_per_gpu
        else:
            try:
                count = instances_per_gpu[model_name]
            except TypeError:
                count = instances_per_gpu
            except KeyError:
                raise ValueError(
                    f"Must specify number ofinstances for model {model_name}"
                )

        config_str = blob.download_as_bytes().decode()
        config_str = re.sub(
            "\n  count: [0-9]+\n", f"\n  count: {count}\n", config_str
        )

        blob_name = blob.name
        blob.delete()
        blob = model_repo_bucket.blob(blob_name)
        blob.upload_from_string(
            config_str, content_type="application/octet-stream"
        )


def configure_wait_and_run(
    cluster: cloud.gke.Cluster,
    client_manager: gcp.ClientVMManager,
    runner: utils.RunParallel,
    num_nodes: int,
    clients_per_node: int,
    blob_names: typing.List[str]
):
    utils.configure_vms_parallel(client_manager.instances)

    # now wait for all the deployments to come online
    # and grab the associated IPs of their load balancers
    server_ips = []
    for i in range(num_nodes):
        cluster.k8s_client.wait_for_deployment(name=f"tritonserver-{i}")
        ip = cluster.k8s_client.wait_for_service(name=f"tritonserver-{i}")
        for j in range(clients_per_node):
            server_ips.append(ip)

    models = [
        "gwe2e", "snapshotter", "deepclean_h", "deepclean_l", "postproc", "bbh"
    ]

    blobs_per_client = (len(blob_names) - 1) // clients_per_node + 1
    client_blobs = []
    for i in range(clients_per_node):
        blobs_i = []
        for j in range(blobs_per_client):
            idx = i * blobs_per_client + j
            try:
                blobs_i.append(blob_names[idx])
            except IndexError:
                break
        client_blobs.append(blobs_i)

    start_time = time.time()
    with utils.ServerMonitor(
        list(set(server_ips)),
        (
            f"num-nodes={num_nodes}_"
            f"clients-per-node={clients_per_node}_"
            "server-stats.csv"
        ),
        models
    ) as monitor:
        runner(client_manager.instances, client_blobs, server_ips)
        end_time = time.time()

    monitor.check()
    return end_time - start_time


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
    clients_per_node: int,
    instances_per_gpu: typeo.MaybeDict(int),
    vcpus_per_gpu: int,
    kernel_stride: float,
    generation_rate: float
):
    cluster_manager = cloud.GKEClusterManager(
        project=project, zone=zone, credentials=service_account_key_file
    )

    # first figure out how many clients we'll need based
    # on the number of files that we have to go through,
    # assuming (as we are right now) that it's one file per client
    storage_client = StorageClient(credentials=cluster_manager.client.credentials)

    data_bucket = storage_client.get_bucket(data_bucket_name)
    blobs = data_bucket.list_blobs(
        prefix="archive/frames/O2/hoft_C02/H1/H-H1_HOFT_C02-11854"
    )
    blob_names = [blob.name for blob in blobs]
    num_blobs = len(blob_names)

    # set up the Triton snapshotter config so that the
    # appropriate number of snapshot instances are available
    # on each node
    streams_per_gpu = (clients_per_node - 1) // gpus_per_node + 1
    update_model_configs(
        storage_client,
        model_repo_bucket_name,
        streams_per_gpu,
        instances_per_gpu
    )

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

            with client_manager as client_manager:
                runner = utils.RunParallel(
                    model_name="gwe2e",
                    model_version=1,
                    generation_rate=1000,
                    sequence_id=1001,
                    bucket_name=data_bucket_name,
                    kernel_stride=kernel_stride,
                    sample_rate=4000,
                    chunk_size=256
                )
                delta = configure_wait_and_run(
                    cluster,
                    client_manager,
                    runner,
                    num_nodes,
                    clients_per_node,
                    blob_names
                )
                throughput = 4096 * num_blobs / (kernel_stride * delta)
                print(
                    "Predicted on {} s worth of data in "
                    "{:0.1f} s, throughput {:0.2f}".format(
                        4096 * num_blobs, delta, throughput)
                )


if __name__ == "__main__":
    parser = typeo.make_parser(main)
    flags = parser.parse_args()
    main(**vars(flags))
