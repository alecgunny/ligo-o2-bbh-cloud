import os
import pickle
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
    run_config: utils.RunConfig
):
    model_repo_bucket = client.get_bucket(model_repo_bucket_name)
    for blob in model_repo_bucket.list_blobs():
        # only updating config protobufs
        if not blob.name.endswith("config.pbtxt"):
            continue

        model_name = blob.name.split("/")[0]
        if model_name == "gwe2e":
            # ensemble model doesn't use instance groups
            continue
        count = run_config.instances_per_gpu[model_name]

        # replace the instance group count
        # in the config protobuf
        config_str = blob.download_as_bytes().decode()
        config_str = re.sub(
            "\n  count: [0-9]+\n", f"\n  count: {count}\n", config_str
        )

        # delete the existing blob and
        # replace it with the updated config
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
    run_config: utils.RunConfig,
    blob_names: typing.List[str]
):
    utils.configure_vms_parallel(client_manager.instances)

    # now wait for all the deployments to come online
    # and grab the associated IPs of their load balancers
    unique_ips = []
    for i in range(run_config.num_nodes):
        cluster.k8s_client.wait_for_deployment(name=f"tritonserver-{i}")
        ip = cluster.k8s_client.wait_for_service(name=f"tritonserver-{i}")
        unique_ips.append(ip)

    # assign blobs to clients in a manner such that the
    # number of blobs per client is as even as possible,
    # and that all clients are given sequential blobs
    # e.g. blobs [a b c d e f g] for 3 clients would produce
    # client_blobs = [[a, b, c], [d, e], [f, g]]
    blobs_per_client, remainder_blobs = divmod(
        len(blob_names), run_config.total_clients
    )
    client_blobs, idx = [], 0
    for i in range(run_config.total_clients):
        if i < remainder_blobs:
            num_blobs = blobs_per_client + 1
        else:
            num_blobs = blobs_per_client

        client_blobs.append(blob_names[idx: idx + num_blobs])
        idx += num_blobs

    # establish some parameters for our server monitor
    with open(os.path.join(runner.output_dir, "config.pkl"), "wb") as f:
        pickle.dump(run_config, f)
    fname = os.path.join(runner.output_dir, "server-stats.csv")

    # assign ips to clients in a similar manner as above, only don't
    # worry about sequential ordering because it doesn't matter which
    # server a client makes requests to. Also don't care about the
    # length going a little long since we zip in `runner` anyway
    client_ips = [
        ip for j in range(run_config.clients_per_node) for ip in unique_ips
    ]

    # run the clients while monitoring the server,
    # keep track of how much time it takes us
    start_time = time.time()
    with utils.ServerMonitor(unique_ips, fname) as monitor:
        runner(client_manager.instances, client_blobs, client_ips)
        end_time = time.time()

    # check to make sure the monitor didn't encounter
    # any errors and return the time delta for the run
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
    run_config = utils.RunConfig(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        clients_per_node=clients_per_node,
        instances_per_gpu=instances_per_gpu,
        vcpus_per_gpu=vcpus_per_gpu,
        kernel_stride=kernel_stride,
        generation_rate=generation_rate
    )
    output_dir = os.path.join("outputs", "run-" + run_config.id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # instantiate our cluster manager up front to establish
    # some credentials that we'll use for other objects
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
    update_model_configs(
        storage_client,
        model_repo_bucket_name,
        run_config
    )

    # set up a VM manager with a connection we can
    # use to execute commands over SSH and copy output
    # from the run via SCP
    client_connection = gcp.VMConnection(username, ssh_key_file)
    client_manager = gcp.ClientVMManager(
        project=project,
        zone=zone,
        prefix="o2-client",
        service_account_key_file=service_account_key_file,
        connection=client_connection
    )

    # describe our cluster as a resource, then use the
    # manager to manage it in a context that will
    # delete the resource once we're done with it
    cluster_resource = container.Cluster(
        name=cluster_name,
        node_pools=[container.NodePool(
            name="default-pool",
            initial_node_count=2,
            config=container.NodeConfig()
        )]
    )
    with cluster_manager.manage_resource(cluster_resource) as cluster:
        # deploy the nvidia driver daemonset to the cluster
        # so that our GPU node pool will be ready to go
        cluster.deploy_gpu_drivers()

        # describe a node pool resource then manage it with our
        # cluster object in the same manner
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
            # values to fill in wild cards on tritonserver deploy yaml
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

            # another cheap context to make sure our
            # VMs delete once we're done with our run
            with client_manager as client_manager:
                # baseline config that will be
                # shared between all the VM jobs
                runner = utils.RunParallel(
                    model_name="gwe2e",
                    model_version=1,
                    generation_rate=generation_rate,
                    sequence_id=1001,
                    bucket_name=data_bucket_name,
                    kernel_stride=kernel_stride,
                    sample_rate=4000,
                    chunk_size=256,
                    output_dir=output_dir
                )

                # run the clients on the VM and get the
                # time delta on our side
                delta = configure_wait_and_run(
                    cluster,
                    client_manager,
                    runner,
                    run_config,
                    blob_names
                )

                # report some of the metrics then exit all
                # our contexts to destroy the resources we
                # spun up
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
