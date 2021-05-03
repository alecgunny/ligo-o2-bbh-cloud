import argparse
import os

import cloud_utils as cloud
import gcp


def main(
    service_account_key_file,
    ssh_key_file,
    username,
    project,
    zone,
    cluster_name,
    num_nodes,
    gpus_per_node,
    vcpus_per_gpu,
    clients_per_node,
    generation_rate
):
    cluster_manager = cloud.GKEClusterManager(
        project=project, zone=zone, credentials=service_account_key_file
    )

    client_connection = gcp.VMConnection(username, ssh_key_file)
    client_manager = gcp.ClientVMManager(
        project=project,
        zone=zone,
        prefix="o2-client",
        service_account_key_file=service_account_key_file,
        connection=client_connection
    )

    cluster_config = cloud.container.Cluster(
        name=cluster_name,
        node_pools=[cloud.container.NodePool(
            name="default-pool",
            initial_node_count=2,
            config=cloud.container.NodeConfig()
        )]
    )
    with cluster_manager.manage_resource(cluster_config) as cluster:
        cluster.deploy_gpu_drivers()

        vcpus_per_node = vcpus_per_gpu * gpus_per_node
        node_pool_config = cloud.create_gpu_node_pool_config(
            vcpus=vcpus_per_node,
            gpus=gpus_per_node,
            gpu_type="t4",
            num_initial_nodes=num_nodes
        )
        with cluster.manage_resource(node_pool_config):
            values = {
                "numGPUs": gpus_per_node,
                "tritonTag": "20.11"
            }
            deploy_file = os.path.join("apps", "deploy.yaml")

            # deploy all the triton server instances
            # on to the nodes in our node pool
            for i in range(num_nodes):
                values["name"] = f"tritonserver-{i}"
                with cloud.deploy_file(deploy_file, values=values) as f:
                    cluster.deploy(f)

                # spin up client vms for each instance
                for j in range(clients_per_node):
                    idx = i * clients_per_node + j
                    client_manager.create_instance(idx, 8)

            # now wait for all the deployments to come online
            # and grab the associated IPs of their load balancers
            server_ips = []
            for i in range(num_nodes):
                cluster.k8s_client.wait_for_deployment(name=f"tritonserver-{i}")
                ip = cluster.k8s_client.wait_for_service(name=f"tritonserver-{i}")
                server_ips.append(ip)

            # now make sure all the clients have come online
            for instance in client_manager.instances:
                instance.wait_until_ready()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service-account-key-file",
        type=str,
        required=True,
        help="Path to service account json"
    )
    parser.add_argument(
        "--ssh-key-file",
        type=str,
        required=True,
        help="Path to key for connecting to client VMs"
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Username for connecting to client VMs"
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="GCP project for billing"
    )
    parser.add_argument(
        "--zone",
        type=str,
        required=True,
        help="GCP zone for running all resources"
    )
    parser.add_argument(
        "--cluster-name",
        type=str,
        required=True,
        help="Name of GKE cluster for hosting server instances"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of server instances to leverage"
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=1,
        help="Number of GPUs per server instance"
    )
    parser.add_argument(
        "--vcpus-per-gpu",
        type=int,
        default=8,
        help="Number of VCPUs to assign per GPU on server instances"
    )
    parser.add_argument(
        "--clients-per-node",
        type=int,
        default=1,
        help="Number of client instances per server instance"
    )
    parser.add_argument(
        "--generation-rate",
        type=float,
        default=800,
        help="Rate at which each client instance sends data to server"
    )

    flags = parser.parse_args()
    main(**vars(flags))
