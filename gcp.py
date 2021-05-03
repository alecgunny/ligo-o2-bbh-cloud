import re
import yaml
import time
import typing
from contextlib import contextmanager

import attr
import paramiko
from requests import HTTPError
from scp import SCPClient

from google.auth.transport.requests import Request as AuthRequest
from google.cloud import compute_v1 as compute
from google.oauth2 import service_account

from cloud_utils.utils import wait_for


@attr.s(auto_attribs=True)
class VMConnection:
    """
    Object for managing connections to client instances
    sharing the same ssh information. Leverages contexts
    and some simple throttling to keep things a bit neat
    """
    username: str
    ssh_key_file: str
    sleep_between_connections: typing.Optional[float] = 1.0

    def __attrs_post_init__(self):
        self._last_time = time.time()
        self._last_ip = None

    def connection(self, instance: "ClientVMInstance"):
        if self._last_ip == instance.ip:
            while (
                (time.time() - self._last_time) < self.sleep_between_connections
            ):
                time.sleep(0.01)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
        client.connect(
            username=self.username,
            hostname=instance.ip,
            key_filename=self.ssh_key_file
        )

        self._last_ip = instance.ip
        self._last_time = time.time()

        return client

    @contextmanager
    def connect(self, instance: "ClientVMInstance"):
        client = self.connection(instance)

        try:
            yield client
        finally:
            client.close()

    @contextmanager
    def scp(self, instance: "ClientVMInstance"):
        client = self.connection(instance)
        scp_client = SCPClient(client.get_transport())

        try:
            yield scp_client
        finally:
            client.close()
            scp_client.close()


@attr.s(auto_attribs=True)
class ClientVMManager:
    """
    Somewhat silly container class for making the
    creation of new client VMs a bit simpler
    """
    project: str
    zone: str
    prefix: str
    service_account_key_file: str
    connection: VMConnection

    def __attrs_post_init__(self):
        client = compute.InstancesClient(credentials=self.credentials)
        list_request = compute.ListInstancesRequest(
            project=self.project, zone=self.zone
        )
        response = client.list(list_request)

        instances = []
        for vm_config in response.items:
            if re.fullmatch(f"{self.prefix}-[0-9]+", vm_config.name):
                ip = vm_config.network_interfaces[0].access_configs[0].nat_i_p
                instances.append(
                    ClientVMInstance(
                        name=vm_config.name, ip=ip, conn=self.connection
                    )
                )
        self.instances = sorted(
            instances, key=lambda i: int(i.name.split("-")[-1])
        )

    @property
    def credentials(self):
        credentials = service_account.Credentials.from_service_account_file(
            self.service_account_key_file,
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/compute",
                "https://www.googleapis.com/auth/devstorage.full_control",
            ]
        )
        credentials.refresh(AuthRequest())
        return credentials

    def create_instance(self, idx: int, vcpus: int):
        instance = ClientVMInstance.create(
            self.prefix + "-" + str(idx),
            self.project,
            self.zone,
            vcpus,
            self.credentials,
            self.connection
        )
        self.instances.insert(idx, instance)
        return instance

    def create_instances(self, N, vcpus):
        for _ in range(N):
            self.create_instance(vcpus)


@attr.s(auto_attribs=True)
class ClientVMInstance:
    """
    Wrapper class for creating, monitoring, and
    connecting to GCP VMs for running benchmark
    client code
    """
    name: str
    ip: str
    conn: VMConnection

    def run(self, cmd: str) -> str:
        """
        Run a command on the remote host
        """
        with self.conn.connect(self) as client:
            stdin, stdout, stderr = client.exec_command(cmd)
            return stdout.read().decode()

    def get(self, filename: str, target: typing.Optional[str] = None) -> None:
        """
        Get a file from the remote host
        """
        with self.conn.scp(self) as client:
            return client.get(filename, target or "")

    @classmethod
    def create(
        cls,
        name,
        project,
        zone,
        vcpus,
        credentials,
        connection
    ):
        with open("startup-script.sh") as f:
            startup_script = f.read()

        client = compute.InstancesClient(credentials=credentials)
        sa = compute.ServiceAccount(
            email=credentials._service_account_email,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        request = compute.InsertInstanceRequest(
            instance_resource=compute.Instance(
                name=name,
                service_accounts=[sa],
                machine_type=f"zones/{zone}/machineTypes/n1-standard-{vcpus}",
                network_interfaces=[
                    compute.NetworkInterface(
                        access_configs=[compute.AccessConfig()],
                    )
                ],
                disks=[
                    compute.AttachedDisk(
                        boot=True,
                        initialize_params=compute.AttachedDiskInitializeParams(
                            source_image=(
                                "projects/debian-cloud/global/"
                                "images/family/debian-10"
                            )
                        )
                    )
                ],
                metadata=compute.Metadata(
                    items=[
                        compute.Items(
                            key="startup-script",
                            value=startup_script
                        )
                    ]
                )
            ),
            project=project,
            zone=zone
        )

        try:
            client.insert(request)
        except HTTPError as e:
            content = yaml.safe_load(e.response.content.decode("utf-8"))
            message = content["error"]["message"]
            if message != (
                f"The resource 'projects/{project}/zones/{zone}/"
                f"instances/{name}' already exists"
            ):
                raise RuntimeError(message) from e

        list_request = compute.ListInstancesRequest(
            project=project, zone=zone, filter=f"name = {name}"
        )

        # wait until we have a valid IP address
        ip = None
        while not ip:
            try:
                vm_config = client.list(list_request).items[-1]
            except IndexError:
                raise RuntimeError(f"Instance {name} failed to create")

            ip = vm_config.network_interfaces[0].access_configs[0].nat_i_p
        return cls(name=name, ip=ip, conn=connection)

    def wait_until_ready(self, timeout=300):
        start_time = time.time()

        def _callback():
            try:
                output = self.run("cat /var/log/daemon.log")
            except paramiko.ssh_exception.NoValidConnectionsError:
                if time.time() - start_time > 60:
                    raise RuntimeError("Couldn't connect to vm instance")
                return False

            if "startup-script exit status 0" not in output:
                if time.time() - start_time > timeout:
                    raise RuntimeError("VM startup didn't complete")
                return False
            return True

        wait_for(
            _callback,
            f"Waiting for VM {self.name} to be ready at IP {self.ip}",
            f"VM {self.name} ready"
        )
