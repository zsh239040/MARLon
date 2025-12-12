# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model a toy Capture the flag exercise

See Jupyter notebook toyctf-simulation.ipynb for an example of
game played on this simulation.
"""
from cyberbattle.simulation import model as m
from cyberbattle.simulation.model import NodeID, NodeInfo, VulnerabilityID, VulnerabilityInfo
from typing import Dict, Iterator, cast, Tuple

default_allow_rules = [
    m.FirewallRule("RDP", m.RulePermission.ALLOW),
    m.FirewallRule("SSH", m.RulePermission.ALLOW),
    m.FirewallRule("HTTPS", m.RulePermission.ALLOW),
    m.FirewallRule("HTTP", m.RulePermission.ALLOW)]

# Network nodes involved in the Capture the flag game
nodes = {
    "Website": m.NodeInfo(
        services=[m.ListeningService("HTTPS"),
                  m.ListeningService("SSH", allowedCredentials=[
                      "ReusedMySqlCred-web"])],
        firewall=m.FirewallConfiguration(incoming=default_allow_rules,
                                         outgoing=default_allow_rules + [
                                             m.FirewallRule("su", m.RulePermission.ALLOW),
                                             m.FirewallRule("sudo", m.RulePermission.ALLOW)]),
        value=100,
        # If can SSH into server then gets FLAG "Shared credentials with
        # database user"
        properties=["MySql", "Ubuntu", "nginx/1.10.3"],
        owned_string="FLAG: Login using insecure SSH user/password",
        vulnerabilities=dict(
            ScanPageContent=m.VulnerabilityInfo(
                description="LeakedGitHubProjectUrl: Website page content shows a link to GitHub "
                            "repo",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["GitHubProject"]),
                reward_string="WEBSITE page content has a link to github -> Github project discovered!",
                cost=1.0
            ),
            ScanPageSource=m.VulnerabilityInfo(
                description="Website page source contains refrence to browseable "
                            "relative web directory",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["Website.Directory"]),
                reward_string="Viewing the web page source reveals a URL to a .txt file and directory on the website",
                cost=1.0
            ),
            CredScanBashHistory=m.VulnerabilityInfo(
                description="bash history leaking creds - FLAG Stealing "
                            "credentials for the monitoring user",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="Website[user=monitor]", port="SSH",
                                       credential="monitorBashCreds")]),
                reward_string="FLAG: SSH history revealed credentials for the monitoring user (monitor)",
                cost=1.0
            ))),

    "Website.Directory": m.NodeInfo(
        services=[m.ListeningService("HTTPS")],
        value=50,
        properties=["Ubuntu", "nginx/1.10.3",
                    "CTFFLAG:Readme.txt-Discover secret data"
                    ],
        vulnerabilities=dict(
            NavigateWebDirectoryFurther=m.VulnerabilityInfo(
                description="Discover MYSQL credentials MySql for user "
                            "'web' in (getting-started.txt)",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="Website", port="MySQL",
                                       credential="ReusedMySqlCred-web")]),
                reward_string="Discover browseable web directory: Navigating to parent URL revealed file `readme.txt`"
                              "with secret data (aflag); and `getting-started.txt` with MYSQL credentials",
                cost=1.0
            ),
            NavigateWebDirectory=m.VulnerabilityInfo(
                description="Discover URL to external sharepoint website "
                            "(in deprecation-checklist.txt)",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["Sharepoint"]),
                reward_string="Navigating to parent URL revealed file `deprecation-checklist.txt` containing"
                              "a URL to an external sharepoint website",
                cost=1.0
            )
        )),

    "Website[user=monitor]": m.NodeInfo(
        services=[m.ListeningService("SSH", allowedCredentials=[]),
                  m.ListeningService("SSH-key", allowedCredentials=["unkownkey"]),
                  m.ListeningService("su", allowedCredentials=["monitorBashCreds"])],
        value=100,
        properties=["MySql", "Ubuntu", "nginx/1.10.3"],
        owned_string="FLAG User escalation by stealing credentials from bash history",
        firewall=m.FirewallConfiguration(
            outgoing=default_allow_rules,
            incoming=[m.FirewallRule("SSH", m.RulePermission.BLOCK,
                                     reason="password authentication disabled! SSH needs private key to authenticate."),
                      m.FirewallRule("sudo", m.RulePermission.BLOCK,
                                     reason="`sudo -u monitor` failed. User 'monitor' not sudoable."
                                            "This warning will be reported!"),
                      m.FirewallRule("su", m.RulePermission.ALLOW)] + default_allow_rules
        ),
        vulnerabilities={
            "CredScan-HomeDirectory":
                m.VulnerabilityInfo(
                    description="azurecredential.txt file in home directory",
                    type=m.VulnerabilityType.LOCAL,
                    outcome=m.LeakedCredentials(credentials=[
                        m.CachedCredential(
                                node="AzureResourceManager[user=monitor]",
                                port="HTTPS",
                                credential="azuread_user_credentials")]),
                    reward_string="SSH: cat ~/azurecreds.txt (running as monitor) revealed Azure user credential!",
                    cost=1.0),
        }),

    "GitHubProject": m.NodeInfo(
        services=[m.ListeningService("GIT")],
        value=10,
        properties=["GitHub", "SasUrlInCommit"],
        vulnerabilities=dict(
            CredScanGitHistory=m.VulnerabilityInfo(
                description="Some secure access token (SAS) leaked in a "
                "reverted git commit",
                type=m.VulnerabilityType.REMOTE,
                precondition=m.Precondition('SasUrlInCommit&GitHub'),
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="AzureStorage",
                                       port="HTTPS",
                                       credential="SASTOKEN1")]),
                rates=m.Rates(probingDetectionRate=0.0,
                              exploitDetectionRate=0.0,
                              successRate=1.0),
                reward_string="CredScan success: Some secure access token (SAS) was leaked in a reverted git commit",
                cost=1.0
            ))),

    "AzureStorage": m.NodeInfo(
        services=[
            m.ListeningService("HTTPS", allowedCredentials=["SASTOKEN1"])],
        value=50,
        properties=["CTFFLAG:LeakedCustomerData"],
        vulnerabilities=dict(
            AccessDataWithSASToken=m.VulnerabilityInfo(
                description="Stealing secrets using a publicly shared "
                            "SAS token",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.CustomerData(),
                rates=m.Rates(successRate=1.0),
                reward_string="Stole data using a publicly shared SAS token",
                cost=1.0
            )
        )),

    'Sharepoint': m.NodeInfo(
        services=[m.ListeningService("HTTPS")],
        value=100,
        properties=["SharepointLeakingPassword"],
        firewall=m.FirewallConfiguration(incoming=[m.FirewallRule("SSH", m.RulePermission.ALLOW),
                                                   m.FirewallRule("HTTP", m.RulePermission.ALLOW),
                                                   m.FirewallRule("HTTPS", m.RulePermission.ALLOW)],
                                         outgoing=[]),
        vulnerabilities=dict(
            ScanSharepointParentDirectory=m.VulnerabilityInfo(
                description="Navigate to SharePoint site, browse parent "
                            "directory",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="AzureResourceManager",
                                       port="HTTPS",
                                       credential="ADPrincipalCreds")]),
                rates=m.Rates(successRate=1.0),
                reward_string="Navigating to the Sharepoint site revealed  AD Service Principal Credentials",
                cost=1.0)
        )),

    'AzureResourceManager': m.NodeInfo(
        services=[m.ListeningService("HTTPS", allowedCredentials=["ADPrincipalCreds", "azuread_user_credentials"])],
        owned_string="FLAG: Shared credentials with database user - Obtained secrets hidden in Azure Managed Resources",
        value=50,
        properties=["CTFFLAG:LeakedCustomerData2"],
        vulnerabilities=dict(
            ListAzureResources=m.VulnerabilityInfo(
                description="AzureVM info, including public IP address",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["AzureVM"]),
                reward_string="Obtained Azure VM and public IP information",
                cost=1.0
            ))),

    'AzureResourceManager[user=monitor]': m.NodeInfo(
        services=[m.ListeningService("HTTPS", allowedCredentials=["azuread_user_credentials"])],
        owned_string="More secrets stolen when logged as interactive `monitor` user in Azure with `az`",
        value=50,
        properties=[],
    ),

    'AzureVM': m.NodeInfo(
        services=[m.ListeningService("PING"),
                  m.ListeningService("SSH")],
        value=100,
        properties=["CTFFLAG:VMPRIVATEINFO"],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("SSH", m.RulePermission.BLOCK,
                                     reason="internet incoming traffic blocked on the VM by NSG firewall")],
            outgoing=[])),

    'client': m.NodeInfo(
        services=[],
        value=0,
        vulnerabilities=dict(
            SearchEdgeHistory=m.VulnerabilityInfo(
                description="Search web history for list of accessed websites",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(["Website"]),
                reward_string="Web browser history revealed website URL of interest",
                cost=1.0
            )),
        agent_installed=True,
        reimagable=False),
    
    'DatabaseServer': m.NodeInfo(
        services=[m.ListeningService("MySQL", allowedCredentials=["ReusedMySqlCred-web"])],
        value=150,
        properties=["MySQL", "Ubuntu"],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("MySQL", m.RulePermission.ALLOW)],
            outgoing=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)]
        ),
        vulnerabilities=dict(
            SQLInjection=m.VulnerabilityInfo(
                description="SQL Injection vulnerability in web application",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="DatabaseServer", port="MySQL",
                                       credential="dbadmin")]),
                reward_string="SQL Injection successful: Obtained database admin credentials",
                cost=1.5
            ),
            WeakPassword=m.VulnerabilityInfo(
                description="Weak password for database admin",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="DatabaseServer", port="MySQL",
                                       credential="weakpassword")]),
                reward_string="Weak password attack successful: Obtained database admin credentials",
                cost=1.0
            ),
            SQLInjection=m.VulnerabilityInfo(
                description="SQL Injection vulnerability in web application",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["AnalyticsServer"]),
                reward_string="SQL Injection exposed analytics server connection",
                cost=1.5
            )
        )
    ),

    'InternalServer': m.NodeInfo(
        services=[m.ListeningService("SSH", allowedCredentials=["internalSSHKey"])],
        value=200,
        properties=["Ubuntu", "Internal"],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("SSH", m.RulePermission.ALLOW)],
            outgoing=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)]
        ),
        vulnerabilities=dict(
            InternalScan=m.VulnerabilityInfo(
                description="Internal network scan reveals additional nodes",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(["DatabaseServer", "BackupServer", "DevServer", "MailServer"]),
                reward_string="Internal scan successful: Discovered additional internal nodes",
                cost=1.0
            ),
            PrivilegeEscalation=m.VulnerabilityInfo(
                description="Privilege escalation vulnerability",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="InternalServer", port="SSH",
                                       credential="root")]),
                reward_string="Privilege escalation successful: Obtained root credentials",
                cost=1.5
            )
        )
    ),

    'BackupServer': m.NodeInfo(
        services=[m.ListeningService("SSH", allowedCredentials=["backupSSHKey"]),
                  m.ListeningService("NFS")],
        value=120,
        properties=["Ubuntu", "Backup"],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("SSH", m.RulePermission.ALLOW),
                      m.FirewallRule("NFS", m.RulePermission.ALLOW)],
            outgoing=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)]
        ),
        vulnerabilities=dict(
            BackupDataLeak=m.VulnerabilityInfo(
                description="Backup data accessible via NFS",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.CustomerData(),
                reward_string="Accessed backup data via NFS",
                cost=1.0
            ),
            WeakNFSConfig=m.VulnerabilityInfo(
                description="Weak NFS configuration allows unauthorized access",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="BackupServer", port="NFS",
                                       credential="backupUser")]),
                reward_string="Weak NFS configuration: Obtained backup user credentials",
                cost=1.0
            ),
            BackupDataLeak=m.VulnerabilityInfo(
                description="Backup data accessible via NFS",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["FileServer", "AnalyticsServer"]),
                reward_string="Backup server scan revealed additional server configurations",
                cost=1.0
            )
        )
    ),

    'DevServer': m.NodeInfo(
        services=[m.ListeningService("SSH", allowedCredentials=["devSSHKey"]),
                  m.ListeningService("Jenkins")],
        value=180,
        properties=["Ubuntu", "Development"],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("SSH", m.RulePermission.ALLOW),
                      m.FirewallRule("Jenkins", m.RulePermission.ALLOW)],
            outgoing=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)]
        ),
        vulnerabilities=dict(
            JenkinsExploit=m.VulnerabilityInfo(
                description="Jenkins exploit allows remote code execution",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="DevServer", port="Jenkins",
                                       credential="jenkinsAdmin")]),
                reward_string="Jenkins exploit successful: Obtained Jenkins admin credentials",
                cost=1.5
            ),
            DevKeyLeak=m.VulnerabilityInfo(
                description="Leaked SSH key for development server",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="DevServer", port="SSH",
                                       credential="devSSHKey")]),
                reward_string="Leaked SSH key: Obtained SSH key for development server",
                cost=1.0
            ),
            JenkinsExploit=m.VulnerabilityInfo(
                description="Jenkins exploit allows remote code execution",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["FileServer"]),
                reward_string="Jenkins build logs revealed file server details",
                cost=1.5
            )
        )
    ),

    'MailServer': m.NodeInfo(
        services=[m.ListeningService("SMTP"),
                  m.ListeningService("IMAP")],
        value=140,
        properties=["Ubuntu", "Mail"],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("SMTP", m.RulePermission.ALLOW),
                      m.FirewallRule("IMAP", m.RulePermission.ALLOW)],
            outgoing=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)]
        ),
        vulnerabilities=dict(
            MailCredentialLeak=m.VulnerabilityInfo(
                description="Mail server credential leak via IMAP",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="MailServer", port="IMAP",
                                       credential="mailUser")]),
                reward_string="Mail server credential leak: Obtained mail user credentials",
                cost=1.0
            ),
            SMTPExploit=m.VulnerabilityInfo(
                description="SMTP exploit allows unauthorized email sending",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.CustomerData(),
                reward_string="SMTP exploit successful: Sent unauthorized emails",
                cost=1.0
            ),
            MailCredentialLeak=m.VulnerabilityInfo(
                description="Mail server credential leak via IMAP",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["DevServer"]),
                reward_string="Email containing development server credentials found",
                cost=1.0
            )
        )
    ),

    'FileServer': m.NodeInfo(
        services=[m.ListeningService("SMB")],
        value=130,
        properties=["Windows", "File"],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("SMB", m.RulePermission.ALLOW)],
            outgoing=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)]
        ),
        vulnerabilities=dict(
            SMBExploit=m.VulnerabilityInfo(
                description="SMB exploit allows unauthorized file access",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.CustomerData(),
                reward_string="SMB exploit successful: Accessed unauthorized files",
                cost=1.0
            ),
            WeakSMBPassword=m.VulnerabilityInfo(
                description="Weak password for SMB user",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="FileServer", port="SMB",
                                       credential="smbUser")]),
                reward_string="Weak password attack successful: Obtained SMB user credentials",
                cost=1.0
            ),
            SMBExploit=m.VulnerabilityInfo(
                description="SMB exploit allows unauthorized file access",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["AnalyticsServer", "MailServer"]),
                reward_string="File share contains connection details to other servers",
                cost=1.0
            )
        )
    ),

    'AnalyticsServer': m.NodeInfo(
        services=[m.ListeningService("SSH", allowedCredentials=["analyticsSSHKey"]),
                  m.ListeningService("Hadoop")],
        value=160,
        properties=["Ubuntu", "Analytics"],
        firewall=m.FirewallConfiguration(
            incoming=[m.FirewallRule("SSH", m.RulePermission.ALLOW),
                      m.FirewallRule("Hadoop", m.RulePermission.ALLOW)],
            outgoing=[m.FirewallRule("HTTPS", m.RulePermission.ALLOW)]
        ),
        vulnerabilities=dict(
            HadoopExploit=m.VulnerabilityInfo(
                description="Hadoop exploit allows unauthorized data access",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.CustomerData(),
                reward_string="Hadoop exploit successful: Accessed unauthorized data",
                cost=1.5
            ),
            AnalyticsKeyLeak=m.VulnerabilityInfo(
                description="Leaked SSH key for analytics server",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="AnalyticsServer", port="SSH",
                                       credential="analyticsSSHKey")]),
                reward_string="Leaked SSH key: Obtained SSH key for analytics server",
                cost=1.0
            ),
            HadoopExploit=m.VulnerabilityInfo(
                description="Hadoop exploit allows unauthorized data access",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["DevServer", "BackupServer"]),
                reward_string="Analytics logs revealed connections to other servers",
                cost=1.5
            )
        )
    ),
}

global_vulnerability_library: Dict[VulnerabilityID, VulnerabilityInfo] = dict([])

# Environment constants
ENV_IDENTIFIERS = m.infer_constants_from_nodes(
    cast(Iterator[Tuple[NodeID, NodeInfo]], list(nodes.items())),
    global_vulnerability_library)


def new_environment() -> m.Environment:
    return m.Environment(
        network=m.create_network(nodes),
        vulnerability_library=global_vulnerability_library,
        identifiers=ENV_IDENTIFIERS
    )
