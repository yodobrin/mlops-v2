@description('The name for the user-assigned managed identity')
param uamiName string
param location string
param github object




resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: uamiName
  location: location
  resource federatedCred 'federatedIdentityCredentials' = {
    name: 'github'
    properties: {
      issuer: github.issuer
      audiences: [ github.audience ]
      subject: github.subject
      // description: 'The GitHub repo will sign in via a federated credential'
    }
  }
}

output principalId string = uami.properties.principalId



