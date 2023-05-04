param contributerId string = 'b24988ac-6180-42a0-ab88-20f7382dd24c'

param uamiName string

param principalId string

resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  
  name: guid(contributerId, uamiName)
  properties: {
    principalId: principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', contributerId)
    principalType: 'ServicePrincipal'
  }

}
