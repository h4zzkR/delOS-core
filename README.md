# delOS AI prealpha system
Attempts to build conceptual chat and voice assistant.

# Intents detection (turnLightOn and turnLightOff example and others)
# Will be more intents in future with an ability to add custom data
`
> turn off light <br>
< turnLightOff <br>
> disable light in the garden <br>
< turnLightOff <br>
> light on in the kitchen <br>
< turnLightOn <br>
> enable illumintation <br>
< turnLightOn <br>
> switch off light <br>
< turnLightOff <br>
`
# Intent tags parsing

# Changelog
- refactoring and moving to pytorch
- Added ability to learn from custom created yaml datasets with small count of examples
- Divided theese models for two separete models, will use a lot of taggers for a lot of intents
- Added first code for NLU: Intent detection and tags extraction
- Init
