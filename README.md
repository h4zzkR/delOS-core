# delOS AI prealpha system
Attempts to build conceptual chat and voice assistant.

# Intents detection (turnLightOn and turnLightOff example)
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
- Added ability to run encoder module on server with CUDA [02.01.21]
- refactoring and moving to pytorch [31.12.20 - happy new year!]
- Added ability to learn from custom created yaml datasets with small count of examples
- Divided theese models for two separete models, will use a lot of taggers for a lot of intents
- Added first code for NLU: Intent detection and tags extraction
- Init
