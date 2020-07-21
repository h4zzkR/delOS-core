# delOS AI prealpha system
Attempts to build conceptual chat and voice assistant.

# Intents detection (turnLightOn and turnLightOff)
`> turn off light
< turnLightOff
> disable light in the garden
< turnLightOff
> light on in the kitchen
< turnLightOn
> enable illumintation
< turnLightOn
> switch off light
< turnLightOff`

# Changelog
- Added ability to learn from custom created yaml datasets with small count of examples
- Divided theese models for two separete models, will use a lot of taggers for a lot of intents
- Added first code for NLU: Intent detection and tags extraction
- Init
