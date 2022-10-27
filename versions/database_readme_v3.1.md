# ASN Database
Version: v3.1

[![DOI:10.5281/zenodo.7257829](https://zenodo.org/badge/doi/10.5281/zenodo.4018965.svg)](https://doi.org/10.5281/zenodo.7257829)

## Description
* Room:
    * Humans: None
* Sources:
    * Directivity: Omnidirectional
    * Frequency response: Flat
* Receiver:
    * Directivity: Omnidirectional
    * Frequency response: Flat

### Spacing
* Distance between sources: 40 cm
* Distance between receiver nodes: 60 cm
* Radius of microphones around the node center: 5 cm
* Minimum distance from walls and key objects: 20 cm
* Number of sources: 202
* Number of nodes: 98
* Number of microphones: 392
* Number of impulse responses: 79184

### Reverberation
| Room          |      $T_{30}$       |  Floor Area  |
|----------     |:-------------:      |------:|
| Living room   | 0.46 s              | 33.09 $\text{m}^2$ |
| Bedroom       | 0.31 s              | 7.47 $\text{m}^2$  |
| Toilet        | 0.35 s              | 1.40 $\text{m}^2$  |
| Hall          | 0.41 s              | 3.69 $\text{m}^2$  |
| Bathroom      | 0.44 s              | 4.32 $\text{m}^2$  |

## CATT Simulation settings
* Scattering: Suface + edge
* Predict SxR settings:
    * 2: Detailed auralization
    * Number of cones: 15000
    * Length: Suggested
    * Air Absorption: On
    * Diffraction: 
        * 1st order [d] (All possible source-Edge Source-receiver combinations)
    * B-format order: 1st

### Simulation Details
* Sample rate: 44100 Hz
