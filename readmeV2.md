# Documentação statemachineV2 e BackgroundSubtractionV2

# StatemachineV2.py

## Defining transitions

For ease of identification in code, all transition names are preceded
with 'trigger_' prefix

### Transitions Hierarchical level 0

- 'trigger_initialize'

    Unconditional initialization, from start to operational.
    Start is a pseudo-state, waiting for the signal
    trigger_initialize, after the first image is ready for 
    analysis

- 'trigger_terminate'

    Unconditional terminate, from any state/substate

### Transitions Hierarchical level 1

- 'trigger_workplaceBlocked'
If any object is in the workplace, transition
to Error state, and reconfigure system

- 'reflexive_error'
Reflexive transition while error is not cleared

# If no contour is detected in the workplace, transition to
# configuration state to reset system settings.
['trigger_emptyWorkplace',
    'operational_error', 'operational_configuration'],

# When configuration ends, if no object is at the workplace,
# transition to Monitoring state
['trigger_workplaceReady',
    'operational_configuration', 'operational_monitoring'],

# If an object remains in the workplace after a long time,
# the Background Subtraction algorithm integrate it as background.
# In this condition, after the object is removed, it remains a 
# "shadow" in the image mask at the last location of the object.
['trigger_timeout',
    'operational_monitoring', 'operational_error'],

# At the takeImage state, the system captures and make the image
# available for transmission to the image recognition Server.
['trigger_imageSent',
    'operational_takeImage', 'operational_monitoring'],

# Transitions Hierarchical level 2
# Reflexive transition
['reflexive_workplaceFree',
    'operational_monitoring_workplaceFree', '='],

# At the monitoring state, it will capture images, and change to the
# state tracking if any object enters the scene, and keeps moving.
['trigger_movementDetected',
    'operational_monitoring_workplaceFree',
    'operational_monitoring_tracking'],

# Reflexive transition
['reflexive_tracking',
    'operational_monitoring_tracking', '='],

# At the tracking state, if no object is in the scene, fallback to
# the workplaceFree state.
['trigger_noObject',
    'operational_monitoring_tracking',
    'operational_monitoring_workplaceFree'],

# If the object is present, check if it is away from borders and if
# it has no movement in the scene at the centered state.
['trigger_objectStopped',
    'operational_monitoring_tracking',
    'operational_monitoring_centered'],

# If the object is reaching the margin region near the image borders,
# it is not considered at rest, going back to the tracking state.
['trigger_readjustPosition',
    'operational_monitoring_centered',
    'operational_monitoring_tracking'],

# The object is centered and immobile, so take the current frame and
# make it available to the image recognition service.
['trigger_imageOk',
    'operational_monitoring_centered',
    'operational_takeImage'],