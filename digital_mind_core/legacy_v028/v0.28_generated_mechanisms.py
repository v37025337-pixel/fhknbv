"""Generated and blind-validated by v0.28 Mechanism Genesis."""

def stability_switch(history, current):
    return (current if (-1 > (-(abs((history - current))))) else history)

GENERATED_MECHANISMS = {
    'stability_switch': {
        'source_sha256': 'b485b2953e5d8e483b846ca388b47cb610a7973f9701b4f9471ce4acde5d40f7',
        'cost': 10,
        'counterexamples_added': 6,
        'function': stability_switch,
    },
}
