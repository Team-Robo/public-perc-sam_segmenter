"""The public target_object syntax; independent of ROS and model libraries."""
import re


def parse_targets(value, *, text=True):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('target_object is required (for example "cup,bottle")')
    targets = []
    for token in value.split(','):
        token = ' '.join(token.split()).lower()
        if not token or token == 'all':
            raise ValueError('target_object must contain explicit, nonempty targets; "all" is unsupported')
        if len(token) > 256:
            raise ValueError('Each target_object prompt must be at most 256 characters')
        if text and re.fullmatch(r'[+-]?\d+(\.\d+)?', token):
            raise ValueError('Text SAM requires object names, not numeric class IDs')
        if token not in targets:
            targets.append(token)
    if len(targets) > 16:
        raise ValueError('target_object supports at most 16 distinct targets')
    return tuple(targets)
